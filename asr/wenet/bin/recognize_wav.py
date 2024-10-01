# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen, Di Wu)
# Copyright (c) 2024 Rev.com (authors: Nishchal Bhandari)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import logging
import os
from itertools import groupby, chain
from pathlib import Path
from math import ceil
from typing import Generator, List
import yaml

import torch
import torch.nn.functional as F

import torchaudio
from torchaudio.compliance import kaldi

from wenet.transformer.cmvn import GlobalCMVN
from wenet.transformer.search import DecodeResult
from wenet.text.rev_bpe_tokenizer import RevBpeTokenizer
from wenet.utils.cmvn import load_cmvn
from wenet.utils.init_model import init_model
from wenet.utils.init_tokenizer import init_tokenizer
from wenet.utils.ctc_utils import get_blank_id
from wenet.bin.ctc_align import ctc_align, adjust_model_time_offset



def get_args():
    parser = argparse.ArgumentParser(
        description="Run automatic speech recognition on a given wav file using the Rev model."
    )
    parser.add_argument("--config", required=True, help="config file")
    parser.add_argument("--audio_file", required=True, help="Audio to transcribe")
    parser.add_argument(
        "--gpu", type=int, default=-1, help="gpu id for this rank, -1 for cpu"
    )
    parser.add_argument("--checkpoint", required=True, help="checkpoint model")
    parser.add_argument("--tokenizer-symbols", help="Path to tk.units.txt. Overrides the config path.")
    parser.add_argument("--bpe-path", help="Path to tk.model. Overrides the config path.")
    parser.add_argument("--cmvn-path", help="Path to cmvn. Overrides the config path.")
    parser.add_argument(
        "--beam_size", type=int, default=10, help="beam size for search"
    )
    parser.add_argument(
        "--length_penalty",
        type=float,
        default=0.0,
        help="length penalty for attention decoding and joint decoding modes",
    )
    parser.add_argument(
        "--blank_penalty", type=float, default=0.0, help="blank penalty"
    )
    parser.add_argument("--result_dir", required=True, help="asr result file")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of chunks that are decoded in parallel",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=2051,
        help="Size of each chunk that is decoded, in frames",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=[
            "attention",
            "ctc_greedy_search",
            "ctc_prefix_beam_search",
            "attention_rescoring",
            "joint_decoding",
        ],
        default=["attention_rescoring"],
        help="One or more supported decoding mode.",
    )
    parser.add_argument(
        "--ctc_weight",
        type=float,
        default=0.1,
        help="ctc weight for rescoring weight in \
                                  attention rescoring decode mode \
                              ctc weight for rescoring weight in \
                                  transducer attention rescore decode mode \
                              ctc weight for joint decoding mode",
    )

    parser.add_argument(
        "--decoding_chunk_size",
        type=int,
        default=-1,
        help="""decoding chunk size,
                                <0: for decoding, use full chunk.
                                >0: for decoding, use fixed chunk size as set.
                                0: used for training, it's prohibited here""",
    )
    parser.add_argument(
        "--num_decoding_left_chunks",
        type=int,
        default=-1,
        help="number of left chunks for decoding",
    )
    parser.add_argument(
        "--simulate_streaming", action="store_true", help="simulate streaming inference"
    )
    parser.add_argument(
        "--reverse_weight",
        type=float,
        default=0.0,
        help="""right to left weight for attention rescoring
                                decode mode""",
    )

    parser.add_argument(
        "--overwrite_cmvn",
        action="store_true",
        help="overwrite CMVN params in model with those in config file",
    )

    parser.add_argument(
        "--verbatimicity",
        type=float,
        default=1.0,
        help="The level of verbatimicity to run the mode. 0.0 would be nonverbatim, and 1.0 would be verbatim. This value gets passed to the LSL layers.",
    )

    parser.add_argument(
        "--timings_adjustment",
        type=float,
        default=230,
        help="Subtract timings_adjustment milliseconds from each timestamp")

    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Choose logging level for statistics and debugging.",
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(filename)s %(levelname)s: %(message)s",
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    with open(args.config, "r") as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    audio_file = args.audio_file
    test_conf = copy.deepcopy(configs.get("dataset_conf", {}))

    if not "filter_conf" in test_conf:
        test_conf["filter_conf"] = {}

    if "mfcc_conf" in test_conf:
        test_conf["mfcc_conf"]["dither"] = 0.0
    elif not "fbank_conf" in test_conf:
        test_conf["fbank_conf"] = {
            "num_mel_bins": 80,
            "frame_shift": 10,
            "frame_length": 25,
        }
    if "fbank_conf" in test_conf:
        test_conf["fbank_conf"]["dither"] = 0.0

    input_frame_length = test_conf["fbank_conf"]["frame_shift"]
    frame_downsampling_factor = {"linear": 1, "conv2d": 4, "conv2d6": 6, "conv2d8": 8}
    output_frame_length = input_frame_length * frame_downsampling_factor.get(
        configs["encoder_conf"]["input_layer"], 4
    )

    if args.cmvn_path:
        configs["cmvn_conf"]["cmvn_file"] = args.cmvn_path
    else:
        # Check if symbol table path is relative or absolute
        cmvn_path = Path(configs["cmvn_conf"]["cmvn_file"])
        if not cmvn_path.is_absolute():
            # Assume it's adjacent to the model
            configs["cmvn_conf"]["cmvn_file"] = (Path(args.checkpoint).parent / cmvn_path).as_posix()

    if args.tokenizer_symbols:
        configs["tokenizer_conf"]["symbol_table_path"] = args.tokenizer_symbols
    else:
        # Check if symbol table path is relative or absolute
        sym_path = Path(configs["tokenizer_conf"]["symbol_table_path"])
        if not sym_path.is_absolute():
            # Assume it's adjacent to the model
            configs["tokenizer_conf"]["symbol_table_path"] = (Path(args.checkpoint).parent / sym_path).as_posix()

    if args.bpe_path:
        configs["tokenizer_conf"]["bpe_path"] = args.bpe_path
    else:
        # Check if bpe model path is relative or absolute
        bpe_path = Path(configs["tokenizer_conf"]["bpe_path"])
        if not bpe_path.is_absolute():
            # Assume it's adjacent to the model
            configs["tokenizer_conf"]["bpe_path"] = (Path(args.checkpoint).parent / bpe_path).as_posix()
    tokenizer = init_tokenizer(configs)

    feats = compute_feats(
        audio_file,
        num_mel_bins=test_conf["fbank_conf"]["num_mel_bins"],
        frame_length=test_conf["fbank_conf"]["frame_length"],
        frame_shift=test_conf["fbank_conf"]["frame_shift"],
        dither=test_conf["fbank_conf"]["dither"],
    )  # Shape is (1, num_frames, num_mel_bins)

    # Pad and reshape into chunks and batches
    def feats_batcher(infeats, chunk_size, batch_size, device):
        batch_num_feats = chunk_size * batch_size
        num_batches = ceil(infeats.shape[1] / batch_num_feats)
        for b in range(num_batches):
            feats_batch = infeats[
                :, b * batch_num_feats : b * batch_num_feats + batch_num_feats, :
            ]
            feats_lengths = torch.tensor([chunk_size] * batch_size, dtype=torch.int32)
            if b == num_batches - 1:
                # last batch can be smaller than batch size
                last_batch_size = ceil(feats_batch.shape[1] / chunk_size)
                last_batch_num_feats = chunk_size * last_batch_size
                # Apply padding if needed
                pad_amt = last_batch_num_feats - feats_batch.shape[1]
                if pad_amt > 0:
                    feats_lengths = torch.tensor(
                        [chunk_size] * last_batch_size, dtype=torch.int32
                    )
                    feats_lengths[-1] -= pad_amt
                    feats_batch = F.pad(
                        input=feats_batch,
                        pad=(0, 0, 0, pad_amt, 0, 0),
                        mode="constant",
                        value=0,
                    )
            yield feats_batch.reshape(
                -1, chunk_size, test_conf["fbank_conf"]["num_mel_bins"]
            ), feats_lengths.to(device)

    configs["output_dim"] = len(tokenizer.symbol_table)

    # Init asr model from configs
    args.jit = False
    model, configs = init_model(args, configs)

    # from merge section (JPR)
    if args.overwrite_cmvn and (configs["cmvn_file"] is not None):
        mean, istd = load_cmvn(configs["cmvn_file"], configs["is_json_cmvn"])
        global_cmvn = GlobalCMVN(
            torch.from_numpy(mean).float(), torch.from_numpy(istd).float()
        )
        model.encoder.global_cmvn = global_cmvn

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)
    model.eval()
    feats = feats.to(device)

    _, blank_id = get_blank_id(configs, tokenizer.symbol_table)
    logging.info("blank_id is {}".format(blank_id))

    files = {}
    for mode in args.modes:
        dir_name = os.path.join(args.result_dir, mode)
        os.makedirs(dir_name, exist_ok=True)
        file_name = os.path.join(dir_name, "text")
        file_name = Path(dir_name) / (Path(audio_file).with_suffix(".ctm").name)
        files[mode] = file_name
    max_format_len = max([len(mode) for mode in args.modes])
    with torch.no_grad():
        # script argument overrides categories in shard
        cat_embs = torch.tensor([args.verbatimicity, 1.0 - args.verbatimicity]).to(
            device
        )

        results = []
        for feats_batch, feats_lengths in feats_batcher(
            feats, args.chunk_size, args.batch_size, device
        ):
            hyps = model.decode(
                    args.modes,
                    feats_batch,
                    feats_lengths,
                    args.beam_size,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    ctc_weight=args.ctc_weight,
                    simulate_streaming=args.simulate_streaming,
                    reverse_weight=args.reverse_weight,
                    context_graph=None,
                    blank_id=blank_id,
                    blank_penalty=args.blank_penalty,
                    length_penalty=args.length_penalty,
                    infos={"tasks": ["transcribe"], "langs": ["en"]},
                    cat_embs=cat_embs,
                )
            results.append(hyps)

    for mode in args.modes:
        with files[mode].open(mode="w") as fp:
            mode_hyps = chain(*(hyp[mode] for hyp in results))
            fp.write(
                "\n".join(
                    hyps_to_ctm(
                        tokenizer, Path(audio_file).name, list(mode_hyps), args.timings_adjustment, args.chunk_size, input_frame_length, output_frame_length
                    )
                )
            )


def compute_feats(
    audio_file: str,
    resample_rate: int = 16000,
    device="cpu",
    num_mel_bins=23,
    frame_length=25,
    frame_shift=10,
    dither=0.0,
) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(audio_file, normalize=False)
    logging.info(f"detected sample rate: {sample_rate}")
    waveform = waveform.to(torch.float)
    if sample_rate != resample_rate:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=resample_rate
        )(waveform)
    waveform = waveform.to(device)
    feats = kaldi.fbank(
        waveform,
        num_mel_bins=num_mel_bins,
        frame_length=frame_length,
        frame_shift=frame_shift,
        dither=dither,
        energy_floor=0.0,
        sample_frequency=resample_rate,
    )
    feats = feats.unsqueeze(0)
    return feats


def hyps_to_ctm(
    tokenizer: RevBpeTokenizer,
    audio_name: str,
    hyps: List[DecodeResult],
    timings_adjustment_ms: int,
    chunk_size: int,
    input_frame_length: int,
    output_frame_length: int
) -> Generator[str, None, None]:
    """Convert a given set of decode results for a single audio file into CTM lines."""

    times = []
    confidences = []
    time_shift_ms = 0
    for hyp in hyps:
        path = ctc_align(hyp.tokens, hyp.times, hyp.tokens_confidence,  tokenizer,
                         output_frame_length, time_shift_ms)
        path = adjust_model_time_offset(path, timings_adjustment_ms)
        time_shift_ms += chunk_size * input_frame_length
        ctm_lines = []
        for line in path:
            start_seconds = line['start_time_ms'] / 1000
            duration_seconds = line['end_time_ms'] / 1000 - start_seconds
            ctm_line = f"{audio_name} 0 {start_seconds:.2f} {duration_seconds:.2f} {line['word']} {line['confidence']:.2f}"
            yield ctm_line


if __name__ == "__main__":
    main()
