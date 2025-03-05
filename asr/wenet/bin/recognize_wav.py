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
from wenet.cli.reverb import load_model, ReverbASR
from wenet import get_available_models


def get_args():
    parser = argparse.ArgumentParser(
        description="Run automatic speech recognition on a given wav file using the Rev model."
    )
    parser.add_argument("--audio_file", required=True, help="Audio to transcribe")
    parser.add_argument("--config", default=None, help="Path to config file")
    parser.add_argument("--checkpoint", default=None, help="Path to Reverb model checkpoint")
    parser.add_argument("--model", default=None, help="Path to directory containing config"
                                                      " and checkpoint for a reverb model"
                                                      " or the name of a pretrained model"
                                                      f" from: {','.join(get_available_models())}")
    parser.add_argument(
        "--gpu", type=int, default=-1, help="gpu id for this rank, -1 for cpu"
    )
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

    model_arg_set = (args.model is not None)
    config_checkpoint_args_set = (args.checkpoint is not None and args.config is not None)

    if model_arg_set == config_checkpoint_args_set:
        raise RuntimeError("One of either --model or (--checkpoint and --config) must be set.")

    if model_arg_set:
        reverb_model = load_model(args.model)
    else:
        reverb_model = ReverbASR(
            args.config,
            args.checkpoint,
            cmvn_path = args.cmvn_path,
            tokenizer_symbols = args.tokenizer_symbols,
            bpe_path = args.bpe_path,
            gpu = args.gpu,
            overwrite_cmvn = args.overwrite_cmvn,
        )

    files = {}
    for mode in args.modes:
        dir_name = os.path.join(args.result_dir, mode)
        os.makedirs(dir_name, exist_ok=True)
        file_name = os.path.join(dir_name, "text")
        file_name = Path(dir_name) / (Path(args.audio_file).with_suffix(".ctm").name)
        files[mode] = file_name

    outputs = reverb_model.transcribe_modes(
        args.audio_file,
        modes = args.modes,
        format = "ctm",
        verbatimicity = args.verbatimicity,
        chunk_size = args.chunk_size,
        batch_size = args.batch_size,
        beam_size = args.beam_size,
        decoding_chunk_size = args.decoding_chunk_size,
        num_decoding_left_chunks = args.num_decoding_left_chunks,
        ctc_weight = args.ctc_weight,
        simulate_streaming = args.simulate_streaming,
        reverse_weight = args.reverse_weight,
        blank_penalty = args.blank_penalty,
        length_penalty = args.length_penalty,
        timings_adjustment = args.timings_adjustment,
    )
    for mode, out in zip(args.modes, outputs):
        with files[mode].open(mode="w") as fp:
            fp.write(out)


if __name__ == "__main__":
    main()
