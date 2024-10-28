import copy
from functools import cached_property, partial
import logging
import os
from itertools import groupby, chain
from pathlib import Path
from math import ceil
from typing import Generator, List, Tuple
import shutil
import yaml

import torch
import torch.nn.functional as F

import torchaudio
from torchaudio.compliance import kaldi

from wenet.cli.utils import hyps_to_ctm, hyps_to_txt
from wenet.transformer.cmvn import GlobalCMVN
from wenet.transformer.search import DecodeResult
from wenet.text.rev_bpe_tokenizer import RevBpeTokenizer
from wenet.utils.cmvn import load_cmvn
from wenet.utils.init_model import init_model
from wenet.utils.init_tokenizer import init_tokenizer
from wenet.utils.ctc_utils import get_blank_id
from wenet.bin.ctc_align import ctc_align, adjust_model_time_offset

_FRAME_DOWNSAMPLING_FACTOR = {
    "linear": 1,
    "conv2d": 4,
    "conv2d6": 6,
    "conv2d8": 8,
}
CACHED_MODELS_DIR = Path.home() / ".cache/reverb"
_MODELS = {
    "reverb_asr_v1": "https://huggingface.co/Revai/reverb-asr",
}


class ReverbASR:
    def __init__(self,
        config,
        checkpoint,
        cmvn_path: str | None = None,
        tokenizer_symbols: str | None = None,
        bpe_path: str | None = None,
        gpu: int = -1,
        overwrite_cmvn: bool = False,
    ):
        self.jit = False

        use_cuda = (gpu >= 0 and torch.cuda.is_available())
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.checkpoint = checkpoint
        with open(config, "r") as fin:
            self.configs = yaml.load(fin, Loader=yaml.FullLoader)

        self.configs["cmvn_conf"]["cmvn_file"] = self._make_path_absolute(
            self.configs["cmvn_conf"]["cmvn_file"],
            cmvn_path,
        )
        self.configs["tokenizer_conf"]["symbol_table_path"] = self._make_path_absolute(
            self.configs["tokenizer_conf"]["symbol_table_path"],
            tokenizer_symbols,
        )
        self.configs["tokenizer_conf"]["bpe_path"] = self._make_path_absolute(
            self.configs["tokenizer_conf"]["bpe_path"],
            bpe_path,
        )

        self.tokenizer = init_tokenizer(self.configs)
        _, self.blank_id = get_blank_id(self.configs, self.tokenizer.symbol_table)

        self.configs["output_dim"] = len(self.tokenizer.symbol_table)

        # Init asr model from configs
        self.model, self.configs = init_model(self, self.configs)

        if overwrite_cmvn and (self.configs["cmvn_file"] is not None):
            mean, istd = load_cmvn(self.configs["cmvn_file"], self.configs["is_json_cmvn"])
            global_cmvn = GlobalCMVN(
                torch.from_numpy(mean).float(), torch.from_numpy(istd).float()
            )
            self.model.encoder.global_cmvn = global_cmvn
        self.model = self.model.to(self.device)
        self.model.eval()
        self.test_conf = self.configs['dataset_conf']
        self.input_frame_length = self.test_conf["fbank_conf"]["frame_shift"]
        self.output_frame_length = self.input_frame_length * _FRAME_DOWNSAMPLING_FACTOR.get(
            self.configs["encoder_conf"]["input_layer"], 4
        )

    def _make_path_absolute(
        self,
        config_path: str,
        alternate_path: str | None = None
    ) -> str:
        """Returns the 'alternate_path' if set, otherwise get's the absolute path todo
        'config_path'
        """
        if alternate_path:
            return alternate_path

        config_path = Path(config_path)
        if not config_path.is_absolute():
            # Assume it's adjacent to the model
            checkpoint_parent = Path(self.checkpoint).parent
            config_path = checkpoint_parent / config_path

        return config_path.as_posix()

    def compute_feats(
        self,
        audio_file: str,
        resample_rate: int = 16000,
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
        waveform = waveform.to(self.device)
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

    def feats_batcher(
        self,
        infeats: torch.Tensor,
        chunk_size: int,
        batch_size: int,
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
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
                feats_lengths = torch.tensor(
                    [chunk_size] * last_batch_size, dtype=torch.int32
                )
                # Apply padding if needed
                pad_amt = last_batch_num_feats - feats_batch.shape[1]
                if pad_amt > 0:
                    feats_lengths[-1] -= pad_amt
                    feats_batch = F.pad(
                        input=feats_batch,
                        pad=(0, 0, 0, pad_amt, 0, 0),
                        mode="constant",
                        value=0,
                    )
            yield feats_batch.reshape(
                -1, chunk_size, self.test_conf["fbank_conf"]["num_mel_bins"]
            ), feats_lengths.to(self.device)

    def transcribe_modes(
        self,
        audio_file,
        modes: List[str],
        format: str = "txt",
        verbatimicity: float = 1.0,
        chunk_size: int = 2051,
        batch_size: int = 1,
        beam_size: int = 10,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        ctc_weight: float = 0.1,
        simulate_streaming: bool = False,
        reverse_weight: float = 0.0,
        blank_penalty: float = 0.0,
        length_penalty: float = 0.0,
        timings_adjustment: float = 230
    ) -> list[str]:
        """Transcribe an audio using a list of decoding modes
        accepted by the Reverb model.

        The output will be a list of strings that map to the
        modes provided.
        """
        feats = self.compute_feats(
            audio_file,
            num_mel_bins=self.test_conf["fbank_conf"]["num_mel_bins"],
            frame_length=self.test_conf["fbank_conf"]["frame_length"],
            frame_shift=self.test_conf["fbank_conf"]["frame_shift"],
        )
        feats = feats.to(self.device)

        with torch.no_grad():
            cat_embs = torch.tensor(
                [verbatimicity, 1.0 - verbatimicity]
            ).to(self.device)

            results = []
            for feats_batch, feats_lengths in self.feats_batcher(
                feats, chunk_size, batch_size
            ):
                hyps = self.model.decode(
                    modes,
                    feats_batch,
                    feats_lengths,
                    beam_size,
                    decoding_chunk_size=decoding_chunk_size,
                    num_decoding_left_chunks=num_decoding_left_chunks,
                    ctc_weight=ctc_weight,
                    simulate_streaming=simulate_streaming,
                    reverse_weight=reverse_weight,
                    context_graph=None,
                    blank_id=self.blank_id,
                    blank_penalty=blank_penalty,
                    length_penalty=length_penalty,
                    infos={"tasks": ["transcribe"], "langs": ["en"]},
                    cat_embs=cat_embs,
                )
                results.append(hyps)

        outputs = []
        for mode in modes:
            outputs.append(get_output(
                format,
                self.tokenizer,
                Path(audio_file).name,
                list(chain(*(hyp[mode] for hyp in results))),
                timings_adjustment,
                chunk_size,
                self.input_frame_length,
                self.output_frame_length,
            ))
        return outputs

    def transcribe(
        self,
        audio_file,
        mode: str = "ctc_prefix_beam_search",
        format: str = "txt",
        verbatimicity: float = 1.0,
        chunk_size: int = 2051,
        batch_size: int = 1,
        beam_size: int = 10,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        ctc_weight: float = 0.1,
        simulate_streaming: bool = False,
        reverse_weight: float = 0.0,
        blank_penalty: float = 0.0,
        length_penalty: float = 0.0,
        timings_adjustment: float = 230
    ) -> str:
        """Transcribe an audio in one of the decoding modes
        accepted by the Reverb model.

        The output will be a string.
        """
        return self.transcribe_modes(
            audio_file,
            modes = [mode],
            format = format,
            verbatimicity = verbatimicity,
            chunk_size = chunk_size,
            batch_size = batch_size,
            beam_size = beam_size,
            decoding_chunk_size = decoding_chunk_size,
            num_decoding_left_chunks = num_decoding_left_chunks,
            ctc_weight = ctc_weight,
            simulate_streaming = simulate_streaming,
            reverse_weight = reverse_weight,
            blank_penalty = blank_penalty,
            length_penalty = length_penalty,
            timings_adjustment = timings_adjustment,
        )[0]


def get_output(
    format: str,
    tokenizer: RevBpeTokenizer,
    audio_name: str,
    hyps: List[DecodeResult],
    timings_adjustment_ms: int,
    chunk_size: int,
    input_frame_length: int,
    output_frame_length: int
) -> str:
    format_function = None
    format_delimiter = None
    if format == "txt":
        format_function =  hyps_to_txt
        format_delimiter = " "
    elif format ==  "ctm":
        format_function = partial(hyps_to_ctm, audio_name)
        format_delimiter = "\n"
    else:
        raise ValueError("Invalid output format.")

    output = []
    time_shift_ms = 0
    for hyp in hyps:
        path = ctc_align(hyp.tokens, hyp.times, hyp.tokens_confidence,  tokenizer,
                         output_frame_length, time_shift_ms)
        path = adjust_model_time_offset(path, timings_adjustment_ms)
        time_shift_ms += chunk_size * input_frame_length
        output.extend(list(format_function(path)))
    return format_delimiter.join(output)


def load_model(
    model: str
):
    """Loads a reverb model. If "model" points to a path that exists,
    tries to load a model using those files at "model".
    If not specified, downloads the latest reverb model.
    """
    if Path(model).exists():
        model_dir = Path(model)
        config_path = model_dir / "config.yaml"
        checkpoint_path = list(model_dir.glob("*.pt"))[0]
    elif model in _MODELS:
        model_dir = CACHED_MODELS_DIR / model
        config_path = model_dir / "config.yaml"
        checkpoint_path = model_dir / f"{model}.pt"
        if not (
            CACHED_MODELS_DIR.exists()
            and model_dir.exists()
            and config_path.exists()
            and checkpoint_path.exists()
        ):
            CACHED_MODELS_DIR.parent.mkdir(exist_ok=True, parents=True)
            shutil.rmtree(model_dir, ignore_errors=True)
            download_model(_MODELS[model], model_dir)
    else:
        raise ValueError(f"Please specify a local path to a model or one of our pretrained models: {','.join(get_available_models())}")

    config_path = config_path.resolve()
    checkpoint_path = checkpoint_path.resolve()
    logging.info(f"Loading the model with {config_path = } and {checkpoint_path = }")
    return ReverbASR(
        str(config_path),
        str(checkpoint_path)
    )


def get_available_models():
    return list(_MODELS.keys())


def download_model(
    url: str,
    root: str,
):
    """Clones a repository at `url` and puts
    it at `root`.
    """
    from git import Repo
    Repo.clone_from(url, root)
