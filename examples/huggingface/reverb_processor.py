import json
from typing import List, Optional, Union
import numpy as np
import sentencepiece as spm
import torch
import torchaudio
from torchaudio.compliance import kaldi
from tqdm import tqdm
from transformers import BatchFeature, PreTrainedTokenizer, ProcessorMixin, SequenceFeatureExtractor
from transformers.utils import logging


logger = logging.get_logger(__name__)


class ReverbFeatureExtractor(SequenceFeatureExtractor):
    model_input_names = ["input_features"]
    def __init__(
        self,
        feature_size=80,
        sampling_rate=16000,
        frame_length=25,
        frame_shift=10,
        chunk_length=15,
        padding_value=0.0,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=False,
            **kwargs,
        )
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.chunk_length = chunk_length
        self.max_chunk_size = 2051
        self._processor_class = "CTCWithLM"

    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        device: Optional[str] = "cpu",
        sampling_rate: Optional[int] = None,
        **kwargs,
    ) -> BatchFeature:
        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                ValueError(
                    f"The model corresponding to this feature extractor: {self.__class__.__name__} was trained using a"
                    f" sampling rate of {self.sampling_rate}. Please make sure that the provided `raw_speech` input"
                    f" was sampled with {self.sampling_rate} and not {sampling_rate}."
                    " Attempting a conversion."
                )
        else:
            logger.warning(
                "It is strongly recommended to pass the `sampling_rate` argument to this function. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        is_batched_numpy = isinstance(raw_speech, np.ndarray) and len(raw_speech.shape) > 1
        if is_batched_numpy and len(raw_speech.shape) > 2:
            raise ValueError(f"Only mono-channel audio is supported for input to {self}")
        is_batched = is_batched_numpy or (
            isinstance(raw_speech, (list, tuple)) and (isinstance(raw_speech[0], (np.ndarray, tuple, list)))
        )

        if is_batched:
            raw_speech = [np.asarray([speech], dtype=np.float32) for speech in raw_speech]
        elif not is_batched and not isinstance(raw_speech, np.ndarray):
            raw_speech = np.asarray(raw_speech, dtype=np.float32)
        elif isinstance(raw_speech, np.ndarray) and raw_speech.dtype is np.dtype(np.float64):
            raw_speech = raw_speech.astype(np.float32)

        if not is_batched:
            raw_speech = [np.asarray([raw_speech])]

        fbank_speech, feats_lengths = [], []
        for waveform in raw_speech:
            fbank_speech.append(
                kaldi.fbank(
                    torch.tensor(waveform),
                    num_mel_bins=self.feature_size,
                    frame_length=self.frame_length,
                    frame_shift=self.frame_shift,
                    dither=0.0,
                    energy_floor=0.0,
                    sample_frequency=self.sampling_rate,
                )
            )
            feats_lengths.append(fbank_speech[-1].shape[0])
        fbank_speech = BatchFeature({
            "input_features": fbank_speech,
            "feats_lengths": feats_lengths,
        })
        padded = self.pad(
            fbank_speech,
            padding="max_length",
            max_length=self.max_chunk_size,
        )
        return padded


class ReverbTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        model: str,
        #units: str,
        **kwargs,
    ):
        self.tokenizer = spm.SentencePieceProcessor(model)
        """self.units = dict()
        with open(units, 'r') as units_file:
            for line in tqdm(units_file.readlines()):
                    token, id = line.split()
                    self.units[int(id)] = token.replace('▁', ' ')"""


    def encode(
        self,
        text,
        **kwargs
    ):
        return self.tokenizer.encode(text)

    def decode(
        self,
        token_ids,
        **kwargs,
    ):
        return self.tokenizer.decode(token_ids[token_ids.nonzero()[0]].tolist())
