import sys
import os

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../asr/"))
sys.path.append(project_root)

import numpy as np
import torch
import torchaudio
from transformers import pipeline
from transformers import AutoConfig, AutoModelForSpeechSeq2Seq
from reverb_hf import ReverbModel
from reverb_config import ReverbConfig
from reverb_processor import ReverbFeatureExtractor, ReverbTokenizer


# Register the custom model and config
AutoConfig.register("reverb_asr", ReverbConfig)
AutoModelForSpeechSeq2Seq.register(ReverbConfig, ReverbModel)

# Load configuration
config = ReverbConfig.from_yaml_file("reverb_config.yaml")

# Initialize feature extractor and tokenizer using config
feature_extractor = ReverbFeatureExtractor(return_tensors='pt')
tokenizer = ReverbTokenizer(config.tokenizer_path)

# Initialize model
model = ReverbModel(config)

# Initialize transcription pipeline
transcribe = pipeline(
    "automatic-speech-recognition",
    model=model,
    feature_extractor=feature_extractor,
    tokenizer=tokenizer,
    framework='pt',
    device='cpu', #crucial
    decoder=config.decoder,
    decoder_kwargs={
        "beam_width": config.decoder_beam_width,
        "token_min_logp": config.decoder_token_min_logp
    }
)

# Process audio
AUDIO_PATH = ""
waveform, sample_rate = torchaudio.load(AUDIO_PATH, normalize=False)
#print(waveform)
waveform = np.array(waveform.to(torch.float).reshape(-1))

chunk_size_samples = feature_extractor.chunk_length * sample_rate
for idx in range(0,len(waveform),chunk_size_samples):
    print(transcribe(waveform[idx: idx+chunk_size_samples])['text'])
