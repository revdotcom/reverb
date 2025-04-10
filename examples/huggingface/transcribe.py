import numpy as np
from pyctcdecode import build_ctcdecoder
import torch
import torchaudio
from transformers import pipeline
from transformers import AutoConfig, AutoModelForSpeechSeq2Seq
from reverb_hf import ReverbModel
from reverb_config import ReverbConfig
from reverb_processor import ReverbFeatureExtractor, ReverbTokenizer


AutoConfig.register("reverb_asr", ReverbConfig)
AutoModelForSpeechSeq2Seq.register(ReverbConfig, ReverbModel)
feature_extractor = ReverbFeatureExtractor(return_tensors='pt')
tokenizer = ReverbTokenizer(
    "hf-reverb/tk.model",
)
decoder_ids = []
with open("hf-reverb/tk.units.txt", 'r') as units_file:
    for line in units_file:
        token = line.split()[0]
        if len(token) == 0:
            continue
        if token == '<blank>':
            token = ''
        decoder_ids.append(token)
decoder = build_ctcdecoder(decoder_ids)

transcribe = pipeline(
    "automatic-speech-recognition",
    model="hf-reverb",
    feature_extractor=feature_extractor,
    tokenizer=tokenizer,
    framework='pt',
    device='cpu', #crucial
    decoder=decoder,
    decoder_kwargs={"beam_width": 8, "token_min_logp": -10}
)
AUDIO_PATH = ""
waveform, sample_rate = torchaudio.load(AUDIO_PATH, normalize=False)
#print(waveform)
waveform = np.array(waveform.to(torch.float).reshape(-1))

chunk_size_samples = feature_extractor.chunk_length * sample_rate
for idx in range(0,len(waveform),chunk_size_samples):
    print(transcribe(waveform[idx: idx+chunk_size_samples])['text'])
