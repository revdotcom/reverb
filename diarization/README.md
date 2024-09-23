## Rev's diarization models
This repository contains 2 new speaker diarization models built upon the
[PyAnnote](https://github.com/pyannote/pyannote-audio) framework. These models are trained and intended 
for the usage with ASR system (speaker attributed ASR). 

The smaller model - `Fico` - provides a **16.5%** relative improvement in WDER (Word Diarization Error Rate) 
compared to the baseline pyannote3.0 model, 
evaluated on over 1,250,000 tokens across five different test suites.
The larger model - `Babis` - offers **22.25%** relative improvement over pyannote3.0 model.

## Table of Contents
- [Usage](#usage)
  - [Assigning words to speakers](#assigning-words-to-speakers)
  - [Running training script](#running-training-script)
- [Results](#results)

## Usage
We recommend running on GPU. Dockerfile is CUDA ready and CUDA 12.4+ is required.
The output format is a standard RTTM stored in the output directory with `basename.rttm` format.

You can run diarization on a single audio file (or list of audio files) using the
`infer_pyannote3.0.py` script. The same approach can be used for Docker.
```bash
python infer_pyannote3.0.py /path/to/audios --out-dir /path/to/outdir
```
You can specify the model you want to run via the `--lstm-model` argument - `Revai/fico` or `Revai/babis` 


### Assigning words to speakers
It is possible to assign words to speakers if ASR was previously run.
The script `assign_words2speaker.py` takes a diarization segmentation and ASR transcription in
CTM format to output speaker assignment to tokens (words). 
```bash
python assign_words2speaker.py speaker_segments.rttm words.ctm transcript.rttm
```
The output format used is a slightly modified RTTM. Besides speaker value, start and duration, we 
store token value and token confidence in the RTTM itself.
We used `Orthography Field` (6th column) to store token value and `Confidence Score` (9th column) 
to store token confidence.

### Running training script
We do provide the training script that was used to fine-tune original pyannote3.0 model.
The training script is run as follows:
```bash
python train_pyannote3.0.py --database data/database.yaml
```
The `--database` parameter points to yaml database file; we provide an example file that is 
easy to use. You need to specify .uri, .uem and .rttm files; for a more detailed 
description please refer to pyannote documentation.


## Results
While DER is a valuable metric for assessing the technical performance of a diarization model 
in isolation, WDER is more crucial in the context of ASR because it reflects the combined 
effectiveness of both the diarization and ASR components in producing accurate, 
speaker-attributed text. In practical applications where the accuracy of both “who spoke” 
and “what was spoken” is essential, WDER provides a more meaningful and relevant measure 
for evaluating system performance and guiding improvements.
For this reason we only report WDER metrics. We also plan to add WDER into `pyannote.metrics`
codebase.

### fico
| Test suite                                                                         | WDER  |
|------------------------------------------------------------------------------------|-------|
| [earnings21](https://github.com/revdotcom/speech-datasets/tree/rttm_v1/earnings21) | 0.047 |
 | rev16 | 0.077 | 

### babis
| Test suite                                                                         | WDER  |
|------------------------------------------------------------------------------------|-------|
| [earnings21](https://github.com/revdotcom/speech-datasets/tree/rttm_v1/earnings21) | 0.046 |
| rev16 | 0.078 |
