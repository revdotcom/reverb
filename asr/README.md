## Table of Contents
- [Getting Started](#getting-started)
- [About](#about)
- [Code](#code)
- [Features](#features)
- [Benchmarking](#benchmarking)
- [Acknowledgements](#acknowledgements)

## Getting Started <a name="getting-started"></a>

### Usage
```
python wenet/bin/recognize_wav.py --config model.yaml \
    --checkpoint model.pt \
    --audio hello_world.wav
```

## About <a name="about"></a>
Rev’s Reginald ASR model was trained on 200,000 hours of English speech, all expertly transcribed by humans - the largest corpus of human transcribed audio ever used to train an open-source model. The quality of this data has produced the world’s most accurate English automatic speech recognition (ASR) system, using an efficient model architecture that can be run on either CPU or GPU. Additionally, Reginald provides user control over the level of verbatimicity of the output transcript, making it ideal for both clean, readable transcription and use-cases like audio editing that require transcription of every spoken word including hesitations and re-wordings. Users can specify fully verbatim, fully non-verbatim, or anywhere in between for their transcription output.

## Code <a name="code"></a>
The folder `wenet` is taken a fork of the [WeNet](https://github.com/wenet-e2e/wenet) repository, with some modifications made for Rev-specific architecture.

The folder `wer_evaluation` contains instructions and code for running different benchmark utlities. These scripts are not specific to the Reginald architecture.

## Features <a name="features"></a>

### Transcription Style Options <a name="transcription-options"></a>
Reginald was trained to produce transcriptions in either a verbatim style, in which every word is transcribed as spoken; or a non-verbatim style, in which disfluencies may be removed from the transcript. 

Users can specify Reginald's output style with the `verbatimicity` parameter. 1 corresponds to a verbatim transcript and 0 corresponds to a non-verbatim transcript. Values between 0 and 1 are accepted and may correspond to a semi-non-verbatim style. See our demo [here](https://huggingface.co/spaces/Revai/reginald-demo) to test the `verbatimicity` parameter with your own audio.

### Decoding Options <a name="decoding-options"></a>

Reginald uses the joint CTC/attention architecture described [here](https://arxiv.org/pdf/2102.01547) and [here](https://www.rev.com/blog/speech-to-text-technology/what-makes-revs-v2-best-in-class), and supports multiple modes of decoding. Users can specify one or more modes of decoding to `recognize_wav.py` and separate output directories will be created for each decoding mode. 

Decoding options are: 
- `attention`
- `ctc_greedy_search`
- `ctc_prefix_beam_search`
- `attention_rescoring`
- `joint_decoding`

## Usage <a name="usage"></a>
```
python wenet/bin/recognize_wav.py --config model.yaml \
    --checkpoint model.pt \
    --audio hello_world.wav \
    --modes ctc_prefix_beam_search attention_rescoring \
    --gpu 0 \
    --verbatimicity 1.0
```

Or check out our demo [on HuggingFace](https://huggingface.co/spaces/Revai/reginald-demo).


## Benchmarking <a name="benchmarking"></a>
See wer_evaluation folder for details and results.

## Acknowledgments <a name="acknowledgements"></a>
Special thanks to the Wenet team for their work and for making it available under an open-source license.


