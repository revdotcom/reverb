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
Rev’s ASR model was trained on 200,000 hours of English speech, all expertly transcribed by humans - the largest corpus of human transcribed audio ever used to train an open-source model. The quality of this data has produced the world’s most accurate English automatic speech recognition (ASR) system, using an efficient model architecture that can be run on either CPU or GPU. Additionally, Reginald provides user control over the level of verbatimicity of the output transcript, making it ideal for both clean, readable transcription and use-cases like audio editing that require transcription of every spoken word including hesitations and re-wordings. Users can specify fully verbatim, fully non-verbatim, or anywhere in between for their transcription output.

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

Unlike many ASR providers, Rev primarily uses long-form speech recognition corpora for benchmarking. We use each model to produce a transcript of an entire audio file, then use [fstalign](https://github.com/revdotcom/fstalign) to align and score the complete transcript. We report micro-average WER across all of the reference words in a given test suite. We have included our scoring scripts in this repository so that anyone can replicate our work, benchmark other models, or experiment with new long-form test suites. 

Here, we’ve benchmarked Rev’s model against the best performing open-source models currently available: OpenAI’s Whisper large-v3 and NVIDIA’s Canary-1B, both accessed through HuggingFace. Note that both of these models have significantly more parameters than Reginald. We use simple chunking with no overlap - 30s chunks for Whisper and Canary, and 20s chunks for Reginald. These results use CTC prefix beam search with attention rescoring. For Whisper and Canary, we use NeMo to normalize the model outputs before scoring.

For long-form ASR, we’ve used three corpora: Rev16 (podcasts), [Earnings21](https://github.com/revdotcom/speech-datasets/tree/main/earnings21) (earnings calls from US-based companies), and [Earnings22](https://github.com/revdotcom/speech-datasets/tree/main/earnings22) (earnings calls from global companies).

| Model            | Earnings21 | Earnings22 | Rev16 |
|------------------|------------|------------|-------|
| Rev (Verbatim)   |       9.68 |      13.68 | 10.30 |
| Whisper Large V3 |      14.26 |      19.05 | 10.86 |
| Canary 1B        |      14.40 |      19.01 | 13.82 |

See the wer_evaluation folder for benchmarking scripts and usage instructions.

## Acknowledgments <a name="acknowledgements"></a>
Special thanks to the Wenet team for their work and for making it available under an open-source license.


