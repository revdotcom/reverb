![Rev Logo](resources/logo_purple.png#gh-light-mode-only)
![Rev Logo](resources/logo_white.png#gh-dark-mode-only)
# Reverb
Open source inference and evaluation code for Rev's state-of-the-art speech recognition and diarization models. The speech recognition (ASR) code uses the [WeNet](https://github.com/wenet-e2e/wenet) framework and the speech diarization code uses the [Pyannote](https://github.com/pyannote/pyannote-audio) framework. More detailed model descriptions can be found in our [blog](https://www.rev.com/blog/speech-to-text-technology/) and the models can be downloaded from [huggingface](https://huggingface.co/Revai). 

## Table of Contents
- [ASR](#asr)
- [Diarization](#diarization)
- [Installation](#installation)
  - [Docker Image](#docker-image)
- [Hosting the Model](#hosting-the-model)
- [License](#license)


### ASR
Speech-to-text code based on the WeNet framework. See [the ASR folder](https://github.com/revdotcom/reverb/tree/main/asr) for more details and usage instructions. 

Long-form speech recognition WER results:
| Model            | Earnings21 | Earnings22 | Rev16 |
|------------------|------------|------------|-------|
| Reverb ASR   |       9.68 |      13.68 | 10.30 |
| Whisper Large-v3 |      14.26 |      19.05 | 10.86 |
| Canary-1B        |      14.40 |      19.01 | 13.82 |

### Diarization
Speaker diarization code based on the Pyannote framework. See [the diarization folder](https://github.com/revdotcom/reverb/tree/main/diarization) for more details and usage instructions.

Long-form WDER results, in combination with Rev's ASR:
| Model            | Earnings21 |  Rev16 |
|------------------|------------|-------|
| Pyannote3.0  |    0.051    |   0.090   |
| Reverb Diarization V1 |      0.047 |   0.077 |
| Reverb Diarization V2 |      0.046 |   0.078 |

## Installation
We recommend using a virtual environment with a tool such as [anaconda](https://anaconda.org/). You might need to set your 
`HUGGINGFACE_ACCESS_TOKEN` as well since the model itself (ASR and diarization) is downloaded from the HF hub.

```bash
conda create -n reverb-env python=3.10
conda activate reverb-env
```

```bash
pip install -r asr/requirements.txt
pip install -r diarization/requirements.txt
export PYTHONPATH="$(pwd)"/asr:$PYTHONPATH  # adding this to make wenet/ work
```

Make sure that git lfs is correctly installed on your system.
```bash
git lfs install
git clone https://huggingface.co/Revai/reverb
```

### Docker Image
Alternatively, you can use Docker to run ASR and/or diarization without needing to install dependencies
directly on your system. First, make sure Docker is installed on your system. If you wish to run
on NVIDIA GPU, more steps might be required.
Then, run the following command to build the Docker image:
```bash
docker build -t reverb . --build-arg HUGGINGFACE_ACCESS_TOKEN=${YOUR_HUGGINGFACE_ACCESS_TOKEN} 
```

And to run docker
```bash
sudo docker run --entrypoint "/bin/bash" --gpus all --rm -it reverb
```

# Hosting the Model
If your usecase requires a to deploy these models at a larger scale and maintaining strict
security requirements, consider using our other release: https://github.com/revdotcom/revai.
This setup will give you full control over the deployment of our models on your own infrastructure
without the need for internet connectivity or cloud dependencies.

# License
See LICENSE for details.
