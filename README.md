## Reginald
Open source inference and evaluation code for Rev's state-of-the-art speech recognition and diarization models. The speech recognition (ASR) code uses the [WeNet](https://github.com/wenet-e2e/wenet) framework and the speech diarization code uses the [Pyannote](https://github.com/pyannote/pyannote-audio) framework. More information can be found on our [blog](rev.com/blog) and the models can be downloaded from [huggingface](https://huggingface.co/Revai). 

## Table of Contents
- [ASR](#asr)
- [Diarization](#diarization)
- [Installation](#installation)
  - [Docker Image](#docker-image) 


### ASR
Speech-to-text code based on the WeNet framework. See [the ASR folder](https://github.com/revdotcom/reginald/tree/main/asr) for more details and usage instructions. 

### Diarization
Speaker diarization code based on the Pyannote framework. See [the diarization folder](https://github.com/revdotcom/reginald/tree/main/diarization) for more details and usage instructions.

## Installation
We recommend using a virtual environment with a tool such as [anaconda](https://anaconda.org/). You might need to set your 
`HUGGINGFACE_ACCESS_TOKEN` as well since the model itself (ASR and diarization) is downloaded from the HF hub.

```bash
conda create -n reginald-env python=3.10
conda activate reginald-env
```

```bash
pip install -r asr/requirements.txt
pip install -r diarization/requirements.txt
export PYTHONPATH="$(pwd)"/asr:$PYTHONPATH  # adding this to make wenet/ work
```

Make sure that git lfs is correctly installed on your system.
```bash
git lfs install
git clone https://huggingface.co/Revai/reginald
```

### Docker Image
Alternatively, you can use Docker to run the `reginald` model without needing to install dependencies
directly on your system. First, make sure Docker is installed on your system. If you wish to run
on NVIDIA GPU, more steps might be required.
Then, run the following command to build the Docker image:
```bash
docker build -t reginald . --build-arg HUGGINGFACE_ACCESS_TOKEN=${YOUR_HUGGINGFACE_ACCESS_TOKEN} 
```

And to run docker
```bash
sudo docker run --entrypoint "/bin/bash" --gpus all --rm -it reginald
```

### License
See LICENSE for details.
