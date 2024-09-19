## Reginald
Open source inference and evaluation code for Rev's Speech models.

## Table of Contents
- [ASR](#asr)
- [Diarization](#diarization)
- [Installation](#installation)
  - [Docker Image](#docker-image) 


### ASR
Speech-to-text models/code based on the WeNet framework.

### Diarization
Speaker diarization models/code based on the Pyannote framework.

## Installation
We recommend using virtual environment with a tool such as [anaconda](https://anaconda.org/). You might need to set your 
`HUGGINGFACE_ACCESS_TOKEN` as well since the model itself (ASR and diarization) is downloaded 
from HF hub.

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
Alternatively, you can use Docker to run `reginald` model without needing to install dependencies
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
