FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        git-lfs \
        locales && \
    rm -rf /var/lib/apt/lists/*

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

WORKDIR /workspace
COPY . /workspace/

ARG HUGGINGFACE_ACCESS_TOKEN
ENV HUGGINGFACE_ACCESS_TOKEN=${HUGGINGFACE_ACCESS_TOKEN}

# manually download ASR model
# diarization will be download automatically when running the script due to HF integration
RUN git lfs install
RUN git clone https://${HUGGINGFACE_ACCESS_TOKEN}:${HUGGINGFACE_ACCESS_TOKEN}@huggingface.co/Revai/reverb-asr


RUN pip3 install -r /workspace/asr/requirements.txt
RUN pip3 install -r /workspace/diarization/requirements.txt

ENV PYTHONPATH=/workspace/asr/:$PYTHONPATH

RUN python3 /workspace/asr/wenet/bin/recognize_wav.py --help
RUN python3 /workspace/diarization/infer_pyannote3.0.py --help
