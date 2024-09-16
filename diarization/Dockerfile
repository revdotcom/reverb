FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        locales && \
    rm -rf /var/lib/apt/lists/*

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

ARG HUGGINGFACE_ACCESS_TOKEN
ENV HUGGINGFACE_ACCESS_TOKEN=${HUGGINGFACE_ACCESS_TOKEN}

WORKDIR /workspace
COPY . /workspace/
# FIXME

RUN pip3 install -r /workspace/requirements.txt

RUN python3 /workspace/infer_pyannote3.0.py --help
RUN python3 /workspace/train_pyannote3.0.py --help

ENTRYPOINT ["python", "/workspace/infer_pyannote3.0.py"]
CMD ["test_audio.wav", "--out-dir", "outdir"]
# FIXME example test file(s)