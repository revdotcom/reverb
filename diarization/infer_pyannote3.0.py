#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2024
# Author: Jan Profant <jan.profant@rev.com>
# All Rights Reserved

import argparse
import os
from pathlib import Path

import torch

from pyannote.audio import Pipeline


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on audio files')
    parser.add_argument('audios', nargs='+')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--hf-access-token', type=str, required=False, default=None)
    # we offer 2 models, reverb-diarization-v1 that is faster and a little bit less accurate,
    # and reverb-diarization-v2, the most accurate model but considerably slower
    parser.add_argument('--pipeline-model', type=str, required=False,
                        choices=['Revai/reverb-diarization-v1',
                                 'Revai/reverb-diarization-v2'],
                        default='Revai/reverb-diarization-v1')

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    hf_access_token = args.hf_access_token if args.hf_access_token else os.environ['HUGGINGFACE_ACCESS_TOKEN']
    finetuned_pipeline = Pipeline.from_pretrained(args.pipeline_model, use_auth_token=hf_access_token)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    finetuned_pipeline.to(torch.device(device))

    for audio in args.audios:
        print('Processing', audio)
        annotation = finetuned_pipeline(audio)
        with open(args.out_dir / f'{os.path.splitext(os.path.basename(audio))[0]}.rttm', 'w') as f:
            annotation.write_rttm(f)
