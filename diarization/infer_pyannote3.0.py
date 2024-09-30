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

from pyannote.audio import Model
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization

V1_MIN_DURATION_OFF = 0.2263
V1_CLUSTERING_THRESHOLD = 0.6939
V1_CLUSTERING_METHOD = 'centroid'
V1_MIN_CLUSTER_SIZE = 15

V2_MIN_DURATION_OFF = 0.5281
V2_CLUSTERING_THRESHOLD = 0.6814
V2_CLUSTERING_METHOD = 'centroid'
V2_MIN_CLUSTER_SIZE = 17

BATCH_SIZE = 64


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on audio files')
    parser.add_argument('audios', nargs='+')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--hf-access-token', type=str, required=False, default=None)
    # we offer 2 models, reverb-diarization-v1 that is faster and a little bit less accurate, and reverb-diarization-v2, the most accurate model
    # but considerably slower
    parser.add_argument('--lstm-model', type=str, required=False,
                        choices=['Revai/reverb-diarization-v1', 'Revai/reverb-diarization-v2'], default='Revai/reverb-diarization-v1')

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    hf_access_token = args.hf_access_token if args.hf_access_token else os.environ['HUGGINGFACE_ACCESS_TOKEN']
    segmentation_model = Model.from_pretrained(args.lstm_model,
                                               use_auth_token=hf_access_token)
    embedding_model = Model.from_pretrained('Revai/wespeaker-voxceleb-resnet34-LM',
                                            use_auth_token=hf_access_token)

    finetuned_pipeline = SpeakerDiarization(
        segmentation=segmentation_model,
        embedding=embedding_model,
        embedding_exclude_overlap=True,
        clustering='AgglomerativeClustering',
        use_auth_token=hf_access_token)

    if args.lstm_model == 'Revai/fico':
        min_duration_off = V1_MIN_DURATION_OFF
        clustering_threshold = V1_CLUSTERING_THRESHOLD
        clustering_method = V1_CLUSTERING_METHOD
        min_cluster_size = V1_MIN_CLUSTER_SIZE
    else:
        min_duration_off = V2_MIN_DURATION_OFF
        clustering_threshold = V2_CLUSTERING_THRESHOLD
        clustering_method = V2_CLUSTERING_METHOD
        min_cluster_size = V2_MIN_CLUSTER_SIZE

    finetuned_pipeline._segmentation.batch_size = BATCH_SIZE

    finetuned_pipeline = finetuned_pipeline.instantiate({
        'segmentation': {
            'min_duration_off': min_duration_off
        },
        'clustering': {
            'threshold': clustering_threshold,
            'method': clustering_method,
            'min_cluster_size': min_cluster_size
        }
    })

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    finetuned_pipeline.to(torch.device(device))

    for audio in args.audios:
        print('Processing', audio)
        annotation = finetuned_pipeline(audio)
        with open(args.out_dir / f'{os.path.splitext(os.path.basename(audio))[0]}.rttm', 'w') as f:
            annotation.write_rttm(f)

