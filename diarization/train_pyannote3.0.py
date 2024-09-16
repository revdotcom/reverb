#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2024
# Author: Jan Profant <jan.profant@rev.com>
# All Rights Reserved
import argparse
import os

import torch

from pyannote.audio import Pipeline, Model
from pyannote.database import FileFinder, registry
from pyannote.audio.tasks import Segmentation
from types import MethodType
from torch.optim import Adam
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning import Trainer

torch.set_float32_matmul_precision('high')


def configure_optimizers(self):
    return Adam(self.parameters(), lr=1e-4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Pyannote LSTM model')
    parser.add_argument('--database', type=str, required=True)
    parser.add_argument('--hf-access-token', type=str, required=False, default=None)

    args = parser.parse_args()
    registry.load_database(args.database)
    dataset = registry.get_protocol('audiodb.SpeakerDiarization.train_protocol',
                                     preprocessors={'audio': FileFinder()})

    hf_access_token = args.hf_access_token if args.hf_access_token else os.environ['HUGGINGFACE_ACCESS_TOKEN']
    model = Model.from_pretrained(
        "pyannote/segmentation-3.0",
        use_auth_token=args.hf_access_token)  # start from pre-trained model
    print(model)

    task = Segmentation(
        dataset,
        duration=model.specifications.duration,
        max_num_speakers=len(model.specifications.classes),
        batch_size=64,
        num_workers=16,
        loss="bce",
        vad_loss="bce")

    model.configure_optimizers = MethodType(configure_optimizers, model)
    model.task = task
    model.setup(stage='fit')

    monitor, direction = task.val_monitor
    checkpoint = ModelCheckpoint(
        monitor=monitor,
        mode=direction,
        save_top_k=1,
        every_n_epochs=1,
        save_last=False,
        save_weights_only=False,
        filename="{epoch}",
        verbose=False,
    )
    early_stopping = EarlyStopping(
        monitor=monitor,
        mode=direction,
        min_delta=0.0,
        patience=10,
        strict=True,
        verbose=False,
    )

    callbacks = [RichProgressBar(), checkpoint, early_stopping]

    # we train for at most 20 epochs (might be shorter in case of early stopping)

    trainer = Trainer(accelerator="gpu",
                      callbacks=callbacks,
                      max_epochs=20,
                      gradient_clip_val=0.5)
    trainer.fit(model)
