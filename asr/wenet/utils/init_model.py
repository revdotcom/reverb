# Copyright (c) 2022 Binbin Zhang (binbzha@qq.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch

from wenet.finetune.lora.utils import mark_only_lora_as_trainable, lora_state_dict
from wenet.k2.model import K2Model
from wenet.paraformer.cif import Cif
from wenet.paraformer.layers import SanmDecoder, SanmEncoder
from wenet.paraformer.paraformer import Paraformer, Predictor
from wenet.transducer.joint import TransducerJoint
from wenet.transducer.predictor import (ConvPredictor, EmbeddingPredictor,
                                        RNNPredictor)
from wenet.transducer.transducer import Transducer
from wenet.transformer.asr_model import ASRModel
from wenet.transformer.cmvn import GlobalCMVN
from wenet.transformer.ctc import CTC
from wenet.transformer.encoder import TransformerEncoder, ConformerEncoder, LanguageSpecificConformerEncoder
from wenet.transformer.decoder import BiTransformerDecoder, TransformerDecoder, LanguageSpecificBiTransformerDecoder, LanguageSpecificTransformerDecoder
from wenet.branchformer.encoder import BranchformerEncoder
from wenet.e_branchformer.encoder import EBranchformerEncoder
from wenet.squeezeformer.encoder import SqueezeformerEncoder
from wenet.efficient_conformer.encoder import EfficientConformerEncoder
from wenet.ctl_model.encoder import DualTransformerEncoder, DualConformerEncoder
from wenet.ctl_model.asr_model_ctl import CTLModel
from wenet.whisper.whisper import Whisper
from wenet.transformer.context_adaptor import ContextAdaptor
from wenet.utils.cmvn import load_cmvn
from wenet.utils.checkpoint import load_checkpoint, download_checkpoint_from_wandb, load_trained_modules
from wenet.finetune.lora.encoder import (LoRATransformerEncoder,
                                         LoRAConformerEncoder)
from wenet.transformer.ts_asr_model import init_ts_asr_model

import yaml
import numpy as np
import logging

WENET_ENCODER_CLASSES = {
    "transformer": TransformerEncoder,
    "conformer": ConformerEncoder,
    "lslconformer": LanguageSpecificConformerEncoder,
    "squeezeformer": SqueezeformerEncoder,
    "efficientConformer": EfficientConformerEncoder,
    "branchformer": BranchformerEncoder,
    "e_branchformer": EBranchformerEncoder,
    "dual_transformer": DualTransformerEncoder,
    "dual_conformer": DualConformerEncoder,
    'sanm_encoder': SanmEncoder,
    "lora_transformer": LoRATransformerEncoder,
    "lora_conformer": LoRAConformerEncoder,
}

WENET_DECODER_CLASSES = {
    "transformer": TransformerDecoder,
    "bitransformer": BiTransformerDecoder,
    "lsltransformer": LanguageSpecificTransformerDecoder,
    "lslbitransformer": LanguageSpecificBiTransformerDecoder,
    "sanm_decoder": SanmDecoder,
}

WENET_CTC_CLASSES = {
    "ctc": CTC,
}

WENET_PREDICTOR_CLASSES = {
    "rnn": RNNPredictor,
    "embedding": EmbeddingPredictor,
    "conv": ConvPredictor,
    "cif_predictor": Cif,
    "paraformer_predictor": Predictor,
}

WENET_JOINT_CLASSES = {
    "transducer_joint": TransducerJoint,
}

WENET_MODEL_CLASSES = {
    "asr_model": ASRModel,
    "ctl_model": CTLModel,
    "whisper": Whisper,
    "k2_model": K2Model,
    "transducer": Transducer,
    'paraformer': Paraformer,
}


def init_model(args, configs):

    # TODO(xcsong): Forcefully read the 'cmvn' attribute.
    if configs.get('cmvn', None) == 'global_cmvn':
        mean, istd = load_cmvn(configs['cmvn_conf']['cmvn_file'],
                               configs['cmvn_conf']['is_json_cmvn'])
        if 'add_cat_emb' in configs['dataset_conf'] and configs['dataset_conf']['add_cat_emb']:
            # if we add a category embedding to every frame, then the CMVN matrices need to be bigger
            # to match the new feature size
            # we use mean=zero and std=1 so that CMVN doesn't impact these features
            # JPR : do we need to edit this when finetuning the model ?
            emb_len = configs['dataset_conf']['cat_emb_conf']['emb_len']
            extra_zeros = np.zeros((emb_len,))
            mean = np.concatenate((mean, extra_zeros))
            extra_ones = np.ones((emb_len,))
            istd = np.concatenate((istd, extra_ones))
        global_cmvn = GlobalCMVN(
            torch.from_numpy(mean).float(),
            torch.from_numpy(istd).float())
    else:
        global_cmvn = None

    input_dim = configs['input_dim']
    vocab_size = configs['output_dim']

    # Teacher student section
    ts_config = configs.get('ts_conf', None)
    is_teacher_student = ts_config is not None
    if is_teacher_student:
        teacher_train_yaml = ts_config['teacher_yaml']
        teacher_checkpoint = ts_config['teacher_checkpoint']
        # Read config
        with open(teacher_train_yaml, 'r') as fin:
            teacher_configs = yaml.load(fin, Loader=yaml.FullLoader)
        teacher_model, teacher_configs = init_model(args, teacher_configs)
        logging.info(f"Since teacher_student config is found, loading teacher model from {teacher_checkpoint}")
        load_checkpoint(teacher_model, teacher_checkpoint)

    language_specific_layers = configs['dataset_conf'].get('pass_cat_emb', False)

    model_type = configs.get('model', 'asr_model')
    encoder_type = configs.get('encoder', 'conformer')
    if language_specific_layers:
        configs['encoder_conf']['num_langs'] = configs['dataset_conf']['cat_emb_conf']['emb_len']
    else:
        configs['encoder_conf']['num_langs'] = 0

    # JPR : let's disable this for now, relying on the augmentation of the regular Conformer class modifications
    if False and (language_specific_layers and model_type != 'ctl_model'):
        configs['encoder_conf']['num_langs'] = configs['dataset_conf']['cat_emb_conf']['emb_len']
        encoder_type = 'lslconformer'

    decoder_type = configs.get('decoder', 'bitransformer')

    # JPR : for now, we can leave this here, we'll touch the decoder part later to merge the lsl usage
    if language_specific_layers:
        configs['decoder_conf']['num_langs'] = configs['dataset_conf']['cat_emb_conf']['emb_len']
        if configs['decoder_conf'].get('r_num_blocks',0) > 0 and configs['model_conf']['reverse_weight'] > 0.0:
            assert 0.0 < configs['model_conf']['reverse_weight'] < 1.0
            decoder_type = 'lslbitransformer'
        else:
            decoder_type = 'lsltransformer'

    ctc_type = configs.get('ctc', 'ctc')

    if hasattr(args, 'use_lora') and args.use_lora:
        encoder_type = "lora_" + encoder_type

    encoder = WENET_ENCODER_CLASSES[encoder_type](
        input_dim,
        global_cmvn=global_cmvn,
        **configs['encoder_conf'],
        **configs['encoder_conf']['efficient_conf']
        if 'efficient_conf' in configs['encoder_conf'] else {})

    decoder = WENET_DECODER_CLASSES[decoder_type](vocab_size,
                                                  encoder.output_size(),
                                                  **configs['decoder_conf'])

    ctc = WENET_CTC_CLASSES[ctc_type](
        vocab_size,
        encoder.output_size(),
        blank_id=configs['ctc_conf']['ctc_blank_id']
        if 'ctc_conf' in configs else 0)

    deep_biasing = configs['dataset_conf'].get('deep_bias_conf', {}).get('deep_biasing', False) # TODO: this shouldn't be in dataset_conf for this

    if deep_biasing:
        context_adaptor = ContextAdaptor(vocab_size, encoder.output_size())
    else:
        context_adaptor = None

    if model_type == "transducer":
        predictor_type = configs.get('predictor', 'rnn')
        joint_type = configs.get('joint', 'transducer_joint')
        predictor = WENET_PREDICTOR_CLASSES[predictor_type](
            vocab_size, **configs['predictor_conf'])
        joint = WENET_JOINT_CLASSES[joint_type](vocab_size,
                                                **configs['joint_conf'])
        model = WENET_MODEL_CLASSES[model_type](
            vocab_size=vocab_size,
            blank=0,
            predictor=predictor,
            encoder=encoder,
            attention_decoder=decoder,
            joint=joint,
            ctc=ctc,
            special_tokens=configs.get('tokenizer_conf',
                                       {}).get('special_tokens', None),
            **configs['model_conf'])
    elif model_type == 'paraformer':
        predictor_type = configs.get('predictor', 'cif')
        predictor = WENET_PREDICTOR_CLASSES[predictor_type](
            **configs['predictor_conf'])
        model = WENET_MODEL_CLASSES[model_type](
            vocab_size=vocab_size,
            encoder=encoder,
            decoder=decoder,
            predictor=predictor,
            ctc=ctc,
            **configs['model_conf'],
            special_tokens=configs.get('tokenizer_conf',
                                       {}).get('special_tokens', None),
        )
    else:
        model = WENET_MODEL_CLASSES[model_type](
            vocab_size=vocab_size,
            encoder=encoder,
            decoder=decoder,
            ctc=ctc,
            context_adaptor=context_adaptor,
            special_tokens=configs.get('tokenizer_conf',
                                       {}).get('special_tokens', None),
            **configs['model_conf'])

    # If set, finds the last WandB checkpoint with optimizer and downloads it
    if hasattr(args, 'load_from_wandb') and args.load_from_wandb:
        args.checkpoint = download_checkpoint_from_wandb(model, os.path.join(args.model_dir, "wandb-snapshot_and_optimizer.pt"))

    # If specify checkpoint, load some info from checkpoint
    if hasattr(args, 'checkpoint') and args.checkpoint is not None:
        infos = load_checkpoint(model, args.checkpoint, def_strict=False)
    elif hasattr(args, 'checkpoint') and args.enc_init is not None:
        infos = load_trained_modules(model, args)
    else:
        infos = {}
    configs["init_infos"] = infos

    if hasattr(args, 'only_optimize_lora') and args.only_optimize_lora:
        mark_only_lora_as_trainable(model, bias='lora_only')

    #print(configs)

    if language_specific_layers:
        model.lsl_enc = True
        model.lsl_dec = True
    else:
        model.lsl_enc = False
        model.lsl_dec = False

    model.add_cat_embs = configs['dataset_conf'].get('add_cat_emb', False)
    use_cat_embs = language_specific_layers or model.add_cat_embs
    model.cat_labels = []
    if use_cat_embs:
        model.cat_labels = [""]*configs['dataset_conf']['cat_emb_conf']['emb_len']
        for k in configs['dataset_conf']['cat_emb_conf']['one_hot_ids']:
            model.cat_labels[configs['dataset_conf']['cat_emb_conf']['one_hot_ids'][k]] = k

    logging.info(f"model.lsl_enc = {model.lsl_enc}, model.lsl_dec = {model.lsl_dec}, model.add_cat_embs = {model.add_cat_embs}, model.cat_labels = {model.cat_labels}")

    # Tie emb.weight to decoder.output_layer.weight
    if model.decoder.tie_word_embedding:
        model.decoder.tie_or_clone_weights(jit_mode=args.jit)

    if is_teacher_student:
        teacher_student_model = init_ts_asr_model(teacher_model, model, configs)
        return teacher_student_model, configs
    else:
        return model, configs
