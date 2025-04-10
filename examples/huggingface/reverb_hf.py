# Following https://huggingface.co/docs/transformers/en/custom_models

from typing import List, Optional, Tuple, Union
import torch
from transformers import PreTrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput
from wenet.transformer.asr_model import ASRModel
from wenet.transformer.cmvn import GlobalCMVN
from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import LanguageSpecificBiTransformerDecoder
from wenet.transformer.encoder import ConformerEncoder
from reverb_config import ReverbConfig

class ReverbModel(PreTrainedModel):
    config_class = ReverbConfig
    main_input_name = "input_features"

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        global_cmvn = GlobalCMVN(
            torch.Tensor(config.cmvn_mean),
            torch.Tensor(config.cmvn_istd),
        )
        encoder = ConformerEncoder(
            config.input_dim,
            global_cmvn=global_cmvn,
            activation_type=config.encoder_activation_type,
            attention_dropout_rate=config.encoder_attention_dropout_rate,
            attention_heads=config.encoder_attention_heads,
            causal=config.encoder_causal,
            cnn_module_kernel=config.encoder_cnn_module_kernel,
            cnn_module_norm=config.encoder_cnn_module_norm,
            dropout_rate=config.encoder_dropout_rate,
            input_layer=config.encoder_input_layer,
            linear_units=config.encoder_linear_units,
            normalize_before=config.encoder_normalize_before,
            num_blocks=config.encoder_num_blocks,
            num_langs=config.encoder_num_langs,
            output_size=config.encoder_output_size,
            pos_enc_layer_type=config.encoder_pos_enc_layer_type,
            positional_dropout_rate=config.encoder_positional_dropout_rate,
            selfattention_layer_type=config.encoder_selfattention_layer_type,
            use_cnn_module=config.encoder_use_cnn_module,
            use_dynamic_chunk=config.encoder_use_dynamic_chunk,
        )

        decoder = LanguageSpecificBiTransformerDecoder(
            config.output_dim,
            config.encoder_output_size,
            attention_heads=config.decoder_attention_heads,
            dropout_rate=config.decoder_dropout_rate,
            linear_units=config.decoder_linear_units,
            num_blocks=config.decoder_num_blocks,
            num_langs=config.decoder_num_langs,
            positional_dropout_rate=config.decoder_positional_dropout_rate,
            r_num_blocks=config.decoder_r_num_blocks,
            self_attention_dropout_rate=config.decoder_self_attention_dropout_rate,
            src_attention_dropout_rate=config.decoder_src_attention_dropout_rate,
        )

        ctc = CTC(
            config.output_dim,
            config.encoder_output_size,
            config.ctc_blank_id,
        )

        self.model = ASRModel(
            vocab_size=config.output_dim,
            encoder=encoder,
            decoder=decoder,
            ctc=ctc,
            special_tokens=config.special_tokens,
            ctc_weight=config.ctc_weight,
            lsm_weight=config.lsm_weight,
            reverse_weight=config.reverse_weight,
        )
        self.model.lsl_enc = True
        self.model.lsl_dec = True

    def forward(
        self,
        input_features=None,
        feats_lengths=None,
        labels=None,
        labels_lengths=None,
        **kwargs,
    ):
        output = self.model.hf_forward(
            input_features,
            feats_lengths=feats_lengths,
            labels=labels,
            labels_lengths=labels_lengths,
        )
        return Seq2SeqLMOutput(
            logits=output['ctc_probs'],
            loss=output['loss'],
        )
