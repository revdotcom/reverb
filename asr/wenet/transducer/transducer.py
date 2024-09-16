from typing import Dict, List, Optional, Tuple, Union

import torch
import torchaudio
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from wenet.transducer.predictor import PredictorBase
from wenet.transducer.search.greedy_search import basic_greedy_search
from wenet.transducer.search.prefix_beam_search import PrefixBeamSearch
from wenet.transformer.asr_model import ASRModel
from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import BiTransformerDecoder, TransformerDecoder
from wenet.transformer.label_smoothing_loss import LabelSmoothingLoss
from wenet.utils.common import (IGNORE_ID, add_blank, add_sos_eos,
                                reverse_pad_list)


class Transducer(ASRModel):
    """Transducer-ctc-attention hybrid Encoder-Predictor-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        blank: int,
        encoder: nn.Module,
        predictor: PredictorBase,
        joint: nn.Module,
        attention_decoder: Optional[Union[TransformerDecoder,
                                          BiTransformerDecoder]] = None,
        ctc: Optional[CTC] = None,
        ctc_weight: float = 0,
        ignore_id: int = IGNORE_ID,
        reverse_weight: float = 0.0,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        transducer_weight: float = 1.0,
        attention_weight: float = 0.0,
        transducer_type: str = "optimized_transducer",
        enable_k2: bool = False,
        delay_penalty: float = 0.0,
        warmup_steps: float = 25000,
        lm_only_scale: float = 0.25,
        am_only_scale: float = 0.0,
        special_tokens: dict = None,
    ) -> None:
        assert attention_weight + ctc_weight + transducer_weight == 1.0
        super().__init__(vocab_size,
                         encoder,
                         attention_decoder,
                         ctc,
                         ctc_weight,
                         ignore_id,
                         reverse_weight,
                         lsm_weight,
                         length_normalized_loss,
                         special_tokens=special_tokens)

        self.blank = blank
        self.transducer_weight = transducer_weight
        self.attention_decoder_weight = 1 - self.transducer_weight - self.ctc_weight

        self.predictor = predictor
        self.joint = joint
        self.bs = None
        self.transducer_type = transducer_type

        # k2 rnnt loss
        self.enable_k2 = enable_k2
        self.delay_penalty = delay_penalty
        if delay_penalty != 0.0:
            assert self.enable_k2 is True
        self.lm_only_scale = lm_only_scale
        self.am_only_scale = am_only_scale
        self.warmup_steps = warmup_steps
        self.simple_am_proj: Optional[nn.Linear] = None
        self.simple_lm_proj: Optional[nn.Linear] = None
        if self.enable_k2:
            self.simple_am_proj = torch.nn.Linear(self.encoder.output_size(),
                                                  vocab_size)
            self.simple_lm_proj = torch.nn.Linear(self.predictor.output_size(),
                                                  vocab_size)

        # Note(Mddct): decoder also means predictor in transducer,
        # but here decoder is attention decoder
        del self.criterion_att
        if attention_decoder is not None:
            self.criterion_att = LabelSmoothingLoss(
                size=vocab_size,
                padding_idx=ignore_id,
                smoothing=lsm_weight,
                normalize_length=length_normalized_loss,
            )

        if transducer_type == "optimized_transducer":
            # print("WARNING: import optimized_transducer is disabled due to `GLIBC_2.27' not found")
            import optimized_transducer
            self.criterion_transducer = optimized_transducer.transducer_loss

    @torch.jit.ignore(drop=True)
    def optimized_transducer_loss(
            self,
            logits,
            targets,
            logit_lengths,
            target_lengths,
            blank
    ):
        return self.criterion_transducer(
            logits,
            targets,
            logit_lengths,
            target_lengths,
            blank=blank,
            reduction="mean",
            from_log_softmax=False,
            # one_sym_per_frame=True, # TODO: Add to config
            )

    def forward(
        self,
        batch: dict,
        device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Frontend + Encoder + predictor + joint + loss
        """
        speech = batch['feats'].to(device)
        speech_lengths = batch['feats_lengths'].to(device)
        text = batch['target'].to(device)
        text_lengths = batch['target_lengths'].to(device)
        steps = batch.get('steps', 0)
        cat_embs = batch.get('cat_embs', None).to(device)
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (speech.shape[0] == speech_lengths.shape[0] == text.shape[0] ==
                text_lengths.shape[0]), (speech.shape, speech_lengths.shape,
                                         text.shape, text_lengths.shape)

        # Encoder
        encoder_out, encoder_mask = self.encoder(speech, speech_lengths, cat_embs = cat_embs)
        
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        # predictor
        ys_in_pad = add_blank(text, self.blank, self.ignore_id)
        predictor_out = self.predictor(ys_in_pad)
        # joint
        joint_out = self.joint(encoder_out, predictor_out)
        # NOTE(Mddct): some loss implementation require pad valid is zero
        # torch.int32 rnnt_loss required
        rnnt_text = text.to(torch.int64)
        rnnt_text = torch.where(rnnt_text == self.ignore_id, 0,
                                rnnt_text).to(torch.int32)
        rnnt_text_lengths = text_lengths.to(torch.int32)
        encoder_out_lens = encoder_out_lens.to(torch.int32)
        loss = torchaudio.functional.rnnt_loss(joint_out,
                                               rnnt_text,
                                               encoder_out_lens,
                                               rnnt_text_lengths,
                                               blank=self.blank,
                                               reduction="mean")
        loss_rnnt = loss

        # optional attention decoder
        loss_att: Optional[torch.Tensor] = None
        if self.attention_decoder_weight != 0.0 and self.attention_decoder is not None:
            loss_att, _ = self._calc_att_loss(encoder_out, encoder_mask, text,
                                              text_lengths)

        # optional ctc
        loss_ctc: Optional[torch.Tensor] = None
        if self.ctc_weight != 0.0 and self.ctc is not None:
            loss_ctc, _ = self.ctc(encoder_out, encoder_out_lens, text,
                                   text_lengths)
        else:
            loss_ctc = None

        if loss_ctc is not None:
            loss = loss + self.ctc_weight * loss_ctc.sum()
        if loss_att is not None:
            loss = loss + self.attention_decoder_weight * loss_att.sum()
        # NOTE: 'loss' must be in dict
        return {
            'loss': loss,
            'loss_att': loss_att,
            'loss_ctc': loss_ctc,
            'loss_rnnt': loss_rnnt,
        }

    def init_bs(self):
        if self.bs is None:
            self.bs = PrefixBeamSearch(self.encoder, self.predictor,
                                       self.joint, self.ctc, self.blank)

    def _cal_transducer_score(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        hyps_lens: torch.Tensor,
        hyps_pad: torch.Tensor,
    ):
        # ignore id -> blank, add blank at head
        hyps_pad_blank = add_blank(hyps_pad, self.blank, self.ignore_id)
        xs_in_lens = encoder_mask.squeeze(1).sum(1).int()

        # 1. Forward predictor
        predictor_out = self.predictor(hyps_pad_blank)
        # 2. Forward joint
        joint_out = self.joint(encoder_out, predictor_out)
        rnnt_text = hyps_pad.to(torch.int64)
        rnnt_text = torch.where(rnnt_text == self.ignore_id, 0,
                                rnnt_text).to(torch.int32)
        # 3. Compute transducer loss
        loss_td = torchaudio.functional.rnnt_loss(joint_out,
                                                  rnnt_text,
                                                  xs_in_lens,
                                                  hyps_lens.int(),
                                                  blank=self.blank,
                                                  reduction='none')
        return loss_td * -1

    def _cal_attn_score(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos,
                                            self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # reverse the seq, used for right to left decoder
        r_ys_pad = reverse_pad_list(ys_pad, ys_pad_lens, float(self.ignore_id))
        r_ys_in_pad, r_ys_out_pad = add_sos_eos(r_ys_pad, self.sos, self.eos,
                                                self.ignore_id)
        # 1. Forward decoder
        decoder_out, r_decoder_out, _ = self.attention_decoder(
            encoder_out, encoder_mask, ys_in_pad, ys_in_lens, r_ys_in_pad,
            self.reverse_weight)
        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        r_loss_att = torch.tensor(0.0)
        if self.reverse_weight > 0.0:
            r_loss_att = self.criterion_att(r_decoder_out, r_ys_out_pad)
        loss_att = loss_att * (
            1 - self.reverse_weight) + r_loss_att * self.reverse_weight
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )
        return loss_att, acc_att

    def greedy_search(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
        n_steps: int = 64,
    ) -> List[List[int]]:
        """ greedy search

        Args:
            speech (torch.Tensor): (batch=1, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
        Returns:
            List[List[int]]: best path result
        """
        # TODO(Mddct): batch decode
        assert speech.size(0) == 1
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        # TODO(Mddct): forward chunk by chunk
        _ = simulate_streaming
        # Let's assume B = batch_size
        encoder_out, encoder_mask = self.encoder(
            speech,
            speech_lengths,
            decoding_chunk_size,
            num_decoding_left_chunks,
        )
        encoder_out_lens = encoder_mask.squeeze(1).sum()
        hyps = basic_greedy_search(self,
                                   encoder_out,
                                   encoder_out_lens,
                                   n_steps=n_steps)

        return hyps

    @torch.jit.export
    def forward_encoder_chunk(
        self,
        xs: torch.Tensor,
        offset: int,
        required_cache_size: int,
        att_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        cnn_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        return self.encoder.forward_chunk(xs, offset, required_cache_size,
                                          att_cache, cnn_cache)

    @torch.jit.export
    def forward_predictor_step(
            self, xs: torch.Tensor, cache: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        assert len(cache) == 2
        # fake padding
        padding = torch.zeros(1, 1)
        return self.predictor.forward_step(xs, padding, cache)

    @torch.jit.export
    def forward_joint_step(self, enc_out: torch.Tensor,
                           pred_out: torch.Tensor) -> torch.Tensor:
        return self.joint(enc_out, pred_out)

    @torch.jit.export
    def forward_predictor_init_state(
            self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.predictor.init_state(1)
