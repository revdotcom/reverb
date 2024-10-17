# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Di Wu)
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
# Modified from ESPnet(https://github.com/espnet/espnet)
import logging
from typing import List, Optional, Tuple, Dict, Any, Set

import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
#import logging

try:
    import k2
    from icefall.utils import get_texts
    from icefall.decode import get_lattice, Nbest, one_best_decoding
except ImportError:
    logging.debug('Failed to import k2 and icefall. \
        Notice that they are necessary for hlg_onebest and hlg_rescore')

from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import TransformerDecoder
from wenet.transformer.encoder import BaseEncoder
from wenet.transformer.context_adaptor import ContextAdaptor
from wenet.transformer.label_smoothing_loss import LabelSmoothingLoss
from wenet.transformer.search import (ctc_greedy_search,
                                      ctc_prefix_beam_search,
                                      attention_beam_search,
                                      attention_rescoring,
                                      joint_decoding,
                                      DecodeResult)
from wenet.utils.mask import (make_pad_mask, subsequent_mask)
from wenet.utils.common import (IGNORE_ID, add_sos_eos, th_accuracy,
                                reverse_pad_list)
from wenet.utils.context_graph import ContextGraph

from wenet.onmt_translate.beam_search import GNMTGlobalScorer
from wenet.onmt_translate.beam_search import BeamSearch

from wenet.espnet.beam_search_timesync import BeamSearchTimeSync

class ASRModel(torch.nn.Module):
    """CTC-attention hybrid Encoder-Decoder model"""
    words: Dict[str, int]
    word_prefixes: Dict[str, int]
    tok_to_str: Dict[int, str]
    pre_beam_ratio: float

    def __init__(self,
                 vocab_size: int,
                 encoder: BaseEncoder,
                 decoder: TransformerDecoder,
                 ctc: CTC,
                 ctc_weight: float = 0.5,
                 ignore_id: int = IGNORE_ID,
                 reverse_weight: float = 0.0,
                 lsm_weight: float = 0.0,
                 length_normalized_loss: bool = False,
                 special_tokens: Optional[dict] = None,
                 apply_non_blank_embedding: bool = False,
                 non_spike_loss_weight: float = 0.0,
                 context_adaptor: Optional[ContextAdaptor] = None,
                 lexicon_path: Optional[str] = None,
                 token_path: Optional[str] = None):
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.sos = (vocab_size - 1 if special_tokens is None else
                    special_tokens.get("<sos>", vocab_size - 1))
        self.eos = (vocab_size - 1 if special_tokens is None else
                    special_tokens.get("<eos>", vocab_size - 1))
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.reverse_weight = reverse_weight
        self.apply_non_blank_embedding = apply_non_blank_embedding
        self.non_spike_loss_weight = non_spike_loss_weight

        # self.lsl_enc = False #JPR : for now
        # self.lsl_dec = False #JPR : for now
        # self.add_cat_embs = False #JPR : for now

        self.encoder = encoder
        self.decoder = decoder
        self.ctc = ctc
        self.criterion_att = LabelSmoothingLoss(
            size=vocab_size,
            padding_idx=ignore_id,
            smoothing=lsm_weight,
            normalize_length=length_normalized_loss,
        )
        self.context_adaptor = context_adaptor

        # JDF - for joint decoding
        # ctc_weight and bonus chosen through experimentation
        self.joint_ctc_weight: float = 0.5
        self.length_bonus: float = 0.5
        self.pre_beam_ratio: float = 2 # default was 1.5 in ESPnet

        self.words: Dict[str, int] = dict()
        self.word_prefixes: Dict[str, int] = dict()
        if lexicon_path is not None:
            f = open(lexicon_path)
            for line in f:
                p = line.strip().split()
                self.words[p[0]] = 1 #self.words.append(p[0])
                ws = ''
                for sw in p[1:]:
                    ws = ws + sw
                    self.word_prefixes[ws] = 1 #self.word_prefixes.append(ws)

        self.tok_to_str: Dict[int, str] = dict()
        if token_path is not None:
            for line in open(token_path):
                s = line.strip().split()
                self.tok_to_str[int(s[1])] = s[0]

    @torch.jit.ignore(drop=True)
    def forward(
        self,
        batch: dict,
        device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Frontend + Encoder + Decoder + Calc loss"""
        speech = batch['feats'].to(device)
        speech_lengths = batch['feats_lengths'].to(device)
        text = batch['target'].to(device)
        text_lengths = batch['target_lengths'].to(device)
        text_lengths = batch['target_lengths'].to(device)

        if 'cat_embs' in batch:
            cat_embs = batch['cat_embs'].to(device)
        else:
            cat_embs = None

        # TODO: verify if both can be handled together
        if 'cv_list' in batch:
            cv_list = batch['cv_list'].to(device)
            cv_list_lengths = batch['cv_list_lengths'].to(device)
        else:
            cv_list = None
            cv_list_lengths = None

        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (speech.shape[0] == speech_lengths.shape[0] == text.shape[0] ==
                text_lengths.shape[0]), (speech.shape, speech_lengths.shape,
                                         text.shape, text_lengths.shape)
        # 1. Encoder
        if self.context_adaptor is not None and (cv_list is not None and cv_list_lengths is not None):
            encoder_out, encoder_mask, encoder_layers_out = self.encoder.forward_return_layers(speech, speech_lengths, cat_embs = cat_embs)
            encoded_cv = self.context_adaptor.encode_cv(cv_list, cv_list_lengths)
            encoder_out = encoder_out + self.context_adaptor(encoder_layers_out, encoded_cv)
        else:
            encoder_out, encoder_mask = self.encoder(speech, speech_lengths, cat_embs = cat_embs)

        encoder_out_lens = encoder_mask.squeeze(1).sum(1)

        # 2a. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc, ctc_probs = self.ctc(encoder_out, encoder_out_lens, text,
                                           text_lengths)
        else:
            loss_ctc, ctc_probs = None, None

        # 2b. Attention-decoder branch
        # use non blank (token level) embedding for decoder
        if self.apply_non_blank_embedding:
            assert self.ctc_weight != 0
            assert ctc_probs is not None
            encoder_out, encoder_mask = self.filter_blank_embedding(
                ctc_probs, encoder_out)
        if self.ctc_weight != 1.0:
            loss_att, acc_att = self._calc_att_loss(
                encoder_out, encoder_mask, text, text_lengths, {
                    "cat_embs": cat_embs,
                    "langs": batch["langs"],
                    "tasks": batch["tasks"]
                })
        else:
            loss_att = None
            acc_att = None

        if loss_ctc is None:
            loss = loss_att
        elif loss_att is None:
            loss = loss_ctc
        else:
            loss = self.ctc_weight * loss_ctc + (1 -
                                                 self.ctc_weight) * loss_att
        return {
            "loss": loss,
            "loss_att": loss_att,
            "loss_ctc": loss_ctc,
            "th_accuracy": acc_att,
        }

    @torch.jit.ignore(drop=True)
    def _forward_ctc(
            self, encoder_out: torch.Tensor, encoder_mask: torch.Tensor,
            text: torch.Tensor,
            text_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)
        loss_ctc, ctc_probs = self.ctc(encoder_out, encoder_out_lens, text,
                                       text_lengths)
        return loss_ctc, ctc_probs

    def filter_blank_embedding(
            self, ctc_probs: torch.Tensor,
            encoder_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = encoder_out.size(0)
        maxlen = encoder_out.size(1)
        top1_index = torch.argmax(ctc_probs, dim=2)
        indices = []
        for j in range(batch_size):
            indices.append(
                torch.tensor(
                    [i for i in range(maxlen) if top1_index[j][i] != 0]))

        select_encoder_out = [
            torch.index_select(encoder_out[i, :, :], 0,
                               indices[i].to(encoder_out.device))
            for i in range(batch_size)
        ]
        select_encoder_out = pad_sequence(select_encoder_out,
                                          batch_first=True,
                                          padding_value=0).to(
                                              encoder_out.device)
        xs_lens = torch.tensor([len(indices[i]) for i in range(batch_size)
                                ]).to(encoder_out.device)
        T = select_encoder_out.size(1)
        encoder_mask = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, 1, T)
        encoder_out = select_encoder_out
        return encoder_out, encoder_mask

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_mask: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        infos: Dict[str, List[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos,
                                            self.ignore_id)
        ys_in_lens = ys_pad_lens + 1
        # reverse the seq, used for right to left decoder
        r_ys_pad = reverse_pad_list(ys_pad, ys_pad_lens, float(self.ignore_id))
        r_ys_in_pad, r_ys_out_pad = add_sos_eos(r_ys_pad, self.sos, self.eos,
                                                self.ignore_id)
        # 1. Forward decoder
        if self.lsl_dec:
            cat_embs = infos.get('cat_embs', None) if infos else None
            decoder_out, r_decoder_out, _ = self.decoder(
                encoder_out, encoder_mask, ys_in_pad, ys_in_lens, r_ys_in_pad,
                self.reverse_weight, cat_embs)
        else:
            decoder_out, r_decoder_out, _ = self.decoder(
                encoder_out, encoder_mask, ys_in_pad, ys_in_lens, r_ys_in_pad,
                self.reverse_weight, None)

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

    def _forward_encoder(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
        cat_embs: Optional[torch.Tensor] = None,
        cv: Optional[torch.Tensor] = None,
        cv_lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Let's assume B = batch_size
        # 1. Encoder
        if simulate_streaming and decoding_chunk_size > 0:
            encoder_out, encoder_mask = self.encoder.forward_chunk_by_chunk(
                speech,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks
            )  # (B, maxlen, encoder_dim)
        else:
            encoder_out, encoder_mask = self.encoder(
                speech,
                speech_lengths,
                decoding_chunk_size=decoding_chunk_size,
                num_decoding_left_chunks=num_decoding_left_chunks,
                cat_embs=cat_embs,
            )  # (B, maxlen, encoder_dim)
        return encoder_out, encoder_mask

    @torch.jit.ignore(drop=True)
    def ctc_logprobs(self,
                     encoder_out: torch.Tensor,
                     blank_penalty: float = 0.0,
                     blank_id: int = 0):
        if blank_penalty > 0.0:
            logits = self.ctc.ctc_lo(encoder_out)
            logits[:, :, blank_id] -= blank_penalty
            ctc_probs = logits.log_softmax(dim=2)
        else:
            ctc_probs = self.ctc.log_softmax(encoder_out)

        return ctc_probs

    def decode(
        self,
        methods: List[str],
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        ctc_weight: float = 0.0,
        simulate_streaming: bool = False,
        reverse_weight: float = 0.0,
        context_graph: ContextGraph = None,
        blank_id: int = 0,
        blank_penalty: float = 0.0,
        length_penalty: float = 0.0,
        infos: Dict[str, List[str]] = None,
        cat_embs: Optional[torch.Tensor] = None,
        cv : Optional[torch.Tensor] = None,
        cv_lengths : Optional[torch.Tensor] = None,
    ) -> Dict[str, List[DecodeResult]]:
        """ Decode input speech

        Args:
            methods:(List[str]): list of decoding methods to use, which could
                could contain the following decoding methods, please refer paper:
                https://arxiv.org/pdf/2102.01547.pdf
                   * ctc_greedy_search
                   * ctc_prefix_beam_search
                   * atttention
                   * attention_rescoring
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion
            reverse_weight (float): right to left decoder weight
            ctc_weight (float): ctc score weight

        Returns: dict results of all decoding methods
        """
        assert speech.shape[0] == speech_lengths.shape[0]
        assert decoding_chunk_size != 0
        encoder_out, encoder_mask = self._forward_encoder(
            speech,
            speech_lengths,
            decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming,
            cat_embs=cat_embs,
            cv=cv,
            cv_lengths=cv_lengths)
        encoder_lens = encoder_mask.squeeze(1).sum(1)

        ctc_probs = self.ctc_logprobs(encoder_out, blank_penalty, blank_id)
        results = {}
        if 'attention' in methods:
            results['attention'] = attention_beam_search(self,
                                                         encoder_out,
                                                         encoder_mask,
                                                         beam_size,
                                                         length_penalty,
                                                         infos,
                                                         cat_embs=cat_embs)
        if 'ctc_greedy_search' in methods:
            results['ctc_greedy_search'] = ctc_greedy_search(
                ctc_probs, encoder_lens, blank_id)
        if 'ctc_prefix_beam_search' in methods:
            ctc_prefix_result = ctc_prefix_beam_search(ctc_probs, encoder_lens,
                                                       beam_size,
                                                       context_graph, blank_id)
            results['ctc_prefix_beam_search'] = ctc_prefix_result
        if 'attention_rescoring' in methods:
            # attention_rescoring depends on ctc_prefix_beam_search nbest
            if 'ctc_prefix_beam_search' in results:
                ctc_prefix_result = results['ctc_prefix_beam_search']
            else:
                ctc_prefix_result = ctc_prefix_beam_search(
                    ctc_probs, encoder_lens, beam_size, context_graph,
                    blank_id)
            if self.apply_non_blank_embedding:
                encoder_out, _ = self.filter_blank_embedding(
                    ctc_probs, encoder_out)
            results['attention_rescoring'] = attention_rescoring(
                self,
                ctc_prefix_result,
                encoder_out,
                encoder_lens,
                ctc_weight,
                reverse_weight,
                infos,
                cat_embs=cat_embs)
        if 'joint_decoding' in methods:
            results['joint_decoding'] = joint_decoding(self, encoder_out, encoder_lens,
                                                       ctc_probs, ctc_weight, beam_size,
                                                       length_bonus = length_penalty,
                                                       cat_embs = cat_embs)
        return results


    def load_hlg_resource_if_necessary(self, hlg, word):
        if not hasattr(self, 'hlg'):
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            self.hlg = k2.Fsa.from_dict(torch.load(hlg, map_location=device))
        if not hasattr(self.hlg, "lm_scores"):
            self.hlg.lm_scores = self.hlg.scores.clone()
        if not hasattr(self, 'word_table'):
            self.word_table = {}
            with open(word, 'r') as fin:
                for line in fin:
                    arr = line.strip().split()
                    assert len(arr) == 2
                    self.word_table[int(arr[1])] = arr[0]

    @torch.no_grad()
    def hlg_onebest(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
        hlg: str = '',
        word: str = '',
        symbol_table: Dict[str, int] = None,
    ) -> List[int]:
        self.load_hlg_resource_if_necessary(hlg, word)
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)
        ctc_probs = self.ctc.log_softmax(
            encoder_out)  # (1, maxlen, vocab_size)
        supervision_segments = torch.stack(
            (torch.arange(len(encoder_mask)), torch.zeros(len(encoder_mask)),
             encoder_mask.squeeze(dim=1).sum(dim=1).cpu()),
            1,
        ).to(torch.int32)
        lattice = get_lattice(nnet_output=ctc_probs,
                              decoding_graph=self.hlg,
                              supervision_segments=supervision_segments,
                              search_beam=20,
                              output_beam=7,
                              min_active_states=30,
                              max_active_states=10000,
                              subsampling_factor=4)
        best_path = one_best_decoding(lattice=lattice, use_double_scores=True)
        hyps = get_texts(best_path)
        hyps = [[symbol_table[k] for j in i for k in self.word_table[j]]
                for i in hyps]
        return hyps

    @torch.no_grad()
    def hlg_rescore(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        decoding_chunk_size: int = -1,
        num_decoding_left_chunks: int = -1,
        simulate_streaming: bool = False,
        lm_scale: float = 0,
        decoder_scale: float = 0,
        r_decoder_scale: float = 0,
        hlg: str = '',
        word: str = '',
        symbol_table: Dict[str, int] = None,
    ) -> List[int]:
        self.load_hlg_resource_if_necessary(hlg, word)
        device = speech.device
        encoder_out, encoder_mask = self._forward_encoder(
            speech, speech_lengths, decoding_chunk_size,
            num_decoding_left_chunks,
            simulate_streaming)  # (B, maxlen, encoder_dim)
        ctc_probs = self.ctc.log_softmax(
            encoder_out)  # (1, maxlen, vocab_size)
        supervision_segments = torch.stack(
            (torch.arange(len(encoder_mask)), torch.zeros(len(encoder_mask)),
             encoder_mask.squeeze(dim=1).sum(dim=1).cpu()),
            1,
        ).to(torch.int32)
        lattice = get_lattice(nnet_output=ctc_probs,
                              decoding_graph=self.hlg,
                              supervision_segments=supervision_segments,
                              search_beam=20,
                              output_beam=7,
                              min_active_states=30,
                              max_active_states=10000,
                              subsampling_factor=4)
        nbest = Nbest.from_lattice(
            lattice=lattice,
            num_paths=100,
            use_double_scores=True,
            nbest_scale=0.5,
        )
        nbest = nbest.intersect(lattice)
        assert hasattr(nbest.fsa, "lm_scores")
        assert hasattr(nbest.fsa, "tokens")
        assert isinstance(nbest.fsa.tokens, torch.Tensor)

        tokens_shape = nbest.fsa.arcs.shape().remove_axis(1)
        tokens = k2.RaggedTensor(tokens_shape, nbest.fsa.tokens)
        tokens = tokens.remove_values_leq(0)
        hyps = tokens.tolist()

        # cal attention_score
        hyps_pad = pad_sequence([
            torch.tensor(hyp, device=device, dtype=torch.long) for hyp in hyps
        ], True, self.ignore_id)  # (beam_size, max_hyps_len)
        ori_hyps_pad = hyps_pad
        hyps_lens = torch.tensor([len(hyp) for hyp in hyps],
                                 device=device,
                                 dtype=torch.long)  # (beam_size,)
        hyps_pad, _ = add_sos_eos(hyps_pad, self.sos, self.eos, self.ignore_id)
        hyps_lens = hyps_lens + 1  # Add <sos> at begining
        encoder_out_repeat = []
        tot_scores = nbest.tot_scores()
        repeats = [tot_scores[i].shape[0] for i in range(tot_scores.dim0)]
        for i in range(len(encoder_out)):
            encoder_out_repeat.append(encoder_out[i:i + 1].repeat(
                repeats[i], 1, 1))
        encoder_out = torch.concat(encoder_out_repeat, dim=0)
        encoder_mask = torch.ones(encoder_out.size(0),
                                  1,
                                  encoder_out.size(1),
                                  dtype=torch.bool,
                                  device=device)
        # used for right to left decoder
        r_hyps_pad = reverse_pad_list(ori_hyps_pad, hyps_lens, self.ignore_id)
        r_hyps_pad, _ = add_sos_eos(r_hyps_pad, self.sos, self.eos,
                                    self.ignore_id)
        reverse_weight = 0.5
        decoder_out, r_decoder_out, _ = self.decoder(
            encoder_out, encoder_mask, hyps_pad, hyps_lens, r_hyps_pad,
            reverse_weight)  # (beam_size, max_hyps_len, vocab_size)
        decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)
        decoder_out = decoder_out
        # r_decoder_out will be 0.0, if reverse_weight is 0.0 or decoder is a
        # conventional transformer decoder.
        r_decoder_out = torch.nn.functional.log_softmax(r_decoder_out, dim=-1)
        r_decoder_out = r_decoder_out

        decoder_scores = torch.tensor([
            sum([decoder_out[i, j, hyps[i][j]] for j in range(len(hyps[i]))])
            for i in range(len(hyps))
        ],
                                      device=device)
        r_decoder_scores = []
        for i in range(len(hyps)):
            score = 0
            for j in range(len(hyps[i])):
                score += r_decoder_out[i, len(hyps[i]) - j - 1, hyps[i][j]]
            score += r_decoder_out[i, len(hyps[i]), self.eos]
            r_decoder_scores.append(score)
        r_decoder_scores = torch.tensor(r_decoder_scores, device=device)

        am_scores = nbest.compute_am_scores()
        ngram_lm_scores = nbest.compute_lm_scores()
        tot_scores = am_scores.values + lm_scale * ngram_lm_scores.values + \
            decoder_scale * decoder_scores + r_decoder_scale * r_decoder_scores
        ragged_tot_scores = k2.RaggedTensor(nbest.shape, tot_scores)
        max_indexes = ragged_tot_scores.argmax()
        best_path = k2.index_fsa(nbest.fsa, max_indexes)
        hyps = get_texts(best_path)
        hyps = [[symbol_table[k] for j in i for k in self.word_table[j]]
                for i in hyps]
        return hyps

    @torch.jit.export
    def subsampling_rate(self) -> int:
        """ Export interface for c++ call, return subsampling_rate of the
            model
        """
        return self.encoder.embed.subsampling_rate

    @torch.jit.export
    def right_context(self) -> int:
        """ Export interface for c++ call, return right_context of the model
        """
        return self.encoder.embed.right_context

    @torch.jit.export
    def sos_symbol(self) -> int:
        """ Export interface for c++ call, return sos symbol id of the model
        """
        return self.sos

    @torch.jit.export
    def eos_symbol(self) -> int:
        """ Export interface for c++ call, return eos symbol id of the model
        """
        return self.eos

    @torch.jit.export
    def forward_encoder_chunk(
        self,
        xs: torch.Tensor,
        offset: int,
        required_cache_size: int,
        att_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        cnn_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        cat_embs: Optional[torch.Tensor] = None,
        verbose: bool = False,
        encoded_cv: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Export interface for c++ call, give input chunk xs, and return
            output from time 0 to current chunk.

        Args:
            xs (torch.Tensor): chunk input, with shape (b=1, time, mel-dim),
                where `time == (chunk_size - 1) * subsample_rate + \
                        subsample.right_context + 1`
            offset (int): current offset in encoder output time stamp
            required_cache_size (int): cache size required for next chunk
                compuation
                >=0: actual cache size
                <0: means all history cache is required
            att_cache (torch.Tensor): cache tensor for KEY & VALUE in
                transformer/conformer attention, with shape
                (elayers, head, cache_t1, d_k * 2), where
                `head * d_k == hidden-dim` and
                `cache_t1 == chunk_size * num_decoding_left_chunks`.
            cnn_cache (torch.Tensor): cache tensor for cnn_module in conformer,
                (elayers, b=1, hidden-dim, cache_t2), where
                `cache_t2 == cnn.lorder - 1`

        Returns:
            torch.Tensor: output of current input xs,
                with shape (b=1, chunk_size, hidden-dim).
            torch.Tensor: new attention cache required for next chunk, with
                dynamic shape (elayers, head, ?, d_k * 2)
                depending on required_cache_size.
            torch.Tensor: new conformer cnn cache required for next chunk, with
                same shape as the original cnn_cache.

        """

        if verbose:
            print(self.add_cat_embs)
            print(self.get_cat_labels())
            print(cat_embs)

        if self.add_cat_embs:
            assert cat_embs is not None
            if verbose:
                print(cat_embs.size(0))
                print(len(self.get_cat_labels()))
                print(xs.shape)

            #assert cat_embs.size(0) == len(self.cat_labels)

            # cat_embs.size() = [n,]
            # unsqueezes --> [1, 1, n]
            # expand --> [batch, time, n]
            per_frame_cat_embs = cat_embs.unsqueeze(0).unsqueeze(0).expand(
                xs.size(0), xs.size(1), -1)
            xs = torch.cat((xs, per_frame_cat_embs), 2)
            if verbose:
                print(xs.shape)

        if self.lsl_enc:
            if verbose:
                print("passing cat_emb to encoder")
        else:
            cat_embs = None

        if self.context_adaptor is not None and encoded_cv is not None:
            if verbose:
                print("encoder deep biasing")
            e_out, e_layers_out, r_att_cache, r_cnn_cache = self.encoder.forward_chunk_return_layers(xs,
                                                                                                     offset,
                                                                                                     required_cache_size,
                                                                                                     att_cache,
                                                                                                     cnn_cache,
                                                                                                     cat_embs=cat_embs)
            e_out = e_out + self.context_adaptor(e_layers_out, encoded_cv)
            return (e_out, r_att_cache, r_cnn_cache)
        else:
            return self.encoder.forward_chunk(xs, offset, required_cache_size,
                                              att_cache, cnn_cache, cat_embs=cat_embs)

    @torch.jit.export
    def encode_cv(self,
                  cv: torch.Tensor,
                  cv_lengths: torch.Tensor):
        if self.context_adaptor is not None:
            return self.context_adaptor.encode_cv(cv, cv_lengths)
        else:
            return None

    # Original method from WeNet
    @torch.jit.export
    def ctc_activation_orig(self, xs: torch.Tensor) -> torch.Tensor:
        return self.ctc.log_softmax(xs)

    # Patched version
    @torch.jit.export
    def ctc_activation(self, xs: torch.Tensor) -> torch.Tensor:
        return self.ctc_activation_orig(xs)

    # Patched version, with instrumentation
    @torch.jit.export
    def ctc_activation_more(
        self, xs: torch.Tensor, missing_blank_from_top2_threshold: float,
        non_blank_prob_threshold: float, top_blank_prob_penalty: float
    ) -> Tuple[torch.Tensor, int, int, float, bool, int]:
        """ Export interface for c++ call, apply linear transform and log
            softmax before ctc
        Args:
            xs (torch.Tensor): encoder output
            missing_blank_from_top2_threshold : float : ratio of frames without a blank in any of the top-2 positions of the ctc probs.
            non_blank_prob_threshold : float : if the non-blank in 2nd place has a logprob larger than this value, let's apply the hack
            top_blank_prob_penalty : float : logprob penalty to apply to the blank token when the hack triggers

        Returns:
            torch.Tensor: activation before ctc
            int : number of frames in this chunk
            int : number of frames with non blank in both top and 2nd position
            float : ration of non-blank in top 2
            bool : did the condition matched to apply the hack or not
            int : number of frames where the blank token probability were impacted

        """
        #return self.ctc.log_softmax(xs)
        # ugly slow hack
        verbose = False
        apply_hack = False

        maxlen = xs.size(1)
        ctc_probs = self.ctc.log_softmax(xs).squeeze(0)
        head_vals, head_index = ctc_probs.topk(2)

        ## This commented out section was restricting the number of cases where the hack
        ## would apply, but didn't changed the accuracy drops as expected, so we'll remove this for now
        ## until the instrumentation would allow us to understand better
        # top_nonblank = (head_index[:,0] != 0).sum()

        # vF, first_top_non_blank = torch.max(head_index[:,0] != 0, dim=0)
        # vL, last_top_non_blank = torch.max(torch.flip(head_index[:,0], dims=(0,)) != 0, dim=0)
        # match = vF and vL
        # eff_len = maxlen - (last_top_non_blank + first_top_non_blank) + 1
        # if top_nonblank > 0:
        #     dd = top_nonblank
        # else:
        #     dd = torch.tensor(1)

        # f_eff_len = int(eff_len)
        # f_top_nonblank = int(top_nonblank)
        # f_ratio = float((eff_len-top_nonblank) / dd)
        # if verbose:
        #     print(f"len {maxlen}, eff_len = {eff_len}, top-non-blank = {top_nonblank}, match {match}, loss_nsp {f_ratio}")

        # apply_hack2 = dd > 7 and maxlen > 24

        # TODO : have this as external parameters to the method
        threshold_blank_missing_from_top2 = missing_blank_from_top2_threshold
        total_only_nonblank_in_top2 = torch.all(head_index > 0, dim=1).sum()
        ratio_non_blank_in_top2 = total_only_nonblank_in_top2 / maxlen

        #if True and ratio_non_blank_in_top2 < threshold_blank_missing_from_top2 and maxlen > 24:
        if True and ratio_non_blank_in_top2 < threshold_blank_missing_from_top2:
            apply_hack = True

        f_maxlen = int(maxlen)
        f_total_only_nonblank_in_top2 = int(total_only_nonblank_in_top2)
        f_ratio_non_blank_in_top2 = float(ratio_non_blank_in_top2)
        if verbose:
            #print(f"JPR applying-hack : decision {f_total_only_nonblank_in_top2} / {f_maxlen} = {f_ratio_non_blank_in_top2}, {apply_hack}, apply 2 : {apply_hack2}")
            print(
                f"JPR applying-hack : decision {f_total_only_nonblank_in_top2} / {f_maxlen} = {f_ratio_non_blank_in_top2}, {apply_hack}"
            )

        changes = 0
        if apply_hack:
            #if apply_hack and apply_hack2: # v14
            #print("JPR applying-hack")
            for t in range(0, maxlen):
                logp = ctc_probs[t]  # (vocab_size,)
                top_k_logp, top_k_index = logp.topk(2)  # (beam_size,)
                if top_k_index[0] == 0:  # best token is blank
                    #if top_k_logp[1] > -2: #2nd best token has a good-enough prob (v8)
                    #if top_k_logp[1] > -2 and top_k_logp[1] <= -1: #2nd best token has a good-enough prob # v9
                    if top_k_logp[1] > non_blank_prob_threshold:
                        changes += 1
                        logp[0] -= top_blank_prob_penalty
                        # leaving traces below for now
                        # logp[0] = -100.0 # push the value of blank down the drain #v8, v9
                        #logp[0] -= 10.0 # push the value of blank down the drain, v10, good for 18L
                        #logp[0] -= 7.0 # push the value of blank down the drain, v11
                        #logp[0] -= 12.0 # push the value of blank down the drain, v12
                        #logp[0] -= 10.0 # push the value of blank down the drain, v13

                # new version of the hack prefers not have this leg below
                # elif top_k_logp[0] > -2:
                #     logp[0] = -100.0 # push the value of blank down the drain
            if verbose:
                print(
                    f"JPR applying-hack effective: changes/maxlen = {changes}/{maxlen}, ratio_non_blank_in_top2 = {ratio_non_blank_in_top2}"
                )
        elif verbose:
            print("JPR not-applying-hack")

        ctc_probs = ctc_probs.unsqueeze(0)

        #return ctc_probs
        return ctc_probs, maxlen, f_total_only_nonblank_in_top2, f_ratio_non_blank_in_top2, apply_hack, changes

    @torch.jit.export
    def is_bidirectional_decoder(self) -> bool:
        """
        Returns:
            torch.Tensor: decoder output
        """
        if self.decoder is not None:
            if hasattr(self.decoder, 'right_decoder'):
                return True
            else:
                return False
        else:
            return False

    @torch.jit.export
    def get_cat_labels(self) -> List[str]:
        """
        Returns:
            List[str]: labels of categories, in order, for use with add_cat_embs and pass_cat_embs
        """
        if hasattr(self, 'cat_labels'):
            return self.cat_labels
        else:
            return []

    @torch.jit.export
    def forward_attention_decoder(
        self,
        hyps: torch.Tensor,
        hyps_lens: torch.Tensor,
        encoder_out: torch.Tensor,
        reverse_weight: float = 0,
        cat_embs: Optional[torch.Tensor] = None,
        verbose: bool = False
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """ Export interface for c++ call, forward decoder with multiple
            hypothesis from ctc prefix beam search and one encoder output
        Args:
            hyps (torch.Tensor): hyps from ctc prefix beam search, already
                pad sos at the begining
            hyps_lens (torch.Tensor): length of each hyp in hyps
            encoder_out (torch.Tensor): corresponding encoder output
            r_hyps (torch.Tensor): hyps from ctc prefix beam search, already
                pad eos at the begining which is used fo right to left decoder
            reverse_weight: used for verfing whether used right to left decoder,
            > 0 will use.

        Returns:
            torch.Tensor: decoder output
        """
        assert encoder_out.size(0) == 1
        num_hyps = hyps.size(0)
        assert hyps_lens.size(0) == num_hyps
        encoder_out = encoder_out.repeat(num_hyps, 1, 1)
        encoder_mask = torch.ones(num_hyps,
                                  1,
                                  encoder_out.size(1),
                                  dtype=torch.bool,
                                  device=encoder_out.device)

        # input for right to left decoder
        # this hyps_lens has count <sos> token, we need minus it.
        r_hyps_lens = hyps_lens - 1
        # this hyps has included <sos> token, so it should be
        # convert the original hyps.
        r_hyps = hyps[:, 1:]
        #   >>> r_hyps
        #   >>> tensor([[ 1,  2,  3],
        #   >>>         [ 9,  8,  4],
        #   >>>         [ 2, -1, -1]])
        #   >>> r_hyps_lens
        #   >>> tensor([3, 3, 1])

        # NOTE(Mddct): `pad_sequence` is not supported by ONNX, it is used
        #   in `reverse_pad_list` thus we have to refine the below code.
        #   Issue: https://github.com/wenet-e2e/wenet/issues/1113
        # Equal to:
        #   >>> r_hyps = reverse_pad_list(r_hyps, r_hyps_lens, float(self.ignore_id))
        #   >>> r_hyps, _ = add_sos_eos(r_hyps, self.sos, self.eos, self.ignore_id)
        max_len = torch.max(r_hyps_lens)
        index_range = torch.arange(0, max_len, 1).to(encoder_out.device)
        seq_len_expand = r_hyps_lens.unsqueeze(1)
        seq_mask = seq_len_expand > index_range  # (beam, max_len)
        #   >>> seq_mask
        #   >>> tensor([[ True,  True,  True],
        #   >>>         [ True,  True,  True],
        #   >>>         [ True, False, False]])
        index = (seq_len_expand - 1) - index_range  # (beam, max_len)
        #   >>> index
        #   >>> tensor([[ 2,  1,  0],
        #   >>>         [ 2,  1,  0],
        #   >>>         [ 0, -1, -2]])
        index = index * seq_mask
        #   >>> index
        #   >>> tensor([[2, 1, 0],
        #   >>>         [2, 1, 0],
        #   >>>         [0, 0, 0]])
        r_hyps = torch.gather(r_hyps, 1, index)
        #   >>> r_hyps
        #   >>> tensor([[3, 2, 1],
        #   >>>         [4, 8, 9],
        #   >>>         [2, 2, 2]])
        r_hyps = torch.where(seq_mask, r_hyps, self.eos)
        #   >>> r_hyps
        #   >>> tensor([[3, 2, 1],
        #   >>>         [4, 8, 9],
        #   >>>         [2, eos, eos]])
        r_hyps = torch.cat([hyps[:, 0:1], r_hyps], dim=1)
        #   >>> r_hyps
        #   >>> tensor([[sos, 3, 2, 1],
        #   >>>         [sos, 4, 8, 9],
        #   >>>         [sos, 2, eos, eos]])

        if self.decoder is not None:
            if self.lsl_dec:
                if verbose:
                    print("passing cat_emb to decoder")
                    print(cat_embs)
                decoder_out, r_decoder_out, _ = self.decoder(
                    encoder_out, encoder_mask, hyps, hyps_lens, r_hyps,
                    reverse_weight,
                    cat_embs)  # (num_hyps, max_hyps_len, vocab_size)
            else:
                decoder_out, r_decoder_out, _ = self.decoder(
                    encoder_out, encoder_mask, hyps, hyps_lens, r_hyps,
                    reverse_weight,
                    None)  # (num_hyps, max_hyps_len, vocab_size)
            decoder_out = torch.nn.functional.log_softmax(decoder_out, dim=-1)

            # right to left decoder may be not used during decoding process,
            # which depends on reverse_weight param.
            # r_dccoder_out will be 0.0, if reverse_weight is 0.0
            r_decoder_out = torch.nn.functional.log_softmax(r_decoder_out,
                                                            dim=-1)
        else:
            decoder_out, r_decoder_out = None, None
        return decoder_out, r_decoder_out

    def onmt_attention_decoding(
        self,
        encoder_out: torch.Tensor,
        beam_size: int = 5,
        reverse_weight: float = 0,
        cat_embs: Optional[torch.Tensor] = None,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # defaults copied from ONMT, except beam size
        # Alpha and Beta values for Google Length + Coverage penalty
        # Described here: https://arxiv.org/pdf/1609.08144.pdf, Section 7
        # When α = 0 and β = 0, our decoder falls back to pure beam search by probability.
        scorer = GNMTGlobalScorer(alpha=1., beta=0., length_penalty='avg', coverage_penalty='none')
        decode_strategy = BeamSearch(
            beam_size,
            batch_size=encoder_out.size(0),
            pad=0, #<blank>
            bos=10000, #<sos/eos>
            eos=10000, #<sos/eos>
            unk=1, #<unk> - 29 is <unknown>
            start=10000, # do we start with anything??
            n_best=1,
            global_scorer=scorer,
            min_length=0,
            max_length=encoder_out.size(1),
            return_attention=True,
            block_ngram_repeat=0,
            stepwise_penalty=False,
            ratio=0.0,
            ban_unk_token=False,
        )

        batch_size = encoder_out.size(0)
        maxlen = encoder_out.size(1)
        encoder_dim = encoder_out.size(2)
        running_size = batch_size * beam_size
        encoder_mask = torch.ones(running_size, 1, maxlen)  # (B*N, 1, max_len)
        cache: Optional[List[torch.Tensor]] = None

        if self.lsl_dec and cat_embs is not None:
            if verbose:
                print("passing cat_emb to decoder")
        elif cat_embs is not None:
            if verbose:
                print("NOT passing cat_emb to decoder")
        elif self.lsl_dec:
            # cat_embs not actually getting passed by speech2ctm currently?
            # this case is for: model expecting cat_embs but this function recieves None
            cat_embs = torch.tensor([1., 0.])
            if verbose:
                print("setting cat_emb to [1,0] and passing to decoder")

        # Decoder forward step by step
        (fn_map_state, encoder_out, src_len_tiled) = decode_strategy.initialize(
            encoder_out, torch.LongTensor([maxlen])
        )

        for step in range(decode_strategy.max_length):
            decoder_input = decode_strategy.alive_seq

            hyps_mask = subsequent_mask(step+1).unsqueeze(0).repeat(
                running_size, 1, 1)  # (B*N, i, i)
            log_probs, cache, attns = self.decoder.forward_one_step_with_attn(
                encoder_out, encoder_mask, decoder_input, hyps_mask, cache, cat_embs=cat_embs)
            attn = torch.stack(attns, 0)
            attn = attn.mean(0).mean(1) # mean over layers and attention heads

            decode_strategy.advance(log_probs, attn)
            any_finished = decode_strategy.is_finished.any()
            if any_finished:
                decode_strategy.update_finished()
                if decode_strategy.done:
                    break

            select_indices = decode_strategy.select_indices

            if any_finished:
                # Reorder states.
                if isinstance(encoder_out, tuple):
                    encoder_out = tuple(x.index_select(0, select_indices) for x in encoder_out)
                else:
                    encoder_out = encoder_out.index_select(0, select_indices)
                for i in range(len(cache)):
                    cache[i] = cache[i].index_select(0, select_indices)

                src_len_tiled = src_len_tiled.index_select(0, select_indices)

        hyps = decode_strategy.predictions
        scores = decode_strategy.scores
        # note: this version of attention decoding doesn't save token scores - we could add that later
        return hyps[0][0].unsqueeze(0), scores[0][0].unsqueeze(0), scores[0][0].unsqueeze(0).unsqueeze(1)

    def espnet_joint_decoding(
        self,
        encoder_out: torch.Tensor,
        beam_size: int = 5,
        reverse_weight: float = 0,
        cat_embs: Optional[torch.Tensor] = None,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # outputs: n_best_hyps, n_best_scores, one_best_confs, one_best_start_frames, one_best_end_frames

        if cat_embs is None:
            # cat_embs not actually getting passed by speech2ctm currently?
            cat_embs = torch.tensor([1., 0.])
            #if verbose:
            print("setting cat_emb to [1,0] and passing to decoder")

        log_probs = self.ctc.log_softmax(encoder_out).unsqueeze(0)
        decoder = self.decoder.left_decoder

        '''
        # ctc_weight and bonus chosen through experimentation
        ctc_weight = 0.5
        length_bonus = 0.1
        pre_beam_ratio = 1. # default was 1.5 in ESPnet
        '''

        weights = dict(
            decoder=1.0 - self.joint_ctc_weight,
            ctc=self.joint_ctc_weight,
            length_bonus=self.length_bonus,
        )

        beam_search = BeamSearchTimeSync(
            sos=10000,
            beam_size=beam_size,
            ctc_probs=log_probs,
            decoder=decoder,
            weights=weights,
            words=self.words,
            word_prefixes = self.word_prefixes,
            tok_to_str=self.tok_to_str,
            pre_beam_ratio=self.pre_beam_ratio,
        )

        n_best_hyps, n_best_scores, start_times, end_times, n_best_confs = beam_search(x = encoder_out, cat_embs = cat_embs)
        return n_best_hyps[0][1:].unsqueeze(0), n_best_scores[0].unsqueeze(0), n_best_confs[0][1:], start_times[0][0,1:], end_times[0][0,1:]

    @torch.jit.export
    def attention_decoding(
        self,
        encoder_out: torch.Tensor,
        beam_size: int = 5,
        reverse_weight: float = 0,
        cat_embs: Optional[torch.Tensor] = None,
        verbose: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        method = "espnet"
        assert method in ["onmt", "espnet"]

        if method == "onmt":
            hyps, scores, token_scores = self.onmt_attention_decoding(encoder_out, beam_size, reverse_weight, cat_embs=cat_embs, verbose=verbose)
            start_times, end_times = (torch.tensor([0.]), torch.tensor([0.]))
        else:
            hyps, scores, token_scores, start_times, end_times = self.espnet_joint_decoding(encoder_out, beam_size, reverse_weight, cat_embs=cat_embs, verbose=verbose)
        return hyps, token_scores, start_times, end_times

def init_asr_model(configs):
    input_dim = configs['input_dim']
    vocab_size = configs['output_dim']

    if configs['cmvn_file'] is not None:
        mean, istd = load_cmvn(configs['cmvn_file'], configs['is_json_cmvn'])
        if 'add_cat_emb' in configs['dataset_conf'] and configs[
                'dataset_conf']['add_cat_emb']:
            # if we add a category embedding to every frame, then the CMVN matrices need to be bigger
            # to match the new feature size
            # we use mean=zero and std=1 so that CMVN doesn't impact these features
            emb_len = configs['dataset_conf']['cat_emb_conf']['emb_len']
            extra_zeros = np.zeros((emb_len, ))
            mean = np.concatenate((mean, extra_zeros))
            extra_ones = np.ones((emb_len, ))
            istd = np.concatenate((istd, extra_ones))

        global_cmvn = GlobalCMVN(
            torch.from_numpy(mean).float(),
            torch.from_numpy(istd).float())
    else:
        global_cmvn = None

    encoder_type = configs.get('encoder', 'conformer')
    decoder_type = configs.get('decoder', 'bitransformer')

    language_specific_layers = configs['dataset_conf'].get(
        'pass_cat_emb', False)

    if language_specific_layers:
        configs['encoder_conf']['num_langs'] = configs['dataset_conf'][
            'cat_emb_conf']['emb_len']
        encoder = LanguageSpecificConformerEncoder(input_dim,
                                                   global_cmvn=global_cmvn,
                                                   **configs['encoder_conf'])
    elif encoder_type == 'conformer':
        encoder = ConformerEncoder(input_dim,
                                   global_cmvn=global_cmvn,
                                   **configs['encoder_conf'])
    else:
        encoder = TransformerEncoder(input_dim,
                                     global_cmvn=global_cmvn,
                                     **configs['encoder_conf'])

    if language_specific_layers:
        configs['decoder_conf']['num_langs'] = configs['dataset_conf'][
            'cat_emb_conf']['emb_len']
        decoder = LanguageSpecificTransformerDecoder(vocab_size,
                                                     encoder.output_size(),
                                                     **configs['decoder_conf'])
    elif decoder_type == 'transformer':
        decoder = TransformerDecoder(vocab_size, encoder.output_size(),
                                     **configs['decoder_conf'])
    else:
        assert 0.0 < configs['model_conf']['reverse_weight'] < 1.0
        assert configs['decoder_conf']['r_num_blocks'] > 0
        decoder = BiTransformerDecoder(vocab_size, encoder.output_size(),
                                       **configs['decoder_conf'])

    if 'focal_ctc' in configs:
        focal_cfg = configs['focal_ctc']
        # if the section is present, we assume it is enabled unless specified otherwise
        do_focal = focal_cfg['enabled'] if 'enabled' in focal_cfg else True
        print(f"Focal CTC will {'be' if do_focal else 'not be'} enabled")
        focal_alpha = focal_cfg['alpha'] if 'alpha' in focal_cfg else 0.5
        focal_gamma = focal_cfg['gamma'] if 'gamma' in focal_cfg else 2
        ctc = CTC(vocab_size,
                  encoder.output_size(),
                  do_focal_loss=do_focal,
                  focal_alpha=focal_alpha,
                  focal_gamma=focal_gamma)
    else:
        do_focal = False
        print(f"Focal CTC will {'be' if do_focal else 'not be'} enabled")
        ctc = CTC(vocab_size, encoder.output_size(), do_focal_loss=do_focal)

    model = ASRModel(
        vocab_size=vocab_size,
        encoder=encoder,
        decoder=decoder,
        ctc=ctc,
        **configs['model_conf'],
    )
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
        model.cat_labels = [
            ""
        ] * configs['dataset_conf']['cat_emb_conf']['emb_len']
        for k in configs['dataset_conf']['cat_emb_conf']['one_hot_ids']:
            model.cat_labels[configs['dataset_conf']['cat_emb_conf']
                             ['one_hot_ids'][k]] = k

    return model
