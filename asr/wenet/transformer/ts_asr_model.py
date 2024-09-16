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

from collections import defaultdict
from typing import List, Optional, Tuple, Dict, Any

import torch

# from torch.nn.utils.rnn import pad_sequence

# from wenet.transformer.cmvn import GlobalCMVN
# from wenet.transformer.ctc import CTC
# from wenet.transformer.decoder import (TransformerDecoder,
#                                        BiTransformerDecoder)
from wenet.transformer.asr_model import ASRModel
from wenet.utils.common import (IGNORE_ID, add_sos_eos, log_add,
                                remove_duplicates_and_blank, th_accuracy,
                                reverse_pad_list)
# from wenet.utils.mask import (make_pad_mask, mask_finished_preds,
#                               mask_finished_scores, subsequent_mask)


class TeacherStudentASRModel(torch.nn.Module):
    """CTC-attention hybrid Encoder-Decoder model"""
    def __init__(
        self,
        vocab_size: int,
        teacher: ASRModel,
        student: ASRModel,
        ts_weight: float = 0.5,
        min_ts_weight: float = 0,
        reg_weight: float = float('nan'),
        oscillate_ts_weight: bool = False,
        decrease_every : int = -1,
        decrease_factor : float = 1,
        top_k_entries : int = -1,
        teacher_yaml : str = None,
        teacher_checkpoint : str = None,
    ):
        #assert 0.0 <= ts_weight <= 1.0, ts_weight
        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.vocab_size = vocab_size
        self.teacher = teacher
        self.student = student
        self.ts_weight = ts_weight
        self.min_ts_weight = min_ts_weight
        self.oscillate_ts_weight = oscillate_ts_weight
        # self.mse_loss = torch.nn.MSELoss(reduction="none")
        #self.mse_loss = torch.nn.KLDivLoss(reduction="none", log_target=True)
        self.kl_loss = torch.nn.KLDivLoss(reduction="sum", log_target=True)
        self.kl_loss2 = torch.nn.KLDivLoss(reduction="sum", log_target=False)

        self.decrease_every = decrease_every
        self.decrease_factor = decrease_factor 
        self.steps_forward = 0

        self.top_k_enties = top_k_entries
        self.show_details = True

        if reg_weight != reg_weight:
            if ts_weight > 1:
                self.reg_weight = 1
            else:
                self.reg_weight = 1 - ts_weight
        else:
            self.reg_weight = reg_weight

        self.teacher.eval()

    def forward(
        self,
        batch: dict,
        device: torch.device,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Frontend + Encoder + Decoder + Calc loss + Teacher vs Student outputs MSE

        Args:
            batch :  
            device :
        """
        speech = batch['feats'].to(device)
        speech_lengths = batch['feats_lengths'].to(device)
        text = batch['target'].to(device)
        text_lengths = batch['target_lengths'].to(device)
        text_lengths = batch['target_lengths'].to(device)

        if 'cat_embs' in batch:
            cat_embs = batch['cat_embs'].to(device)
        else:
            cat_embs = None
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (speech.shape[0] == speech_lengths.shape[0] == text.shape[0] ==
                text_lengths.shape[0]), (speech.shape, speech_lengths.shape,
                                         text.shape, text_lengths.shape)


        ys_in_pad, ys_out_pad = add_sos_eos(text, self.teacher.sos, self.teacher.eos, self.teacher.ignore_id)
        ys_in_lens = text_lengths + 1
        r_ys_pad = reverse_pad_list(text, text_lengths, float(self.teacher.ignore_id))
        r_ys_in_pad, r_ys_out_pad = add_sos_eos(r_ys_pad, self.teacher.sos, self.teacher.eos, self.teacher.ignore_id)
        # TS.0 Let's get the Teacher outputs
        with torch.no_grad():
            teacher_enc_output, teacher_enc_output_mask  = self.teacher.encoder(speech, speech_lengths, cat_embs=cat_embs)
            teacher_enc_ctc_outputs = self.teacher.ctc.log_softmax(teacher_enc_output)
            teacher_dec_out, teacher_r_dec_out, _ = self.teacher.decoder(teacher_enc_output, teacher_enc_output_mask,
                                                                        ys_in_pad, ys_in_lens,
                                                                        r_ys_in_pad,
                                                                        self.teacher.reverse_weight, cat_embs=cat_embs)

        # print(f"teacher dec_out shape {teacher_dec_out.shape}")
        # print(f"teacher encoder output shape: {teacher_enc_output.shape}, teacher_enc_output_mask shape: {teacher_enc_output_mask.shape}, teacher_enc_ctc_outputs shape: {teacher_enc_ctc_outputs.shape}")
        # print("teacher enc mask")
        # print(teacher_enc_output_mask)

        # 1. Encoder
        encoder_out, encoder_mask = self.student.encoder(speech, speech_lengths, cat_embs=cat_embs)
        encoder_out_lens = encoder_mask.squeeze(1).sum(1)

        # TS.1 Let's get the student encoder ctc outputs
        student_enc_ctc_outputs = self.student.ctc.log_softmax(encoder_out)
        # print(f"student encoder_out shape: {encoder_out.shape}, encoder_mask shape: {encoder_mask.shape}, student_enc_ctc_outputs shape: {student_enc_ctc_outputs.shape}")
        # print("student enc mask")
        # print(encoder_mask)

        if self.show_details:
            print(f"We will do top k {self.top_k_enties} entries")

        if self.top_k_enties > 0:
            student_top_k_vals, student_top_k_inds = torch.topk(student_enc_ctc_outputs, self.top_k_enties, dim=2)
            teacher_top_k_vals, teacher_top_k_inds = torch.topk(teacher_enc_ctc_outputs, self.top_k_enties, dim=2)

            # This isn't a typo : we gather the top k entries from the student using the top k indices from the teacher
            # and vice-versa
            xs = student_enc_ctc_outputs.gather(2,teacher_top_k_inds)
            xt = teacher_enc_ctc_outputs.gather(2,student_top_k_inds)

            # let just use the top K elements of both teacher and student.  It's not useful to 
            # try to constrain the student to match all the little details of the teacher that 
            # won't contribute to the sequence

            kl_enc_loss_B1 = self.kl_loss(xs, teacher_top_k_vals)
            kl_enc_loss_B2 = self.kl_loss(student_top_k_vals, xt)

            # kl_enc_loss_B = kl_enc_loss_B1 
            kl_enc_loss = (kl_enc_loss_B1 + kl_enc_loss_B2)/2
            if self.show_details:
                okl_enc_loss = self.kl_loss(student_enc_ctc_outputs, teacher_enc_ctc_outputs)
                print(f"ENCODER : we did top k and got {kl_enc_loss} instead of {okl_enc_loss}") 
        else:
            kl_enc_loss = self.kl_loss(student_enc_ctc_outputs, teacher_enc_ctc_outputs)

        kl_enc_loss = kl_enc_loss / encoder_mask.sum()

        # TS.2 Let's get the student decoder outputs
        student_dec_out, student_r_dec_out, _ = self.student.decoder(encoder_out, encoder_mask,
                                                                        ys_in_pad, ys_in_lens,
                                                                        r_ys_in_pad,
                                                                        self.student.reverse_weight, cat_embs=cat_embs)

        # print(f"student dec_out shape {student_dec_out.shape}")
        # encoder_mask_float = encoder_mask.squeeze(1).float()

        if self.top_k_enties > 0:

            student_dec_true_outputs = torch.log_softmax(student_dec_out, dim=-1)
            teacher_dec_true_outputs = torch.log_softmax(teacher_dec_out, dim=-1)

            student_top_k_vals, student_top_k_inds = torch.topk(student_dec_true_outputs, self.top_k_enties, dim=2)
            teacher_top_k_vals, teacher_top_k_inds = torch.topk(teacher_dec_true_outputs, self.top_k_enties, dim=2)

            # This isn't a typo : we gather the top k entries from the student using the top k indices from the teacher
            # and vice-versa
            xs = student_dec_true_outputs.gather(2,teacher_top_k_inds)
            xt = teacher_dec_true_outputs.gather(2,student_top_k_inds)

            kl_dec_loss_B1 = self.kl_loss(xs, teacher_top_k_vals)
            kl_dec_loss_B2 = self.kl_loss(student_top_k_vals, xt)
            kl_dec_loss = (kl_dec_loss_B1 + kl_dec_loss_B2)/2
            if self.show_details:
                okl_dec_loss = self.kl_loss(torch.log_softmax(student_dec_out, dim=-1), torch.log_softmax(teacher_dec_out, dim=-1))
                print(f"DECODER : we did top k and got {kl_dec_loss} instead of {okl_dec_loss}") 
        else:
            kl_dec_loss = self.kl_loss(torch.log_softmax(student_dec_out, dim=-1), torch.log_softmax(teacher_dec_out, dim=-1))

        kl_dec_loss = kl_dec_loss / encoder_mask.sum()

        # print(f"> mse loss = {mse_loss}")
        # print(f"mse loss = {mse_loss}")

        # 2a. CTC branch
        if self.student.ctc_weight != 0.0:
            loss_ctc, ctc_probs = self.student.ctc(encoder_out, encoder_out_lens, text,
                                           text_lengths)
        else:
            loss_ctc, ctc_probs = None, None


        # 2b. Attention-decoder branch
        # use non blank (token level) embedding for decoder
        if self.student.apply_non_blank_embedding:
            assert self.student.ctc_weight != 0
            assert ctc_probs is not None
            encoder_out, encoder_mask = self.student.filter_blank_embedding(
                ctc_probs, encoder_out)

        if self.student.ctc_weight != 1.0:
            loss_att, acc_att = self.student._calc_att_loss(
                encoder_out, encoder_mask, text, text_lengths, {
                    "cat_embs": cat_embs,
                    "langs": batch["langs"],
                    "tasks": batch["tasks"]
                })
        else:
            loss_att = None
            acc_att = -1.0

        if loss_ctc is None:
            loss = loss_att
        elif loss_att is None:
            loss = loss_ctc
        else:
            loss = self.student.ctc_weight * loss_ctc + (1 -
                                                 self.student.ctc_weight) * loss_att

        ts_loss = kl_enc_loss  * self.student.ctc_weight + (1 - self.student.ctc_weight) * kl_dec_loss
        ts_loss = ts_loss * self.ts_weight + loss * self.reg_weight

        if self.show_details:
            self.show_details = False

        if self.student.training and self.decrease_every > 0:
            self.steps_forward += 1
            if self.steps_forward >= self.decrease_every:
                self.steps_forward = 0
                self.ts_weight = ((self.ts_weight - self.min_ts_weight) * self.decrease_factor) + self.min_ts_weight

        #print(f"mseloss / ctc loss = {mse_loss / loss_ctc}")
        return {"loss":ts_loss, 'ts_weight': self.ts_weight, 'kl_enc_loss':kl_enc_loss,'kl_dec_loss':kl_dec_loss, "student_loss":loss, "loss_att":loss_att, "loss_ctc": loss_ctc, "th_accuracy": acc_att}

    # def eval(self):
    #     self.student.eval()
    #     self.teacher.eval()

    # def train(self):
    #     self.student.train()
    #     # we don't train the teacher, never
    #     self.teacher.eval()

    def train(self, mode: bool = True):
        self.training = mode
        for module in self.student.children():
            module.train(mode)

        for module in self.teacher.children():
            module.train(False)
        return self

    def eval(self):
        return self.train(False)

def init_ts_asr_model(teacher_model, student_model, configs):

    input_dim = configs['input_dim']
    vocab_size = configs['output_dim']

    model = TeacherStudentASRModel(
        vocab_size=vocab_size,
        teacher=teacher_model,
        student=student_model,
        **configs['ts_conf'],
    )
    return model
