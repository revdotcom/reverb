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

from typing import Tuple

import torch
import torch.nn.functional as F


class CTC(torch.nn.Module):
    """CTC module"""

    def __init__(
        self,
        odim: int,
        encoder_output_size: int,
        dropout_rate: float = 0.0,
        reduce: bool = True,
        blank_id: int = 0,
        do_focal_loss: bool = False,
        focal_alpha: float = 0.5,
        focal_gamma: float = 2,
    ):
        """ Construct CTC module
        Args:
            odim: dimension of outputs
            encoder_output_size: number of encoder projection units
            dropout_rate: dropout rate (0.0 ~ 1.0)
            reduce: reduce the CTC loss into a scalar
            blank_id: blank label.
        """
        super().__init__()
        eprojs = encoder_output_size
        self.dropout_rate = dropout_rate
        self.ctc_lo = torch.nn.Linear(eprojs, odim)

        self.do_focal_loss = do_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        self.ctc_loss2 = torch.nn.CTCLoss(reduction='sum')
        self.ctc_loss3 = torch.nn.CTCLoss(reduction='mean')

        if do_focal_loss:
            reduction_type = "none"
            self.ctc_loss = torch.nn.CTCLoss(reduction=reduction_type)
        else:
            reduction_type = "sum" if reduce else "none"
            self.ctc_loss = torch.nn.CTCLoss(blank=blank_id,
                                         reduction=reduction_type,
                                         zero_infinity=True)

    def forward(self, hs_pad: torch.Tensor, hlens: torch.Tensor,
                ys_pad: torch.Tensor,
                ys_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate CTC loss.

        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            hlens: batch of lengths of hidden state sequences (B)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)
        """
        # hs_pad: (B, L, NProj) -> ys_hat: (B, L, Nvocab)
        ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))
        # ys_hat: (B, L, D) -> (L, B, D)
        ys_hat = ys_hat.transpose(0, 1)
        ys_hat = ys_hat.log_softmax(2)

        loss = self.ctc_loss(ys_hat, ys_pad, hlens, ys_lens)
        if self.do_focal_loss:
            dbg=False
            # dbg = (torch.rand(1).item() < 0.05)
            p = torch.exp(-loss)
            if dbg:
                print(f"loss {loss}")
                print(f"log(p) {torch.log(p)}")

            loss = ((self.focal_alpha)*((1-p)**self.focal_gamma)*(loss))
            if dbg:
                print(f"f-loss {loss}")
            loss = torch.mean(loss)
            if dbg:
                loss2 = self.ctc_loss2(ys_hat, ys_pad, hlens, ys_lens)
                loss3 = self.ctc_loss3(ys_hat, ys_pad, hlens, ys_lens)
                loss2 = loss2 / ys_hat.size(1)
                print(f"final loss = {loss}, loss2 {loss2}, loss3 {loss3}")
        else:
            # Batch-size average
            loss = loss / ys_hat.size(1)
        ys_hat = ys_hat.transpose(0, 1)
        return loss, ys_hat

    def log_softmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """log_softmax of frame activations

        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: log softmax applied 3d tensor (B, Tmax, odim)
        """
        return F.log_softmax(self.ctc_lo(hs_pad), dim=2)

    def argmax(self, hs_pad: torch.Tensor) -> torch.Tensor:
        """argmax of frame activations

        Args:
            torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: argmax applied 2d tensor (B, Tmax)
        """
        return torch.argmax(self.ctc_lo(hs_pad), dim=2)
