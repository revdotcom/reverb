from typing import Optional, List

import torch
from torch import nn
from wenet.utils.class_utils import WENET_ACTIVATION_CLASSES


class TransducerJoint(torch.nn.Module):

    def __init__(self,
                 vocab_size: int,
                 enc_output_size: int,
                 pred_output_size: int,
                 join_dim: int,
                 prejoin_linear: bool = True,
                 postjoin_linear: bool = False,
                 joint_mode: str = 'add',
                 activation: str = "tanh",
                 hat_joint: bool = False,
                 dropout_rate: float = 0.1,
                 hat_activation: str = 'tanh'):
        # TODO(Mddct): concat in future
        assert joint_mode in ['add']
        super().__init__()

        self.activatoin = WENET_ACTIVATION_CLASSES[activation]()
        self.prejoin_linear = prejoin_linear
        self.postjoin_linear = postjoin_linear
        self.joint_mode = joint_mode

        if not self.prejoin_linear and not self.postjoin_linear:
            assert enc_output_size == pred_output_size == join_dim
        # torchscript compatibility
        self.enc_ffn: Optional[nn.Linear] = None
        self.pred_ffn: Optional[nn.Linear] = None
        if self.prejoin_linear:
            self.enc_ffn = nn.Linear(enc_output_size, join_dim)
            self.pred_ffn = nn.Linear(pred_output_size, join_dim)
        # torchscript compatibility
        self.post_ffn: Optional[nn.Linear] = None
        if self.postjoin_linear:
            self.post_ffn = nn.Linear(join_dim, join_dim)

        # NOTE: <blank> in vocab_size
        self.hat_joint = hat_joint
        self.vocab_size = vocab_size
        self.ffn_out: Optional[torch.nn.Linear] = None
        if not self.hat_joint:
            self.ffn_out = nn.Linear(join_dim, vocab_size)

        self.blank_pred: Optional[torch.nn.Module] = None
        self.token_pred: Optional[torch.nn.Module] = None
        if self.hat_joint:
            self.blank_pred = torch.nn.Sequential(
                torch.nn.Tanh(), torch.nn.Dropout(dropout_rate),
                torch.nn.Linear(join_dim, 1), torch.nn.LogSigmoid())
            self.token_pred = torch.nn.Sequential(
                WENET_ACTIVATION_CLASSES[hat_activation](),
                torch.nn.Dropout(dropout_rate),
                torch.nn.Linear(join_dim, self.vocab_size - 1))

        self.join_dim = join_dim

    def forward(self,
                enc_out: torch.Tensor,
                pred_out: torch.Tensor,
                pre_project: bool = True) -> torch.Tensor:
        """
        Args:
            enc_out (torch.Tensor): [B, T, E]
            pred_out (torch.Tensor): [B, T, P]
        Return:
            [B,T,U,V]
        """
        if (pre_project and self.prejoin_linear and self.enc_ffn is not None
                and self.pred_ffn is not None):
            enc_out = self.enc_ffn(enc_out)  # [B,T,E] -> [B,T,D]
            pred_out = self.pred_ffn(pred_out)
        if enc_out.ndim != 4:
            enc_out = enc_out.unsqueeze(2)  # [B,T,D] -> [B,T,1,D]
        if pred_out.ndim != 4:
            pred_out = pred_out.unsqueeze(1)  # [B,U,D] -> [B,1,U,D]

        # TODO(Mddct): concat joint
        _ = self.joint_mode
        out = enc_out + pred_out  # [B,T,U,V]

        if self.postjoin_linear and self.post_ffn is not None:
            out = self.post_ffn(out)

        if not self.hat_joint and self.ffn_out is not None:
            out = self.activatoin(out)
            out = self.ffn_out(out)
            return out
        else:
            assert self.blank_pred is not None
            assert self.token_pred is not None
            blank_logp = self.blank_pred(out)  # [B,T,U,1]

            # scale blank logp
            scale_logp = torch.clamp(1 - torch.exp(blank_logp), min=1e-6)
            label_logp = self.token_pred(out).log_softmax(
                dim=-1)  # [B,T,U,vocab-1]
            # scale token logp
            label_logp = torch.log(scale_logp) + label_logp

            out = torch.cat((blank_logp, label_logp), dim=-1)  # [B,T,U,vocab]
            return out

    # Optimized_transducer loss implementation expects other dimensions for joint_out 
    def forward_optimized(self, enc_out: torch.Tensor, pred_out: torch.Tensor, enc_out_len: torch.Tensor, pred_out_len: torch.Tensor):
        """
        Args:
            enc_out (torch.Tensor): [B, T, E]
            pred_out (torch.Tensor): [B, T, P]
        Return:
            [B,T,U,V]
        """
        
        B = enc_out.size(0)

        enc_out = enc_out.unsqueeze(2)  # [B,T,V] -> [B,T,1,V]
        pred_out = pred_out.unsqueeze(1)  # [B,U,V] -> [B,1 U, V]

        encoder_out_list = [enc_out[i, :enc_out_len[i], :, :] for i in range(B)]
        decoder_out_list = [pred_out[i, :, :pred_out_len[i]+1, :] for i in range(B)]

        # TODO(Mddct): concat joint
        _ = self.joint_mode
        # out = enc_out + pred_out  # [B,T,U,V]
        # joint_out: Optional[List[torch.Tensor]] = None
        # if (self.prejoin_linear and self.enc_ffn is not None
        #         and self.pred_ffn is not None):
            # enc_out = self.enc_ffn(enc_out)  # [B,T,E] -> [B,T,V]
            # pred_out = self.pred_ffn(pred_out)
        joint_out = [self.enc_ffn(e) + self.pred_ffn(d) for e, d in zip(encoder_out_list, decoder_out_list)]
        joint_out = [p.reshape(-1, self.join_dim) for p in joint_out]
        out = torch.cat(joint_out)

        # if self.postjoin_linear and self.post_ffn is not None:
        #     out = self.post_ffn(out)

        out = self.activatoin(out)
        out = self.ffn_out(out)
        return out

    # Optimized_transducer loss implementation expects other dimensions for joint_out 
    def forward_optimized(self, enc_out: torch.Tensor, pred_out: torch.Tensor, enc_out_len: torch.Tensor, pred_out_len: torch.Tensor):
        """
        Args:
            enc_out (torch.Tensor): [B, T, E]
            pred_out (torch.Tensor): [B, T, P]
        Return:
            [B,T,U,V]
        """
        
        B = enc_out.size(0)

        enc_out = enc_out.unsqueeze(2)  # [B,T,V] -> [B,T,1,V]
        pred_out = pred_out.unsqueeze(1)  # [B,U,V] -> [B,1 U, V]

        encoder_out_list = [enc_out[i, :enc_out_len[i], :, :] for i in range(B)]
        decoder_out_list = [pred_out[i, :, :pred_out_len[i]+1, :] for i in range(B)]

        # TODO(Mddct): concat joint
        _ = self.joint_mode
        # out = enc_out + pred_out  # [B,T,U,V]
        # joint_out: Optional[List[torch.Tensor]] = None
        # if (self.prejoin_linear and self.enc_ffn is not None
        #         and self.pred_ffn is not None):
            # enc_out = self.enc_ffn(enc_out)  # [B,T,E] -> [B,T,V]
            # pred_out = self.pred_ffn(pred_out)
        joint_out = [self.enc_ffn(e) + self.pred_ffn(d) for e, d in zip(encoder_out_list, decoder_out_list)]
        joint_out = [p.reshape(-1, self.join_dim) for p in joint_out]
        out = torch.cat(joint_out)

        # if self.postjoin_linear and self.post_ffn is not None:
        #     out = self.post_ffn(out)

        out = self.activatoin(out)
        out = self.ffn_out(out)
        return out