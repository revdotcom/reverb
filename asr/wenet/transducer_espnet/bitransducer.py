import torch
import torch.nn.functional as F
# from typeguard import check_argument_types

# from wenet.transformer.transducer.utils import get_transducer_task_io
from wenet.transducer_espnet.abs_decoder import AbsDecoder
from wenet.transducer_espnet.transducer import Transducer

from wenet.utils.common import reverse_pad_list
from torch.nn.utils.rnn import pad_sequence

class BiTransducer(torch.nn.Module):
    """Bidirectional Transducer module"""
    def __init__(
        self,
        joint_network: torch.nn.Module,
        transducer_decoder: AbsDecoder,
        joint_network_r: torch.nn.Module,
        transducer_decoder_r: AbsDecoder,
        ignore_id: int,
        trans_type: str
    ):
        """ Construct CTC module
        Args:
            odim: dimension of outputs
            encoder_output_size: number of encoder projection units
            dropout_rate: dropout rate (0.0 ~ 1.0)
            reduce: reduce the CTC loss into a scalar
        """
        # assert check_argument_types()
        super().__init__()

        # from warprnnt_pytorch import RNNTLoss

        self.transducer_decoder = transducer_decoder
        self.joint_network = joint_network
        self.transducer_decoder_r = transducer_decoder_r
        self.joint_network_r = joint_network_r

        self.blank_id = 0
        self.ignore_id = ignore_id
        self.trans_type = trans_type

        self.transducer = Transducer(joint_network, transducer_decoder, ignore_id, trans_type)
        self.transducer_r = Transducer(joint_network_r, transducer_decoder_r, ignore_id, trans_type)

    def reverse_features_pad_list(self, x_pad: torch.Tensor,
                     x_lengths: torch.Tensor,
                     pad_value: float = -1.0) -> torch.Tensor:
        """Reverse padding for the list of tensors.

        Args:
            ys_pad (tensor): The padded tensor (B, Tokenmax).
            ys_lens (tensor): The lens of token seqs (B)
            pad_value (int): Value for padding.

        Returns:
            Tensor: Padded tensor (B, Tokenmax).

        Examples:
            >>> x
            tensor([[1, 2, 3, 4], [5, 6, 7, 0], [8, 9, 0, 0]])
            >>> pad_list(x, 0)
            tensor([[4, 3, 2, 1],
                    [7, 6, 5, 0],
                    [9, 8, 0, 0]])

        """

        x_pad_reverse = pad_sequence([(torch.flip(x[:i], [0]))
                                for x, i in zip(x_pad, x_lengths)], True,
                                pad_value)
        return x_pad_reverse
    
    def forward(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        labels: torch.Tensor, 
        labels_lengths: torch.Tensor,
    ):
        """Compute Transducer loss.

        Args:
            encoder_out: Encoder output sequences. (B, T, D_enc)
            encoder_out_lens: Encoder output sequences lengths. (B,)
            labels: Label ID sequences. (B, L)

        Return:
            loss_transducer: Transducer loss value.

        """

        encoder_out_r = self.reverse_features_pad_list(encoder_out, encoder_out_lens, 0.0)
        labels_r = reverse_pad_list(labels, labels_lengths, float(self.ignore_id))
        # print(labels.size())  
        # print(labels[1])
        # print(labels_r[1])
        # print(encoder_out.size())
        # print(encoder_out_lens)
        # print(encoder_out[0,394,:5])
        # print(encoder_out[2,394,:5])
        # print(encoder_out_r.size())

        loss_transducer_l = self.transducer(encoder_out, encoder_out_lens, labels, labels_lengths)
        # reverse encoder and lables?
        loss_transducer_r = self.transducer_r(encoder_out_r, encoder_out_lens, labels_r, labels_lengths)
        loss_transducer = 0.7*loss_transducer_l + 0.3*loss_transducer_r

        return loss_transducer
