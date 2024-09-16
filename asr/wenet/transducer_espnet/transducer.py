import torch
import torch.nn.functional as F
# from typeguard import check_argument_types

from wenet.transducer_espnet.utils import get_transducer_task_io
from wenet.transducer_espnet.abs_decoder import AbsDecoder

class Transducer(torch.nn.Module):
    """Transducer module"""
    def __init__(
        self,
        joint_network: torch.nn.Module,
        transducer_decoder: AbsDecoder,
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

        self.blank_id = 0
        self.ignore_id = ignore_id
        self.trans_type = trans_type

        # self.criterion_transducer = RNNTLoss(
        #     blank=self.blank_id,
        #     fastemit_lambda=0.0,
        # )

        print(f"trans_type: {trans_type}")
        if trans_type == "warp-transducer":
            from warprnnt_pytorch import RNNTLoss

            # self.trans_loss = RNNTLoss(blank=blank_id)
            self.criterion_transducer = RNNTLoss(
                blank=self.blank_id,
                fastemit_lambda=0.0,
            )
        elif trans_type == "warp-rnnt":
            from warp_rnnt import rnnt_loss
            self.criterion_transducer = rnnt_loss
            
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # if device.type == "cuda":
            #     try:
            #         from warp_rnnt import rnnt_loss

            #         # self.trans_loss = rnnt_loss
            #         self.criterion_transducer = rnnt_loss
            #     except ImportError:
            #         raise ImportError(
            #             "warp-rnnt is not installed. Please re-setup"
            #             " espnet or use 'warp-transducer'"
            #         )
            # else:
            #     raise ValueError("warp-rnnt is not supported in CPU mode")
        elif trans_type == "optimized_transducer":
            # print("WARNING: import optimized_transducer is disabled due to `GLIBC_2.27' not found")
            import optimized_transducer
            self.criterion_transducer = optimized_transducer.transducer_loss
        elif trans_type == "torchaudio_rnnt":
            from torchaudio.transforms import RNNTLoss

            self.criterion_transducer = RNNTLoss(
                blank=self.blank_id,
                reduction="mean",
            )

    def forward(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        labels: torch.Tensor, 
        labels_lengths: torch.Tensor
    ):
        """Compute Transducer loss.

        Args:
            encoder_out: Encoder output sequences. (B, T, D_enc)
            encoder_out_lens: Encoder output sequences lengths. (B,)
            labels: Label ID sequences. (B, L)

        Return:
            loss_transducer: Transducer loss value.

        """

        # print("forward")
        # print(encoder_out.size())
        # print(encoder_out_lens.size())
        # print(labels.size())

        decoder_in, target, t_len, u_len = get_transducer_task_io(
            labels,
            encoder_out_lens,
            ignore_id=self.ignore_id,
            blank_id=self.blank_id,
        )

        self.transducer_decoder.set_device(encoder_out.device)
        decoder_out = self.transducer_decoder(decoder_in)

        if self.trans_type == "optimized_transducer":
            joint_out = self.joint_network.forward_optimized(
                encoder_out.unsqueeze(2), t_len, decoder_out.unsqueeze(1), u_len
            )
        else:
            joint_out = self.joint_network(
                encoder_out.unsqueeze(2), decoder_out.unsqueeze(1)
            )

        # print(decoder_out.size())
        # print(joint_out.size())
        # print(target.size())

        if self.trans_type == "warp-transducer":

            loss_transducer = self.criterion_transducer(
                joint_out,
                target,
                t_len,
                u_len,
            )
        
        elif self.trans_type == "warp-rnnt":
            # log_probs = torch.log_softmax(pred_pad, dim=-1)

            # loss = self.trans_loss(
            #     log_probs,
            #     target,
            #     pred_len,
            #     target_len,
            #     reduction="mean",
            #     blank=self.blank_id,
            #     gather=True,
            # )

            dtype = joint_out.dtype
            if dtype != torch.float32:
                # warp-transducer and warp-rnnt only support float32
                joint_out = joint_out.to(dtype=torch.float32)

            joint_out = torch.log_softmax(joint_out, dim=-1)
            loss_transducer = self.criterion_transducer(
                joint_out,
                target,
                t_len,
                u_len,
                reduction="mean",
                blank=self.blank_id,
                gather=True,
            )
        elif self.trans_type == "optimized_transducer":
            # loss expects other dimensions for logits than warp_transducer
            # https://github.com/csukuangfj/optimized_transducer
            # optimized_transducer expects that the output shape of the joint network is NOT (N, T, U, V), 
            # but is (sum_all_TU, V), which is a concatenation of 2-D tensors: (T_1 * U_1, V), (T_2 * U_2, V), ..., (T_N, U_N, V).

            loss_transducer = self.criterion_transducer(
                logits=joint_out,
                targets=target,
                logit_lengths=t_len,
                target_lengths=u_len,
                blank=self.blank_id,
                reduction="mean",
                from_log_softmax=False,
                # one_sym_per_frame=True, # TODO: Add to config
            )
        elif self.trans_type == "torchaudio_rnnt":
            loss_transducer = self.criterion_transducer(
                logits=joint_out,
                targets=target,
                logit_lengths=t_len,
                target_lengths=u_len,
            )

        return loss_transducer
