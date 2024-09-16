"""Transducer joint network implementation."""

import torch

from wenet.utils.common import get_activation


class JointNetwork(torch.nn.Module):
    """Transducer joint network module.

    Args:
        joint_output_size: Joint network output dimension
        encoder_output_size: Encoder output dimension.
        decoder_output_size: Decoder output dimension.
        joint_space_size: Dimension of joint space.
        joint_activation_type: Type of activation for joint network.

    """

    def __init__(
        self,
        joint_output_size: int,
        encoder_output_size: int,
        decoder_output_size: int,
        joint_space_size: int = 256,
        joint_activation_type: str = "tanh",
    ):
        """Joint network initializer."""
        super().__init__()

        self.lin_enc = torch.nn.Linear(encoder_output_size, joint_space_size)
        self.lin_dec = torch.nn.Linear(decoder_output_size, joint_space_size)

        self.lin_out = torch.nn.Linear(joint_space_size, joint_output_size)

        self.joint_activation = get_activation(joint_activation_type)

        self.joint_space_size = joint_space_size

    def forward(
        self,
        enc_out: torch.Tensor,
        dec_out: torch.Tensor,
    ) -> torch.Tensor:
        """Joint computation of encoder and decoder hidden state sequences.

        Args:
            enc_out: Expanded encoder output state sequences (B, T, 1, D_enc)
            dec_out: Expanded decoder output state sequences (B, 1, U, D_dec)

        Returns:
            joint_out: Joint output state sequences. (B, T, U, D_out)

        """
        joint_out = self.joint_activation(self.lin_enc(enc_out) + self.lin_dec(dec_out))

        return self.lin_out(joint_out)

    # Optimized_transducer loss implementation expects other dimensions for joint_out 
    def forward_optimized(
        self,
        enc_out: torch.Tensor,
        logit_lengths: torch.Tensor,
        dec_out: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Joint computation of encoder and decoder hidden state sequences.

        Args:
            enc_out: Expanded encoder output state sequences (B, T, 1, D_enc)
            dec_out: Expanded decoder output state sequences (B, 1, U, D_dec)

        Returns:
            joint_out: Joint output state sequences. (B, T, U, D_out)

        """

        B = enc_out.size(0)
        encoder_out_list = [enc_out[i, :logit_lengths[i], :, :] for i in range(B)]
        decoder_out_list = [dec_out[i, :, :target_lengths[i]+1, :] for i in range(B)]

        joint_out = [self.joint_activation(self.lin_enc(e) + self.lin_dec(d)) for e, d in zip(encoder_out_list, decoder_out_list)]
        joint_out = [p.reshape(-1, self.joint_space_size) for p in joint_out]
        joint_out = torch.cat(joint_out)

        return self.lin_out(joint_out)
