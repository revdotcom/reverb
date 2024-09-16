"""Utility functions for Transducer models."""

import torch

# from wenet.transformer.transducer.nets_utils import pad_list

def pad_list(xs, pad_value):
    """Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])

    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]

    return pad

def get_transducer_task_io(
    labels: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    ignore_id: int = -1,
    blank_id: int = 0,
):
    """Get Transducer loss I/O.

    Args:
        labels: Label ID sequences. (B, L)
        encoder_out_lens: Encoder output lengths. (B,)
        ignore_id: Padding symbol ID.
        blank_id: Blank symbol ID.

    Return:
        decoder_in: Decoder inputs. (B, U)
        target: Target label ID sequences. (B, U)
        t_len: Time lengths. (B,)
        u_len: Label lengths. (B,)

    """
    device = labels.device

    labels_unpad = [y[y != ignore_id] for y in labels]
    blank = labels[0].new([blank_id])

    decoder_in = pad_list(
        [torch.cat([blank, label], dim=0) for label in labels_unpad], blank_id
    ).to(device)

    target = pad_list(labels_unpad, blank_id).type(torch.int32).to(device)

    if encoder_out_lens.dim() > 1:
        enc_mask = [m[m != 0] for m in encoder_out_lens]
        encoder_out_lens = list(map(int, [m.size(0) for m in enc_mask]))
    else:
        encoder_out_lens = list(map(int, encoder_out_lens))

    t_len = torch.IntTensor(encoder_out_lens).to(device)
    u_len = torch.IntTensor([y.size(0) for y in labels_unpad]).to(device)

    return decoder_in, target, t_len, u_len
