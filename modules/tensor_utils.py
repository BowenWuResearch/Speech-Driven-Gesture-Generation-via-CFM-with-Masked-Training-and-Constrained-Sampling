import torch as th
from torch import Tensor


def normalize(x: Tensor, mu: Tensor, std: Tensor) -> Tensor:
    dims = len(x.shape)
    while len(mu.shape) < dims:
        mu = mu.unsqueeze(0)
        std = std.unsqueeze(0)
    return (x - mu) / std


def denormalize(x: Tensor, mu: Tensor, std: Tensor) -> Tensor:
    dims = len(x.shape)
    while len(mu.shape) < dims:
        mu = mu.unsqueeze(0)
        std = std.unsqueeze(0)
    return x * std + mu


def interpnd(x: Tensor, *args, **kwargs) -> Tensor:
    assert x.dim() >= 2 and x.dim() <= 5, 'x must be 2D ~ 5D tensor'
    dtype = x.dtype
    x = x.float()
    if x.dim() == 2:
        # (B, T)
        x = x.unsqueeze(1) # (B, 1, T)
        x = th.nn.functional.interpolate(x, *args, **kwargs) # (B, 1, size)
        x = x.squeeze(1) # (B, size)
    else:
        # (B, ..., T)
        x = th.nn.functional.interpolate(x, *args, **kwargs)
    x = x.to(dtype)
    return x


def create_padding_mask(lengths: Tensor, max_length: int = None) -> Tensor:
    # lengths: (B,) of long type
    # True for padded index, False for non-padded index
    if max_length is None:
        max_length = th.max(lengths)
    idxs = th.arange(max_length, device=lengths.device) # (max_len,)
    mask = idxs[None, :] >= lengths[:, None] # (B, max_len)
    return mask