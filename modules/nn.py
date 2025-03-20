# Copyright 2023 Motorica AB, Inc. All Rights Reserved.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer.tisa_transformer import TisaTransformer
from diffusers.models.activations import get_activation
from math import sqrt
from typing import Optional

class Conv1dLayer(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, padding=0, dilation=1):
    super().__init__()
    self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
    nn.init.kaiming_normal_(self.conv1d.weight)

  def forward(self, x):
    return self.conv1d(x.permute(0,2,1)).permute(0,2,1)

def silu(x):
  return x * torch.sigmoid(x)
    
class ResidualBlock(nn.Module):
  def __init__(self, residual_channels, 
            embedding_dim, 
            l_cond_dim,
            nn_name,
            nn_args,
            index):

    super().__init__()
    if nn_name=="tisa":
        dilation_cycle = nn_args["dilation_cycle"]
        dilation=dilation_cycle[(index % len(dilation_cycle))]
        self.nn = TisaTransformer(residual_channels, 2 * residual_channels, d_model=residual_channels, num_blocks=nn_args["num_blocks"], num_heads=nn_args["num_heads"], activation=nn_args["activation"], norm=nn_args["norm"], drop_prob=nn_args["dropout"], d_ff=nn_args["d_ff"], seqlen=nn_args["seq_len"], use_preln=nn_args["use_preln"], bias=nn_args["bias"], dilation=dilation)
    elif nn_name=="conv":
        dilation=2**(index % nn_args["dilation_cycle_length"])
        self.nn = Conv1dLayer(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
    else:
        raise ValueError(f"Unknown nn_name: {nn_name}")
        
    self.l_cond_dim = l_cond_dim
    
    self.diffusion_projection = nn.Linear(embedding_dim, residual_channels)
    self.local_cond_projection = nn.Linear(l_cond_dim, residual_channels)
    self.output_projection = Conv1dLayer(residual_channels, 2 * residual_channels, 1)
    self.residual_channels = residual_channels

  def forward(self, x, diffusion_step, local_cond):
    diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(1)
    y = x + diffusion_step

    if self.l_cond_dim > 0:
        y += self.local_cond_projection(local_cond)
    y = self.nn(y).squeeze(-1)

    gate, filter = torch.chunk(y, 2, dim=2)
    y = torch.sigmoid(gate) * torch.tanh(filter)

    y = self.output_projection(y)
    residual, skip = torch.chunk(y, 2, dim=2)
    return (x + residual) / sqrt(2.0), skip

class SinusoidalPosEmb(torch.nn.Module):
  def __init__(self, dim):
    super().__init__()
    self.dim = dim
    assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

  def forward(self, x, scale=1000):
    if x.ndim < 1:
      x = x.unsqueeze(0)
    device = x.device
    half_dim = self.dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
    emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
    return emb

class TimestepEmbedding(nn.Module):
  def __init__(
    self,
    in_channels: int,
    time_embed_dim: int,
    act_fn: str = "silu",
    out_dim: int = None,
    post_act_fn: Optional[str] = None,
    cond_proj_dim=None,
):
    super().__init__()

    self.linear_1 = nn.Linear(in_channels, time_embed_dim)

    if cond_proj_dim is not None:
      self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
    else:
      self.cond_proj = None

    self.act = get_activation(act_fn)

    if out_dim is not None:
      time_embed_dim_out = out_dim
    else:
      time_embed_dim_out = time_embed_dim
    self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out)

    if post_act_fn is None:
      self.post_act = None
    else:
      self.post_act = get_activation(post_act_fn)

  def forward(self, sample, condition=None):
    if condition is not None:
      sample = sample + self.cond_proj(condition)
    sample = self.linear_1(sample)

    if self.act is not None:
      sample = self.act(sample)

    sample = self.linear_2(sample)

    if self.post_act is not None:
      sample = self.post_act(sample)
    return sample
