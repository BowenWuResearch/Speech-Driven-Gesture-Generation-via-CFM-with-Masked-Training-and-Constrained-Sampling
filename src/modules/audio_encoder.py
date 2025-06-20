import numpy as np
import torch as th
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from loguru import logger
from modules.audio_feature_extraction import N_AUDIO_FEATS
from modules.tensor_utils import interpnd, create_padding_mask


class ZeroEggsSpeechEncoder(nn.Module):
    """
    ubisoft zeroEggs audio feature encoder
    when fps=60 in the original paper, kernel_size=31,
    adapt for fps=10 to set kernel_size=5
    """
    def __init__(self, hidden_size, output_size, kernel_size=15, dropout=0.2):
        super(ZeroEggsSpeechEncoder, self).__init__()

        self.layer0 = nn.Conv1d(
            N_AUDIO_FEATS, hidden_size, kernel_size=1, padding="same", padding_mode="replicate"
        )
        self.drop0 = nn.Dropout(p=dropout)

        self.layer1 = nn.Conv1d(
            hidden_size, output_size, kernel_size=kernel_size, padding="same", padding_mode="replicate"
        )
        self.drop1 = nn.Dropout(p=dropout)

        self.layer2 = nn.Conv1d(output_size, output_size, 1)

    def forward(self, x):
        # x: (B,C,T)
        x = self.drop0(F.elu(self.layer0(x)))
        x = self.drop1(F.elu(self.layer1(x)))
        x = F.elu(self.layer2(x))
        return x


class AudioEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        out_dim: int = None,
        emb_dim: int = None,
        dropout: float = 0.
    ):
        super().__init__()
        if out_dim is None:
            out_dim = hidden_dim
        if emb_dim is None:
            emb_dim = out_dim
        self.base_feat_encoder = ZeroEggsSpeechEncoder(
            N_BASE_FEATURE, hidden_dim, emb_dim, dropout
        )
        if emb_dim != out_dim:
            self.out_proj = nn.Linear(emb_dim, out_dim)
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(self, audio: Tensor, audio_size: Tensor, audio_feats: Tensor):
        """
        Args:
            audio_feats (torch.Tensor): shape (B, S, D)

        Returns:
            h (torch.Tensor): shape (B, S, D)
            padding_mask (torch.Tensor): shape (B, S)
            m (torch.Tensor): shape (B, D)
        """
        # Base features
        audio_feats = rearrange(audio_feats, 'b t d -> b d t')
        h = self.base_feat_encoder(audio_feats) # (B, D, S)
        h = rearrange(h, 'b d t -> b t d')
        
        if hasattr(self, 'out_proj'):
            h = self.out_proj(h)
        
        h = self.out_norm(h)

        # Mask
        padding_mask = create_padding_mask(audio_size) # (B, T)
        padding_mask = interpnd(padding_mask, h.size(1), mode='nearest')

        # Process mean feature
        m = h.mean(dim=1)

        return h, padding_mask, m
