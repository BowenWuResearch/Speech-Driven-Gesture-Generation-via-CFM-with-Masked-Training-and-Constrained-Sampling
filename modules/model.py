import torch as th
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from .nn import *
from .dataset import WINDOW_SIZE


class LDA(nn.Module):
    def __init__(
        self,
        pose_dim,
        residual_layers,
        residual_channels,
        embedding_dim,
        l_cond_dim,
        nn_name='tisa',
        num_blocks=2,
        num_heads=8,
        activation='relu',
        dropout=0.1,
        norm='LN',
        d_ff=1024,
        use_v2=True,
        use_preln=False,
        bias=False,
        dilation_cycle=[0, 1, 2]
    ):
        super().__init__()
        self.input_projection = Conv1dLayer(pose_dim, residual_channels, 1)
        
        self.time_embeddings = SinusoidalPosEmb(residual_channels)
        self.time_mlp = TimestepEmbedding(
            in_channels=residual_channels,
            time_embed_dim=embedding_dim,
            act_fn="silu",
        )
        
        nn_args = dict(
            num_blocks=num_blocks,
            num_heads=num_heads,
            activation=activation,
            dropout=dropout,
            norm=norm,
            d_ff=d_ff,
            seq_len=WINDOW_SIZE,
            use_v2=use_v2,
            use_preln=use_preln,
            bias=bias,
            dilation_cycle=dilation_cycle,
        )
        self.residual_layers = nn.ModuleList([
            ResidualBlock(residual_channels,
                embedding_dim,
                2*l_cond_dim,
                nn_name,
                nn_args,
                i)
            for i in range(residual_layers)
        ])
        self.skip_projection = Conv1dLayer(residual_channels, residual_channels, 1)
        self.output_projection = Conv1dLayer(residual_channels, pose_dim, 1)
        nn.init.zeros_(self.output_projection.conv1d.weight)
        self.l_cond_dim = l_cond_dim
        self.hum_emb = nn.Embedding(25, l_cond_dim)

    def forward(self, x, local_cond, hum_id, t):
        x = self.input_projection(x)
        x = F.relu(x)
        
        t = self.time_embeddings(t)
        t = self.time_mlp(t)
        global_cond = self.hum_emb(hum_id) # (B, Dc)
        global_cond = global_cond.unsqueeze(1).expand(-1, x.size(1), -1) # (B, T, Dc)
        local_cond = torch.cat((local_cond, global_cond), dim=2) # (B, T, 2*Dc)

        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x, t, local_cond)
            skip = skip_connection if skip is None else skip_connection + skip
            
        if skip is not None:
            x = skip / sqrt(len(self.residual_layers))
            
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x
