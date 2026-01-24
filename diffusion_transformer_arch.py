"""
diffusion_transformer_arch.py

Architecture-only: conditional Transformer denoiser for state-action trajectories.
FIXED: Uses 'Prefix Conditioning' (concatenation) to prevent context washout.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

try:
    from diffusers import ModelMixin, ConfigMixin
    from diffusers.configuration_utils import register_to_config
    from diffusers.utils import BaseOutput
except Exception as e:
    raise ImportError(
        "diffusers is required. Install with: pip install diffusers accelerate\n"
        f"Import error: {e}"
    )


class TrajectoryEpsOutput(BaseOutput):
    """Return type with `.sample` = predicted epsilon (noise)."""
    sample: torch.Tensor


class SinusoidalPosEmb(nn.Module):
    """Embed diffusion timestep t into a fixed-size vector."""

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"SinusoidalPosEmb dim must be even, got {dim}")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t[None]
        t = t.float()
        half = self.dim // 2
        freqs = torch.exp(torch.linspace(0, math.log(10000), half, device=t.device))
        args = t[:, None] / freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TrajectoryTransformerDenoiser(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        horizon: int,
        traj_dim: int,
        cond_dim: int,
        hidden_dim: int = 256,
        depth: int = 6,
        time_emb_dim: int = 128,
        dropout: float = 0.0,
        state_dim: int | None = None,
        action_dim: int | None = None,
        n_heads: int = 8,  # Increased default heads for larger capacity
        ff_mult: int = 4,
    ):
        super().__init__()

        self.horizon = int(horizon)
        self.traj_dim = int(traj_dim)
        self.cond_dim = int(cond_dim)
        
        # Auto-adjust n_heads if hidden_dim is not divisible
        if hidden_dim % n_heads != 0:
            # Fallback: find largest divisor or adjust hidden_dim
            # Simplest safety: just round hidden_dim up
            hidden_dim = int(math.ceil(hidden_dim / n_heads) * n_heads)
        
        self.hidden_dim = int(hidden_dim)
        self.n_heads = int(n_heads)

        # 1. Trajectory Embedding (Linear Projection)
        self.traj_proj = nn.Linear(self.traj_dim, self.hidden_dim)
        
        # 2. Condition Embedding (State + Goal) -> The "Prefix Token"
        self.cond_proj = nn.Linear(self.cond_dim, self.hidden_dim)

        # 3. Diffusion Timestep Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # 4. Positional Embedding (Learnable, applied to Trajectory only)
        # We initialize this to be small to not disrupt the initial projection too much
        self.pos_emb = nn.Parameter(torch.zeros(1, self.horizon, self.hidden_dim))
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)

        # 5. Transformer Encoder (Pre-LN for stability)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.n_heads,
            dim_feedforward=int(ff_mult * self.hidden_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True, 
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(depth))

        # 6. Output Head
        self.out_proj = nn.Linear(self.hidden_dim, self.traj_dim)

    def forward(
        self,
        sample: torch.Tensor,  # (B, H, traj_dim)
        timestep: torch.Tensor,  # (B,) or scalar
        cond: torch.Tensor,  # (B, cond_dim)
        return_dict: bool = True,
    ):
        B, H, D = sample.shape
        
        # A. Embed Trajectory
        x_traj = self.traj_proj(sample)  # (B, H, hidden)
        
        # B. Add Positional Embeddings (Trajectory tokens only)
        x_traj = x_traj + self.pos_emb

        # C. Embed Condition (The Prefix Token)
        x_cond = self.cond_proj(cond).unsqueeze(1)  # (B, 1, hidden)

        # D. Embed Global Time
        if isinstance(timestep, int):
            timestep = torch.full((B,), timestep, device=sample.device, dtype=torch.long)
        elif timestep.ndim == 0:
            timestep = timestep.expand(B)
        
        time_emb = self.time_mlp(timestep).unsqueeze(1)  # (B, 1, hidden)

        # E. Inject Time into EVERYTHING (Condition + Trajectory)
        # This ensures the model knows the noise level regardless of which token it looks at.
        x_traj = x_traj + time_emb
        x_cond = x_cond + time_emb

        # F. Concatenate: [Condition_Token, Traj_Token_0, ... Traj_Token_H]
        # This is the "Prefix Conditioning" fix.
        x_seq = torch.cat([x_cond, x_traj], dim=1)  # (B, H+1, hidden)

        # G. Process with Transformer
        x_seq = self.encoder(x_seq)

        # H. Slice output (Remove the Condition token, keep only trajectory)
        x_out = x_seq[:, 1:, :]  # (B, H, hidden)

        # I. Project to Noise
        eps = self.out_proj(x_out)

        if not return_dict:
            return (eps,)
        return TrajectoryEpsOutput(sample=eps)