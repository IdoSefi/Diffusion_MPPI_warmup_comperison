"""
diffusion_cnn_arch.py
FIXED: 
1. Uses Concatenation for conditioning (Time || Cond) to prevent goal washout.
2. Auto-sets dilation to ensure full Receptive Field coverage.
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


def _choose_num_groups(channels: int, max_groups: int = 32) -> int:
    max_groups = min(int(max_groups), int(channels))
    for g in range(max_groups, 0, -1):
        if channels % g == 0:
            return g
    return 1


class _TemporalResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        cond_dim: int,  # Added: Explicit conditioning dimension
        *,
        kernel_size: int = 5,
        dilation: int = 1,
        dropout: float = 0.0,
        num_groups: Optional[int] = None,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        pad = (kernel_size // 2) * dilation
        g = _choose_num_groups(channels) if num_groups is None else int(num_groups)

        self.gn1 = nn.GroupNorm(g, channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)

        self.gn2 = nn.GroupNorm(g, channels)
        self.act2 = nn.SiLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)

        # FiLM takes 'cond_dim' as input (which is now time_emb + state_emb size)
        self.film = nn.Linear(cond_dim, 4 * channels)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        # ctx is the global condition vector
        s1, b1, s2, b2 = self.film(ctx).chunk(4, dim=1)

        h = self.gn1(x)
        h = h * (1.0 + s1.unsqueeze(-1)) + b1.unsqueeze(-1)
        h = self.conv1(self.act1(h))

        h = self.gn2(h)
        h = h * (1.0 + s2.unsqueeze(-1)) + b2.unsqueeze(-1)
        h = self.conv2(self.drop(self.act2(h)))

        return x + h


class TrajectoryCNNDenoiser(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        horizon: int,
        traj_dim: int,
        cond_dim: int,
        hidden_dim: int = 256,
        depth: int = 8,
        time_emb_dim: int = 128,
        dropout: float = 0.0,
        state_dim: int | None = None,
        action_dim: int | None = None,
        dilation_cycle: int | None = None, # Changed: Optional, defaults to depth
    ):
        super().__init__()
        kernel_size = 5

        self.horizon = int(horizon)
        self.traj_dim = int(traj_dim)
        self.cond_dim = int(cond_dim)
        
        # FIX 1: Ensure Receptive Field covers the horizon
        # If dilation_cycle is not provided, set it to 'depth' to maximize RF.
        # With depth=6, k=5: RF ~ 250 steps (plenty for 100).
        if dilation_cycle is None:
            dilation_cycle = depth
        
        self.hidden_dim = int(hidden_dim)

        # 1. Input Projection
        self.in_proj = nn.Conv1d(self.traj_dim, self.hidden_dim, kernel_size=1)

        # 2. Positional Embedding (Sequence)
        self.traj_pos_emb = nn.Embedding(self.horizon, self.hidden_dim)

        # 3. Embedding MLPs
        # We project them separately and then CONCATENATE them.
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(self.cond_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # FIX 2: Context dimension is doubled because we concat [Time, Cond]
        ctx_dim = 2 * self.hidden_dim

        # 4. Residual Blocks
        blocks = []
        for i in range(int(depth)):
            dilation = 2 ** (i % int(dilation_cycle))
            blocks.append(
                _TemporalResBlock(
                    self.hidden_dim,
                    cond_dim=ctx_dim,  # Blocks now accept the larger concatenated context
                    kernel_size=int(kernel_size),
                    dilation=int(dilation),
                    dropout=float(dropout),
                )
            )
        self.blocks = nn.ModuleList(blocks)

        # 5. Output Head
        self.out_proj = nn.Conv1d(self.hidden_dim, self.traj_dim, kernel_size=1)

    def forward(
        self,
        sample: torch.Tensor, 
        timestep: torch.Tensor, 
        cond: torch.Tensor, 
        return_dict: bool = True,
    ):
        B, H, D = sample.shape

        # 1. Inputs
        if isinstance(timestep, int):
            timestep = torch.full((B,), timestep, device=sample.device, dtype=torch.long)
        elif timestep.ndim == 0:
            timestep = timestep.expand(B)

        # 2. Embed Trajectory + Positional Emb
        x = sample.transpose(1, 2)  # (B, D, H)
        x = self.in_proj(x)         # (B, hidden, H)
        
        pos_idxs = torch.arange(H, device=sample.device, dtype=torch.long)
        pos_emb = self.traj_pos_emb(pos_idxs).unsqueeze(0).transpose(1, 2)
        x = x + pos_emb

        # 3. Embed Context (FIX: Concatenation)
        t_emb = self.time_mlp(timestep)  # (B, hidden)
        c_emb = self.cond_mlp(cond)      # (B, hidden)
        
        # Concatenate! This prevents Time from washing out Goal.
        ctx = torch.cat([t_emb, c_emb], dim=-1)  # (B, 2*hidden)

        # 4. Blocks
        for blk in self.blocks:
            x = blk(x, ctx)

        # 5. Output
        eps = self.out_proj(x).transpose(1, 2)

        if not return_dict:
            return (eps,)
        return TrajectoryEpsOutput(sample=eps)