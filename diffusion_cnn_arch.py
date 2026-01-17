"""
diffusion_cnn_arch.py

Architecture-only: conditional CNN denoiser for state-action trajectories.

This is a 1D *temporal* ConvNet (Conv1d over the horizon dimension) with residual
blocks. It follows the "temporal locality" intuition from Diffuser-style models:
each denoising step enforces local consistency, and multiple denoising steps
compose into globally coherent plans.

Contract (same as diffusion_mlp_arch.py):
  forward(sample=x_t, timestep=t, cond=state, return_dict=True) -> output with
  `.sample` being the predicted epsilon (noise) of shape (B, H, traj_dim).
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
    """Return type with `.sample` = predicted epsilon (noise), diffusers-style."""
    sample: torch.Tensor


class SinusoidalPosEmb(nn.Module):
    """Embed diffusion timestep t into a fixed-size vector (sin/cos, standard DDPM trick)."""

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"SinusoidalPosEmb dim must be even, got {dim}")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Accept scalar t or batched (B,). Always produce (B, dim).
        if t.ndim == 0:
            t = t[None]
        t = t.float()

        half = self.dim // 2
        freqs = torch.exp(torch.linspace(0, math.log(10000), half, device=t.device))
        args = t[:, None] / freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


def _choose_num_groups(channels: int, max_groups: int = 32) -> int:
    """Pick a GroupNorm `num_groups` that divides `channels` (stable across hidden sizes)."""
    max_groups = min(int(max_groups), int(channels))
    for g in range(max_groups, 0, -1):
        if channels % g == 0:
            return g
    return 1


class _TemporalResBlock(nn.Module):
    """A simple residual block: GN -> SiLU -> Conv1d -> (+context) -> GN -> SiLU -> Conv1d."""

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.0,
        num_groups: Optional[int] = None,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve length with symmetric padding")

        pad = (kernel_size // 2) * dilation
        g = _choose_num_groups(channels) if num_groups is None else int(num_groups)

        self.gn1 = nn.GroupNorm(g, channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)

        self.gn2 = nn.GroupNorm(g, channels)
        self.act2 = nn.SiLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """
        x:   (B, C, H)
        ctx: (B, C)  broadcast and added after first conv
        """
        h = self.conv1(self.act1(self.gn1(x)))
        h = h + ctx.unsqueeze(-1)
        h = self.conv2(self.drop(self.act2(self.gn2(h))))
        return x + h


class TrajectoryCNNDenoiser(ModelMixin, ConfigMixin):
    """
    Conditional temporal CNN epsilon-model for trajectories.

    Inputs:
      sample:   x_t noisy traj window        (B, H, D)
      timestep: diffusion step index t       (B,) or scalar/int
      cond:     conditioning state/features  (B, C)

    Output:
      eps_hat: predicted noise               (B, H, D)  (returned as `.sample`)
    """

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
        kernel_size: int = 3,
        dilation_cycle: int = 4,
    ):
        super().__init__()

        self.horizon = int(horizon)
        self.traj_dim = int(traj_dim)
        self.cond_dim = int(cond_dim)
        self.state_dim = int(state_dim) if state_dim is not None else None
        self.action_dim = int(action_dim) if action_dim is not None else None

        self.in_proj = nn.Conv1d(self.traj_dim, int(hidden_dim), kernel_size=1)
        self.out_proj = nn.Conv1d(int(hidden_dim), self.traj_dim, kernel_size=1)

        # Global conditioning -> per-channel bias for each residual block.
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(self.cond_dim, int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
        )

        blocks = []
        for i in range(int(depth)):
            dilation = 2 ** (i % int(dilation_cycle))
            blocks.append(
                _TemporalResBlock(
                    int(hidden_dim),
                    kernel_size=int(kernel_size),
                    dilation=int(dilation),
                    dropout=float(dropout),
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(
        self,
        sample: torch.Tensor,  # (B, H, D)
        timestep: torch.Tensor,  # (B,) or scalar/int
        cond: torch.Tensor,  # (B, C)
        return_dict: bool = True,
    ):
        if sample.ndim != 3:
            raise ValueError(f"sample must be (B,H,D), got {tuple(sample.shape)}")
        B, H, D = sample.shape
        if (H != self.horizon) or (D != self.traj_dim):
            raise ValueError(
                f"sample shape mismatch: expected (B,{self.horizon},{self.traj_dim}), got {tuple(sample.shape)}"
            )
        if cond.ndim != 2 or cond.shape != (B, self.cond_dim):
            raise ValueError(f"cond must be (B,{self.cond_dim}), got {tuple(cond.shape)}")

        if isinstance(timestep, int):
            timestep = torch.full((B,), timestep, device=sample.device, dtype=torch.long)
        elif timestep.ndim == 0:
            timestep = timestep.expand(B)

        # (B, H, D) -> (B, D, H)
        x = sample.transpose(1, 2)
        x = self.in_proj(x)  # (B, hidden, H)

        ctx = self.time_mlp(timestep) + self.cond_mlp(cond)  # (B, hidden)
        for blk in self.blocks:
            x = blk(x, ctx)

        eps = self.out_proj(x).transpose(1, 2)  # (B, H, D)

        if not return_dict:
            return (eps,)
        return TrajectoryEpsOutput(sample=eps)


if __name__ == "__main__":
    # Quick sanity check for shapes (not a training example).
    B, H, state_dim, action_dim, C = 4, 30, 6, 2, 6
    traj_dim = state_dim + action_dim
    model = TrajectoryCNNDenoiser(
        horizon=H,
        traj_dim=traj_dim,
        cond_dim=C,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        depth=8,
    )
    x_t = torch.randn(B, H, traj_dim)
    t = torch.randint(0, 100, (B,))
    cond = torch.randn(B, C)
    out = model(sample=x_t, timestep=t, cond=cond)
    print(out.sample.shape)  # (B, H, D)
