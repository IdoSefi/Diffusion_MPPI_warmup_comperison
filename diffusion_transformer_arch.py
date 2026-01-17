"""
diffusion_transformer_arch.py

Architecture-only: conditional Transformer denoiser for state-action trajectories.

This treats each planning timestep as a token and predicts epsilon (noise) for all
timesteps *non-autoregressively*. Conditioning (state/goal vector) and the
diffusion timestep embedding are injected as a global context added to every
token embedding.

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
        if t.ndim == 0:
            t = t[None]
        t = t.float()

        half = self.dim // 2
        freqs = torch.exp(torch.linspace(0, math.log(10000), half, device=t.device))
        args = t[:, None] / freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


def _choose_nheads(d_model: int, preferred: int = 8) -> int:
    """Pick a valid number of attention heads that divides d_model."""
    for h in [preferred, 16, 8, 4, 2, 1]:
        if d_model % h == 0:
            return h
    return 1


class TrajectoryTransformerDenoiser(ModelMixin, ConfigMixin):
    """
    Conditional Transformer epsilon-model for trajectories.

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
        depth: int = 6,
        time_emb_dim: int = 128,
        dropout: float = 0.1,
        state_dim: int | None = None,
        action_dim: int | None = None,
        n_heads: Optional[int] = None,
        ff_mult: int = 4,
    ):
        super().__init__()

        self.horizon = int(horizon)
        self.traj_dim = int(traj_dim)
        self.cond_dim = int(cond_dim)
        self.state_dim = int(state_dim) if state_dim is not None else None
        self.action_dim = int(action_dim) if action_dim is not None else None

        hidden_dim = int(hidden_dim)
        self.hidden_dim = hidden_dim
        self.n_heads = int(_choose_nheads(hidden_dim) if n_heads is None else n_heads)
        if hidden_dim % self.n_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by n_heads={self.n_heads}")

        self.in_proj = nn.Linear(self.traj_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, self.traj_dim)

        # Learned positional embedding along the planning horizon.
        self.pos_emb = nn.Parameter(torch.randn(self.horizon, hidden_dim) * 0.02)

        # Global context (diffusion time + cond) added to every token.
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(self.cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=self.n_heads,
            dim_feedforward=int(ff_mult) * hidden_dim,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(depth))

        self.drop = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()

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

        x = self.in_proj(sample)  # (B, H, E)
        x = x + self.pos_emb.unsqueeze(0)  # (B, H, E)

        ctx = self.time_mlp(timestep) + self.cond_mlp(cond)  # (B, E)
        x = x + ctx.unsqueeze(1)
        x = self.drop(x)

        x = self.encoder(x)  # (B, H, E)
        eps = self.out_proj(x)  # (B, H, D)

        if not return_dict:
            return (eps,)
        return TrajectoryEpsOutput(sample=eps)


if __name__ == "__main__":
    # Quick sanity check for shapes (not a training example).
    B, H, state_dim, action_dim, C = 4, 30, 6, 2, 6
    traj_dim = state_dim + action_dim
    model = TrajectoryTransformerDenoiser(
        horizon=H,
        traj_dim=traj_dim,
        cond_dim=C,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        depth=6,
    )
    x_t = torch.randn(B, H, traj_dim)
    t = torch.randint(0, 100, (B,))
    cond = torch.randn(B, C)
    out = model(sample=x_t, timestep=t, cond=cond)
    print(out.sample.shape)  # (B, H, D)
