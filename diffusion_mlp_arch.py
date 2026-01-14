"""
diffusion_mlp_arch.py

Architecture-only: conditional MLP denoiser for action trajectories, meant to be used with
Hugging Face diffusers schedulers (DDPMScheduler / DDIMScheduler / etc.).

Your training script (later) will handle diffusion steps via the scheduler:
  - add_noise(x0, noise, t)   -> x_t
  - model(x_t, t, cond)       -> eps_hat
  - loss = MSE(eps_hat, noise)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

# diffusers mixins make the module "saveable" like other diffusers models.
# BaseOutput gives the standard `.sample` field used across diffusers.
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


class TrajectoryMLPDenoiser(ModelMixin, ConfigMixin):
    """
    Conditional MLP epsilon-model for trajectories.

    Inputs:
      sample:   x_t noisy action window      (B, H, A)
      timestep: diffusion step index t       (B,) or scalar
      cond:     conditioning state/features  (B, C)

    Output:
      eps_hat: predicted noise               (B, H, A)  (returned as `.sample`)
    """

    @register_to_config
    def __init__(
        self,
        horizon: int,
        action_dim: int,
        cond_dim: int,
        hidden_dim: int = 1024,
        depth: int = 4,
        time_emb_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.cond_dim = int(cond_dim)

        # Map timestep -> embedding. (This embedding is concatenated to inputs.)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )

        # We flatten the whole action window, so an MLP can process it.
        # Input vector = flattened trajectory + timestep embedding + conditioning vector.
        in_dim = self.horizon * self.action_dim + time_emb_dim + self.cond_dim

        layers = []
        d = in_dim
        for _ in range(int(depth)):
            layers += [nn.Linear(d, hidden_dim), nn.SiLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim

        # Output is flattened epsilon for the entire trajectory; we reshape back to (B,H,A).
        layers.append(nn.Linear(d, self.horizon * self.action_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        sample: torch.Tensor,     # x_t: (B, H, A)
        timestep: torch.Tensor,   # t:   (B,) or scalar/int
        cond: torch.Tensor,       # (B, C)
        return_dict: bool = True,
    ):
        # Basic shape checks: catch silent bugs early.
        if sample.ndim != 3:
            raise ValueError(f"sample must be (B,H,A), got {tuple(sample.shape)}")
        B, H, A = sample.shape
        if (H != self.horizon) or (A != self.action_dim):
            raise ValueError(
                f"sample shape mismatch: expected (B,{self.horizon},{self.action_dim}), got {tuple(sample.shape)}"
            )
        if cond.ndim != 2 or cond.shape != (B, self.cond_dim):
            raise ValueError(f"cond must be (B,{self.cond_dim}), got {tuple(cond.shape)}")

        # Normalize timestep to (B,) tensor (diffusers schedulers usually give you this already).
        if isinstance(timestep, int):
            timestep = torch.full((B,), timestep, device=sample.device, dtype=torch.long)
        elif timestep.ndim == 0:
            timestep = timestep.expand(B)

        # Flatten trajectory so the MLP can process all timesteps jointly.
        x_flat = sample.reshape(B, -1)                 # (B, H*A)
        t_emb = self.time_mlp(timestep)                # (B, time_emb_dim)
        h = torch.cat([x_flat, t_emb, cond], dim=-1)   # (B, H*A + time_emb + C)

        eps_flat = self.net(h)                         # (B, H*A)
        eps = eps_flat.reshape(B, self.horizon, self.action_dim)

        # diffusers convention: return object with `.sample`
        if not return_dict:
            return (eps,)
        return TrajectoryEpsOutput(sample=eps)


if __name__ == "__main__":
    # Quick sanity check for shapes (not a training example).
    B, H, A, C = 4, 30, 2, 6
    model = TrajectoryMLPDenoiser(horizon=H, action_dim=A, cond_dim=C)
    x_t = torch.randn(B, H, A)
    t = torch.randint(0, 100, (B,))
    cond = torch.randn(B, C)
    out = model(sample=x_t, timestep=t, cond=cond)
    print(out.sample.shape)  # (B, H, A)
