# diffusion_sampling_simple.py
from __future__ import annotations

from typing import Optional, Union
import numpy as np
import torch

from config import HORIZON, DEVICE  # uses your project config :contentReference[oaicite:1]{index=1}


def _get_action_dim(model) -> int:
    # Prefer diffusers ConfigMixin
    if hasattr(model, "config") and hasattr(model.config, "action_dim"):
        return int(model.config.action_dim)
    # Fallback: plain attribute
    if hasattr(model, "action_dim"):
        return int(model.action_dim)
    raise ValueError(
        "Cannot determine action_dim. Ensure your model has `model.config.action_dim` "
        "or `model.action_dim`."
    )


def _get_eps_hat(model_out) -> torch.Tensor:
    # diffusers BaseOutput style
    if hasattr(model_out, "sample"):
        return model_out.sample
    # raw tensor
    if isinstance(model_out, torch.Tensor):
        return model_out
    # tuple/list
    if isinstance(model_out, (tuple, list)) and len(model_out) > 0 and isinstance(model_out[0], torch.Tensor):
        return model_out[0]
    raise TypeError(f"Unrecognized model output type: {type(model_out)}")


@torch.no_grad()
def sample_action_trajectory(
    *,
    model,
    scheduler,
    cond: Union[np.ndarray, torch.Tensor],     # (C,) or (B,C)
    num_inference_steps: int = 25,
    action_low: float = -1.0,
    action_high: float = 1.0,
    return_numpy: bool = True,
    generator: Optional[torch.Generator] = None,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Sample ONE action trajectory using diffusers scheduler, for your fixed HORIZON.

    Inputs:
      cond: (C,) single state OR (B,C) batch of states

    Output:
      if cond is (C,):    (HORIZON, action_dim)
      if cond is (B,C):   (B, HORIZON, action_dim)
    """
    if device is None:
        device = torch.device(DEVICE)
    else:
        device = torch.device(device)

    # cond -> torch (B,C)
    if isinstance(cond, torch.Tensor):
        cond_t = cond.to(device=device, dtype=torch.float32)
    else:
        cond_t = torch.as_tensor(cond, device=device, dtype=torch.float32)

    if cond_t.ndim == 1:
        cond_t = cond_t.unsqueeze(0)  # (1,C)
    if cond_t.ndim != 2:
        raise ValueError(f"cond must be (C,) or (B,C), got {tuple(cond_t.shape)}")

    B = cond_t.shape[0]
    action_dim = _get_action_dim(model)

    # initial noise
    x = torch.randn((B, HORIZON, action_dim), device=device, generator=generator)

    # denoise
    scheduler.set_timesteps(num_inference_steps, device=device)
    model.eval()

    for t in scheduler.timesteps:
        t_batch = t.expand(B).long()
        out = model(sample=x, timestep=t_batch, cond=cond_t, return_dict=True)
        eps_hat = _get_eps_hat(out)

        step_out = scheduler.step(eps_hat, t, x, generator=generator)
        x = step_out.prev_sample

    x = x.clamp(action_low, action_high)

    if B == 1:
        x = x.squeeze(0)  # (H, A)

    if return_numpy:
        return x.detach().cpu().numpy()
    return x
