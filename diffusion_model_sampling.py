# diffusion_model_sampling.py
"""Simple diffusion sampling for (state, action) trajectories."""
from __future__ import annotations
from typing import Optional, Union
import numpy as np
import torch
from config import HORIZON, DEVICE


def _get_dim(model, name):
    """Get dimension from model config or attribute."""
    if hasattr(model, "config") and hasattr(model.config, name):
        return int(getattr(model.config, name))
    if hasattr(model, name):
        return int(getattr(model, name))
    raise ValueError(f"Cannot determine {name} from model")


@torch.no_grad()
def sample_state_action_trajectory(
    model,
    scheduler,
    cond: Union[np.ndarray, torch.Tensor],  # (C,) or (B,C) - should be NORMALIZED if model trained on normalized
    num_inference_steps: int = 25,
    state_mean: np.ndarray = None,  # For de-normalizing output
    state_std: np.ndarray = None,   # For de-normalizing output
    device: Optional[str] = None,
    return_numpy: bool = True,
):
    """
    Sample (state, action) trajectories with first-state inpainting.
    
    - cond: normalized conditioning state (as the model expects)
    - Returns: de-normalized trajectory if state_mean/std provided, else normalized
    """
    device = torch.device(device or DEVICE)
    cond_t = torch.as_tensor(cond, device=device, dtype=torch.float32)
    if cond_t.ndim == 1:
        cond_t = cond_t.unsqueeze(0)
    
    B = cond_t.shape[0]
    traj_dim = _get_dim(model, "traj_dim")
    state_dim = _get_dim(model, "state_dim")
    
    # Denoise from noise
    x = torch.randn((B, HORIZON, traj_dim), device=device)
    scheduler.set_timesteps(num_inference_steps, device=device)
    model.eval()
    
    for t in scheduler.timesteps:
        x[:, 0, :state_dim] = cond_t[:, :state_dim]  # Inpaint first state
        out = model(sample=x, timestep=t.expand(B).long(), cond=cond_t, return_dict=True)
        eps = out.sample if hasattr(out, "sample") else out[0]
        x = scheduler.step(eps, t, x).prev_sample
    
    x[:, 0, :state_dim] = cond_t[:, :state_dim]  # Final inpaint
    x[..., state_dim:] = x[..., state_dim:].clamp(-1.0, 1.0)  # Clamp actions
    
    # De-normalize states if stats provided
    if state_mean is not None and state_std is not None:
        mean = torch.as_tensor(state_mean, device=device, dtype=torch.float32)
        std = torch.as_tensor(state_std, device=device, dtype=torch.float32)
        x[..., :state_dim] = x[..., :state_dim] * std + mean
    
    if B == 1:
        x = x.squeeze(0)
    return x.cpu().numpy() if return_numpy else x
