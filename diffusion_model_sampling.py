# diffusion_model_sampling.py
"""Simple diffusion sampling for (state, action) trajectories."""
from __future__ import annotations
from typing import Optional, Union
import numpy as np
import torch
from config import HORIZON, DEVICE

# Optional guided-sampling knobs (fallbacks if not defined in config.py)
try:
    from config import (
        GUIDANCE_SCALE,
        GUIDANCE_GAMMA,
        GUIDANCE_SIGMA,
        GUIDANCE_HOLD_L,
        GUIDANCE_LAMBDA_HOLD,
        GUIDANCE_LAMBDA_SMOOTH,
        GUIDANCE_POS_IDX,
        GUIDANCE_GOAL_IDX,
    )
except Exception:
    GUIDANCE_SCALE = 0.0
    GUIDANCE_GAMMA = 0.99
    GUIDANCE_SIGMA = 0.5
    GUIDANCE_HOLD_L = 0
    GUIDANCE_LAMBDA_HOLD = 0.0
    GUIDANCE_LAMBDA_SMOOTH = 0.0
    GUIDANCE_POS_IDX = (0, 1)
    GUIDANCE_GOAL_IDX = (4, 5)


def _get_dim(model, name: str) -> int:
    """Get dimension from model config or attribute."""
    if hasattr(model, "config") and hasattr(model.config, name):
        return int(getattr(model.config, name))
    if hasattr(model, name):
        return int(getattr(model, name))
    raise ValueError(f"Cannot determine {name} from model")


def sample_state_action_trajectory(
    model,
    scheduler,
    cond: Union[np.ndarray, torch.Tensor],  # (C,) or (B,C) - should be NORMALIZED if model trained on normalized
    num_inference_steps: int = 25,
    state_mean: np.ndarray = None,  # For de-normalizing output
    state_std: np.ndarray = None,   # For de-normalizing output
    # --- Guided diffusion knobs (override config defaults) ---
    guidance_scale: Optional[float] = 0.3,
    gamma: Optional[float] = None,
    sigma: Optional[float] = None,
    hold_L: Optional[int] = None,
    lambda_hold: Optional[float] = None,
    lambda_smooth: Optional[float] = None,
    pos_idx: Optional[tuple[int, int]] = None,
    goal_idx: Optional[tuple[int, int]] = None,
    device: Optional[str] = None,
    return_numpy: bool = True,
):
    """
    Sample (state, action) trajectories with first-state inpainting.
    
    - cond: normalized conditioning state (as the model expects)
    - Optional test-time guidance biases trajectories to reach the goal early.
    - After each denoising step, hard-inpaints s0 to match the current env state (MPC consistency).
    - Returns: de-normalized trajectory if state_mean/std provided, else normalized
    """
    device = torch.device(device or DEVICE)
    cond_t = torch.as_tensor(cond, device=device, dtype=torch.float32)
    if cond_t.ndim == 1:
        cond_t = cond_t.unsqueeze(0)
    
    B = cond_t.shape[0]
    traj_dim = _get_dim(model, "traj_dim")
    state_dim = _get_dim(model, "state_dim")
    
    # Resolve guidance knobs (defaults from config.py)
    guidance_scale = float(GUIDANCE_SCALE if guidance_scale is None else guidance_scale)
    gamma = float(GUIDANCE_GAMMA if gamma is None else gamma)
    sigma = float(GUIDANCE_SIGMA if sigma is None else sigma)
    hold_L = int(GUIDANCE_HOLD_L if hold_L is None else hold_L)
    lambda_hold = float(GUIDANCE_LAMBDA_HOLD if lambda_hold is None else lambda_hold)
    lambda_smooth = float(GUIDANCE_LAMBDA_SMOOTH if lambda_smooth is None else lambda_smooth)
    pos_idx = GUIDANCE_POS_IDX if pos_idx is None else pos_idx
    goal_idx = GUIDANCE_GOAL_IDX if goal_idx is None else goal_idx
    
    # Fast sanity checks (avoid silent wrong indexing)
    if max(pos_idx + goal_idx) >= state_dim:
        raise ValueError(
            f"pos_idx={pos_idx} / goal_idx={goal_idx} must be within state_dim={state_dim}"
        )
    if cond_t.shape[-1] < max(goal_idx) + 1:
        raise ValueError(
            f"cond has dim {cond_t.shape[-1]} but goal_idx={goal_idx} requires >= {max(goal_idx)+1}"
        )
    
    # Denoise from noise
    x = torch.randn((B, HORIZON, traj_dim), device=device)
    scheduler.set_timesteps(num_inference_steps, device=device)
    model.eval()
    
    for t in scheduler.timesteps:
        x[:, 0, :state_dim] = cond_t[:, :state_dim]  # Inpaint first state
        
        # Base denoising step (no grad needed for model/scheduler)
        with torch.no_grad():
            out = model(sample=x, timestep=t.expand(B).long(), cond=cond_t, return_dict=True)
            eps = out.sample if hasattr(out, "sample") else out[0]
            x_prev = scheduler.step(eps, t, x).prev_sample
        
        # Optional test-time guidance (autograd w.r.t. x_prev only)
        if guidance_scale > 0.0:
            xg = x_prev.detach().requires_grad_(True)

            # Objective in normalized units (or raw units depending on cond)
            s = xg[..., :state_dim]  # (B, H, state_dim)
            pos = s[..., list(pos_idx)]  # (B, H, 2)
            goal = cond_t[:, list(goal_idx)].unsqueeze(1)  # (B, 1, 2)

            dist2 = ((pos - goal) ** 2).sum(dim=-1)  # (B, H)
            r = torch.exp(-dist2 / (2.0 * (sigma ** 2) + 1e-12))  # (B, H)

            # Discounted early-reaching reward
            w = (gamma ** torch.arange(HORIZON, device=device, dtype=torch.float32)).unsqueeze(0)  # (1, H)
            J = (w * r).sum(dim=1)  # (B,)

            # Optional hold reward on last L steps
            if hold_L > 0 and lambda_hold > 0.0:
                J = J + lambda_hold * r[:, HORIZON - hold_L : HORIZON].sum(dim=1)

            # Optional action smoothness penalty (subtract)
            if lambda_smooth > 0.0:
                a = xg[..., state_dim:]  # (B, H, action_dim)
                da = a[:, 1:, :] - a[:, :-1, :]
                smooth = (da ** 2).sum(dim=-1).sum(dim=1)  # (B,)
                J = J - lambda_smooth * smooth

            grad = torch.autograd.grad(J.sum(), xg, create_graph=False, retain_graph=False)[0]
            x_prev = x_prev + guidance_scale * grad

        # Enforce MPC consistency + action bounds each step
        x_prev[:, 0, :state_dim] = cond_t[:, :state_dim]
        x_prev[..., state_dim:] = x_prev[..., state_dim:].clamp(-1.0, 1.0)
        x = x_prev
    
    # Final inpaint + clamp actions
    x[:, 0, :state_dim] = cond_t[:, :state_dim]
    x[..., state_dim:] = x[..., state_dim:].clamp(-1.0, 1.0)
    
    # De-normalize states if stats provided
    if state_mean is not None and state_std is not None:
        mean = torch.as_tensor(state_mean, device=device, dtype=torch.float32)
        std = torch.as_tensor(state_std, device=device, dtype=torch.float32)
        x[..., :state_dim] = x[..., :state_dim] * std + mean
    
    if B == 1:
        x = x.squeeze(0)
    return x.cpu().numpy() if return_numpy else x
