#!/usr/bin/env python3
"""
train_diffusion.py

Generic trainer for conditional diffusion over state-action trajectories using Hugging Face diffusers.

Project integration:
- imports DATASET_ID, HORIZON, DEVICE, SEED from config.py
- uses MinariDiffusionDataset from minari_dataset.py (returns dict with keys: 'state', 'traj_window')

Model contract (architecture-only):
- forward(sample=x_t, timestep=t, cond=state, return_dict=True) -> output with `.sample` = eps_hat
  where:
    x_t:    (B, H, traj_dim)  where traj_dim = state_dim + action_dim
    t:      (B,) int64 timesteps
    state:  (B, state_dim)

Default model (MLP) expected at:
  diffusion_mlp_arch.py : class TrajectoryMLPDenoiser
but you can swap any architecture via CLI without touching this script.

Example:
  python train_diffusion.py --epochs 20 --batch_size 256 \
    --model_module diffusion_mlp_arch --model_class TrajectoryMLPDenoiser
"""

from __future__ import annotations

import argparse
import importlib
import os
import time
from copy import deepcopy
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import minari

# --- project imports (your repo) ---
from config import (
    DATASET_ID,
    DIFFUSION_HORIZON,
    DEVICE,
    SEED,
    DIFFUSION_LR,
    DIFFUSION_BATCH_SIZE,
    DIFFUSION_TRAIN_STEPS,
    DIFFUSION_NUM_TRAIN_TIMESTEPS,
)
from minari_dataset import MinariDiffusionDataset
from dynamics import AnalyticDoubleIntegrator
from env_utils import MazeHandler

# --- diffusers scheduler ---
try:
    from diffusers import DDPMScheduler
except Exception as e:
    raise ImportError(
        "Hugging Face diffusers is required for this trainer.\n"
        "Install: pip install diffusers accelerate\n"
        f"Original import error: {e}"
    )

# --- wandb for experiment tracking (optional) ---
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[WARN] wandb not installed. Install with: pip install wandb")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dynamic_load_model(module_name: str, class_name: str, kwargs: Dict[str, Any]) -> nn.Module:
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    model = cls(**kwargs)
    return model


def infer_dims_from_batch(batch: Dict[str, torch.Tensor]) -> Tuple[int, int, int, int]:
    """
    Returns: (state_dim, action_dim, traj_dim, horizon)
    """
    state = batch["state"]                 # (B, state_dim)
    traj_window = batch["traj_window"]     # (B, H, traj_dim)
    
    if traj_window.ndim != 3:
        raise ValueError(f"Expected traj_window (B,H,traj_dim), got {tuple(traj_window.shape)}")
    
    state_dim = state.shape[-1]
    horizon = traj_window.shape[1]
    traj_dim = traj_window.shape[2]
    action_dim = traj_dim - state_dim
    
    if action_dim <= 0:
        raise ValueError(f"traj_dim ({traj_dim}) must be > state_dim ({state_dim})")
    
    return state_dim, action_dim, traj_dim, horizon


@torch.no_grad()
def sample_sanity_check(
    model: nn.Module,
    scheduler: DDPMScheduler,
    cond: torch.Tensor,
    horizon: int,
    action_dim: int,
    num_inference_steps: int,
) -> torch.Tensor:
    """
    Optional sampling sanity-check: start from Gaussian noise and denoise.
    This is only for debugging / monitoring; your real planning will likely differ.
    """
    was_training = model.training
    model.eval()
    device = cond.device

    x = torch.randn((cond.shape[0], horizon, action_dim), device=device)
    scheduler.set_timesteps(num_inference_steps, device=device)

    for t in scheduler.timesteps:
        # t is scalar tensor; expand to (B,)
        t_batch = t.expand(cond.shape[0]).long()
        out = model(sample=x, timestep=t_batch, cond=cond, return_dict=True)
        eps_hat = out.sample if hasattr(out, "sample") else out[0]
        step_out = scheduler.step(eps_hat, t, x)
        x = step_out.prev_sample

    out = x.clamp(-1.0, 1.0)
    model.train(was_training)
    return out


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    ema_state: Dict[str, torch.Tensor] | None = None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "args": vars(args),
    }
    if ema_state is not None:
        payload["ema_state"] = ema_state
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    ema_model: nn.Module | None = None,
):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if ema_model is not None and "ema_state" in ckpt:
        ema_model.load_state_dict(ckpt["ema_state"], strict=True)
    step = int(ckpt.get("step", 0))
    return step, ckpt


def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    if ema_model is None:
        return
    if decay <= 0.0:
        return
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.copy_(ema_param * decay + param * (1.0 - decay))
        for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)


@torch.no_grad()
def visualize_trajectories(
    model: nn.Module,
    scheduler: DDPMScheduler,
    dataset_id: str,
    horizon: int,
    traj_dim: int,
    state_dim: int,
    num_inference_steps: int,
    num_trajectories: int,
    epoch: int,
    out_dir: str,
    device: torch.device,
):
    """Sample trajectories from diffusion model and visualize with dynamics rollout.
    
    Reuses sampling and visualization patterns from verify_dataset.py for consistency.
    """
    was_training = model.training
    model.eval()
    
    # Initialize dynamics model (same as verify_dataset.py)
    dynamics = AnalyticDoubleIntegrator(device=device)
    
    # Get maze handler and dataset (same as verify_dataset.py)
    minari_dataset = minari.load_dataset(dataset_id, download=False)
    env = minari_dataset.recover_environment(eval_env=True, render_mode=None)
    maze_handler = MazeHandler(env, device=device)
    dynamics.maze_handler = maze_handler
    
    # Get first episode for reference states (same as verify_dataset.py)
    episode = next(iter(minari_dataset.iterate_episodes()))
    observations = episode.observations['observation']  # Shape: (T, 4)
    goals = episode.observations['desired_goal']        # Shape: (T, 2)
    
    # Concatenate obs + goal to match MinariDiffusionDataset format (cond_dim=6)
    combined_states = np.concatenate([observations, goals], axis=-1)  # Shape: (T, 6)
    
    goal = goals[0, :2]  # For plotting the goal marker
    
    # Calculate starting indices (same pattern as verify_dataset.py)
    num_states = min(num_trajectories, len(combined_states))
    if num_trajectories > 1:
        step_size = len(combined_states) // (num_trajectories + 1)
        indices = [i * step_size for i in range(num_trajectories)]
    else:
        indices = [0]
    
    # Setup figure grid (same as verify_dataset.py)
    num_cols = 3
    num_rows = (num_trajectories + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    
    if num_trajectories == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Generate and plot trajectories
    for plot_idx, start_idx in enumerate(indices):
        if plot_idx >= num_trajectories or start_idx >= len(combined_states):
            break
        
        # Get initial state with obs + goal concatenated (matches training format)
        initial_state = torch.from_numpy(combined_states[start_idx]).float()  # (6,)
        
        # Sample trajectory from diffusion model using sample_sanity_check
        sampled_traj = sample_sanity_check(
            model=model,
            scheduler=scheduler,
            cond=initial_state.unsqueeze(0).to(device),
            horizon=horizon,
            traj_dim=traj_dim,
            state_dim=state_dim,
            num_inference_steps=num_inference_steps,
        )  # (1, H, traj_dim)
        
        # Extract actions from trajectory
        sampled_actions = sampled_traj[0, :, state_dim:].cpu().numpy()  # (H, action_dim)
        sampled_actions = sampled_actions[0].cpu().numpy()  # (horizon, action_dim)
        
        # Rollout trajectory using dynamics (same as verify_dataset.py)
        # Note: dynamics expects full state (6-dim), using only observation part (4-dim) for position
        trajectory = [observations[start_idx].copy()]  # Use raw obs for positions
        current_state = torch.from_numpy(observations[start_idx]).float().unsqueeze(0).to(device)  # (1, 4)
        
        for t in range(horizon):
            action = torch.from_numpy(sampled_actions[t]).float().unsqueeze(0).to(device)  # (1, 2)
            next_state = dynamics(current_state, action)
            trajectory.append(next_state[0].cpu().detach().numpy())
            current_state = next_state
        
        trajectory = np.array(trajectory)  # (horizon+1, 4)
        rollout_positions = trajectory[:, :2]  # (horizon+1, 2)
        
        # Plot with gradient color (same style as verify_dataset.py)
        ax = axes[plot_idx]
        
        # Create gradient colormap from dark red to bright pink
        cmap = LinearSegmentedColormap.from_list('red_to_pink', 
                                                  ['#8B0000', '#FF1493'])  # Dark red to deep pink
        
        # Create line segments
        points = rollout_positions.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create line collection with gradient colors
        lc = LineCollection(segments, cmap=cmap, linewidth=2.5, alpha=0.8)
        lc.set_array(np.linspace(0, 1, len(segments)))
        line = ax.add_collection(lc)
        
        # Add proxy artist for legend (same as verify_dataset.py)
        ax.plot([], [], color='#8B0000', linewidth=2.5, alpha=0.8, 
                label='Diffusion rollout (red→pink)', linestyle='-')
        
        # Plot start, end, goal (same as verify_dataset.py)
        ax.plot(rollout_positions[0, 0], rollout_positions[0, 1], 'go', markersize=12, 
                label='Start', zorder=5)
        ax.plot(rollout_positions[-1, 0], rollout_positions[-1, 1], 'ro', markersize=12, 
                label='End', zorder=5)
        ax.plot(goal[0], goal[1], 'y*', markersize=20, label='Goal', zorder=6)
        
        ax.set_title(f'Start state {start_idx} ({horizon} steps rollout)', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.axis('equal')
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
    
    # Hide unused subplots (same as verify_dataset.py)
    for idx in range(plot_idx + 1, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    viz_dir = os.path.join(out_dir, 'trajectory_viz')
    os.makedirs(viz_dir, exist_ok=True)
    save_path = os.path.join(viz_dir, f'epoch_{epoch:03d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    env.close()
    print(f"[viz] saved: {save_path}")
    
    model.train(was_training)


def main():
    parser = argparse.ArgumentParser()

    # Data / project defaults
    parser.add_argument("--dataset_id", type=str, default=DATASET_ID)
    parser.add_argument("--horizon", type=int, default=DIFFUSION_HORIZON)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--seed", type=int, default=SEED)

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=DIFFUSION_TRAIN_STEPS, help="Stop after this many steps (0 disables)")
    parser.add_argument("--batch_size", type=int, default=DIFFUSION_BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=DIFFUSION_LR)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision (CUDA only).")
    parser.add_argument("--val_frac", type=float, default=0.05, help="Fraction of data for validation split")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (in eval checks)")
    parser.add_argument("--min_delta", type=float, default=0.0, help="Minimum delta for val improvement")
    parser.add_argument("--eval_every_epochs", type=int, default=1, help="Validate every N epochs")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay (0 disables)")

    # Diffusion scheduler (training)
    parser.add_argument("--num_diffusion_steps", type=int, default=DIFFUSION_NUM_TRAIN_TIMESTEPS)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=2e-2)

    # Model loader (architecture-only)
    parser.add_argument("--model_module", type=str, default="diffusion_mlp_arch")
    parser.add_argument("--model_class", type=str, default="TrajectoryMLPDenoiser")
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--time_emb_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Checkpointing / logging
    parser.add_argument("--out_dir", type=str, default="./checkpoints_diffusion")
    parser.add_argument("--save_every_steps", type=int, default=0, help="Save checkpoint every N steps (0 disables)")
    parser.add_argument("--save_every_epochs", type=int, default=50, help="Save checkpoint every N epochs")
    parser.add_argument("--log_every_steps", type=int, default=200)
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint .pt to resume from")

    # Wandb tracking
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases tracking")
    parser.add_argument("--wandb_project", type=str, default="diffusion-mppi", help="W&B project name")
    parser.add_argument("--wandb_group", type=str, default="", help="W&B group name (optional)")
    parser.add_argument("--wandb_run_name", type=str, default="", help="W&B run name (optional, auto-generated if empty)")
    parser.add_argument("--wandb_entity", type=str, default="", help="W&B entity/team name (optional)")

    # Optional sampling sanity-check
    parser.add_argument("--sample_every_steps", type=int, default=0, help="0 disables. e.g., 1000")
    parser.add_argument("--num_inference_steps", type=int, default=DIFFUSION_NUM_TRAIN_TIMESTEPS)
    
    # Trajectory visualization
    parser.add_argument("--visualize_trajectories", type=int, default=0, help="Number of trajectories to visualize per epoch (0 disables)")

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    pin_memory = device.type == "cuda"
    persistent_workers = args.num_workers > 0

    # Dataset split + deterministic loaders
    dataset = MinariDiffusionDataset(args.dataset_id, horizon_T=args.horizon)
    split_generator = torch.Generator().manual_seed(args.seed)
    loader_generator = torch.Generator().manual_seed(args.seed)

    val_len = 0
    if args.val_frac > 0:
        val_len = max(1, int(len(dataset) * args.val_frac))
        if val_len >= len(dataset):
            val_len = max(len(dataset) - 1, 0)
    train_len = len(dataset) - val_len
    if train_len <= 0:
        raise ValueError("Validation split too large; no training samples remain.")

    if val_len > 0:
        train_ds, val_ds = torch.utils.data.random_split(
            dataset, [train_len, val_len], generator=split_generator
        )
    else:
        train_ds, val_ds = dataset, None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        generator=loader_generator,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

    # Infer dims from one batch (keeps this script architecture-agnostic)
    first_batch = next(iter(train_loader))
    state_dim, action_dim, traj_dim, horizon = infer_dims_from_batch(first_batch)
    if horizon != args.horizon:
        print(f"[WARN] train loader horizon={horizon} differs from args.horizon={args.horizon}. Using loader horizon.")
        args.horizon = horizon

    # Build model kwargs; required keys for your arch
    model_kwargs = dict(
        horizon=args.horizon,
        traj_dim=traj_dim,
        cond_dim=state_dim,  # cond is still batch["state"]
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        time_emb_dim=args.time_emb_dim,
        dropout=args.dropout,
    )
    model = dynamic_load_model(args.model_module, args.model_class, model_kwargs).to(device)
    ema_model = None
    if args.ema_decay > 0:
        ema_model = deepcopy(model).to(device)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)

    # Diffusers scheduler (handles add_noise / step)
    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_diffusion_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=False,  # we clamp actions manually when sampling
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    # Initialize wandb if requested
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            print("[ERROR] --use_wandb specified but wandb is not installed. Disabling wandb.")
            args.use_wandb = False
        else:
            wandb_config = {
                "dataset_id": args.dataset_id,
                "horizon": args.horizon,
                "state_dim": state_dim,
                "action_dim": action_dim,
                "traj_dim": traj_dim,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "grad_clip": args.grad_clip,
                "num_diffusion_steps": args.num_diffusion_steps,
                "beta_start": args.beta_start,
                "beta_end": args.beta_end,
                "hidden_dim": args.hidden_dim,
                "depth": args.depth,
                "time_emb_dim": args.time_emb_dim,
                "dropout": args.dropout,
                "model_module": args.model_module,
                "model_class": args.model_class,
                "seed": args.seed,
                "max_steps": args.max_steps,
                "val_frac": args.val_frac,
                "patience": args.patience,
                "min_delta": args.min_delta,
                "eval_every_epochs": args.eval_every_epochs,
                "ema_decay": args.ema_decay,
            }
            wandb_kwargs = {
                "project": args.wandb_project,
                "config": wandb_config,
            }
            if args.wandb_entity:
                wandb_kwargs["entity"] = args.wandb_entity
            if args.wandb_group:
                wandb_kwargs["group"] = args.wandb_group
            if args.wandb_run_name:
                wandb_kwargs["name"] = args.wandb_run_name
            
            wandb.init(**wandb_kwargs)
            wandb.watch(model, log="all", log_freq=args.log_every_steps)
            print(f"[wandb] initialized: project={args.wandb_project}, run={wandb.run.name}")

    global_step = 0
    if args.resume:
        global_step, ckpt = load_checkpoint(args.resume, model, optimizer, ema_model)
        print(f"[RESUME] loaded step={global_step} from {args.resume}")

    model.train()
    t0 = time.time()
    best_val_loss = float("inf")
    epochs_since_improve = 0
    peak_number = 1
    stop_training = False

    for epoch in range(args.epochs):
        for batch in train_loader:
            state = batch["state"].to(device)          # (B, state_dim)
            x0 = batch["traj_window"].to(device)       # (B, H, traj_dim)

            # Safety: keep actions bounded (clamp only actions, not states)
            x0[..., state_dim:] = x0[..., state_dim:].clamp(-1.0, 1.0)

            B = x0.shape[0]
            t = torch.randint(
                low=0,
                high=scheduler.config.num_train_timesteps,
                size=(B,),
                device=device,
                dtype=torch.long,
            )
            noise = torch.randn_like(x0)
            x_t = scheduler.add_noise(x0, noise, t)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(scaler.is_enabled())):
                out = model(sample=x_t, timestep=t, cond=state, return_dict=True)
                eps_hat = out.sample if hasattr(out, "sample") else out[0]
                loss = F.mse_loss(eps_hat, noise)

            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            if ema_model is not None:
                update_ema(ema_model, model, args.ema_decay)

            global_step += 1

            # Logging
            if global_step % args.log_every_steps == 0:
                dt = time.time() - t0
                loss_val = loss.item()
                print(
                    f"[step {global_step:7d}] epoch={epoch:03d} "
                    f"loss={loss_val:.6f} "
                    f"({dt:.1f}s elapsed)"
                )
                
                if args.use_wandb:
                    wandb.log({
                        "train/loss": loss_val,
                        "train/epoch": epoch,
                        "train/step": global_step,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/elapsed_time": dt,
                    }, step=global_step)

            # Optional sampling sanity-check
            if args.sample_every_steps and (global_step % args.sample_every_steps == 0):
                cond_small = state[:8]
                eval_model = ema_model if ema_model is not None else model
                samp = sample_sanity_check(
                    model=eval_model,
                    scheduler=scheduler,
                    cond=cond_small,
                    horizon=args.horizon,
                    traj_dim=traj_dim,
                    state_dim=state_dim,
                    num_inference_steps=args.num_inference_steps,
                )
                samp_mean = samp.mean().item()
                samp_std = samp.std().item()
                samp_min = samp.min().item()
                samp_max = samp.max().item()
                print(f"[sample] shape={tuple(samp.shape)} mean={samp_mean:.4f} std={samp_std:.4f}")
                
                if args.use_wandb:
                    wandb.log({
                        "sample/mean": samp_mean,
                        "sample/std": samp_std,
                        "sample/min": samp_min,
                        "sample/max": samp_max,
                    }, step=global_step)

            # Checkpoint
            if args.save_every_steps > 0 and (global_step % args.save_every_steps == 0):
                ckpt_path = os.path.join(args.out_dir, f"ckpt_step_{global_step}.pt")
                ema_state = ema_model.state_dict() if ema_model is not None else None
                save_checkpoint(ckpt_path, model, optimizer, global_step, args, ema_state=ema_state)
                print(f"[ckpt] saved: {ckpt_path}")
                
                if args.use_wandb:
                    wandb.save(ckpt_path, base_path=os.path.dirname(ckpt_path))

            if args.max_steps and global_step >= args.max_steps:
                stop_training = True
                print(f"[stop] Reached max_steps={args.max_steps}, stopping training.")
                # Save checkpoint before breaking
                ckpt_path = os.path.join(args.out_dir, f"ckpt_early_stop_epoch_{epoch:03d}_step_{global_step}.pt")
                ema_state = ema_model.state_dict() if ema_model is not None else None
                save_checkpoint(ckpt_path, model, optimizer, global_step, args, ema_state=ema_state)
                print(f"[early stop ckpt] saved: {ckpt_path}")
                if args.use_wandb:
                    wandb.save(ckpt_path, base_path=os.path.dirname(ckpt_path))
                break

        if stop_training:
            break

        val_loss = None
        if val_loader is not None and (epoch + 1) % args.eval_every_epochs == 0:
            model_was_training = model.training
            model.eval()
            val_losses = []
            with torch.no_grad():
                for val_batch in val_loader:
                    val_state = val_batch["state"].to(device)
                    val_x0 = val_batch["traj_window"].to(device)
                    val_x0[..., state_dim:] = val_x0[..., state_dim:].clamp(-1.0, 1.0)

                    B_val = val_x0.shape[0]
                    t_val = torch.randint(
                        low=0,
                        high=scheduler.config.num_train_timesteps,
                        size=(B_val,),
                        device=device,
                        dtype=torch.long,
                    )
                    val_noise = torch.randn_like(val_x0)
                    val_x_t = scheduler.add_noise(val_x0, val_noise, t_val)

                    val_out = model(sample=val_x_t, timestep=t_val, cond=val_state, return_dict=True)
                    val_eps_hat = val_out.sample if hasattr(val_out, "sample") else val_out[0]
                    val_loss_batch = F.mse_loss(val_eps_hat, val_noise)
                    val_losses.append(val_loss_batch.item())

            model.train(model_was_training)

            val_loss = float(np.mean(val_losses)) if len(val_losses) > 0 else float("inf")
            scheduler_lr.step(val_loss)

            if args.use_wandb:
                wandb.log({
                    "val/loss": val_loss,
                    "val/epoch": epoch,
                    "train/step": global_step,
                    "train/lr": optimizer.param_groups[0]["lr"],
                }, step=global_step)

            improved = val_loss < (best_val_loss - args.min_delta)
            if improved:
                if epochs_since_improve > 0:
                    peak_number += 1
                best_val_loss = val_loss
                epochs_since_improve = 0
                model_to_save = ema_model if ema_model is not None else model
                ema_state = ema_model.state_dict() if ema_model is not None else None
                best_path = os.path.join(args.out_dir, f"val_best_{peak_number}.pt")
                save_checkpoint(best_path, model_to_save, optimizer, global_step, args, ema_state=ema_state)
                print(f"[best peak {peak_number}] epoch={epoch:03d} step={global_step} val_loss={val_loss:.6f} saved={best_path}")
            else:
                epochs_since_improve += 1

        if stop_training:
            break

        # End of epoch checkpoint (only every N epochs)
        if (epoch + 1) % args.save_every_epochs == 0:
            ckpt_path = os.path.join(args.out_dir, f"ckpt_epoch_{epoch:03d}_step_{global_step}.pt")
            ema_state = ema_model.state_dict() if ema_model is not None else None
            save_checkpoint(ckpt_path, model, optimizer, global_step, args, ema_state=ema_state)
            print(f"[ckpt] saved: {ckpt_path}")
            
            if args.use_wandb:
                wandb.save(ckpt_path, base_path=os.path.dirname(ckpt_path))
        
        # Visualize trajectories (only every N epochs)
        if args.visualize_trajectories > 0 and (epoch + 1) % args.save_every_epochs == 0:
            eval_model = ema_model if ema_model is not None else model
            visualize_trajectories(
                model=eval_model,
                scheduler=scheduler,
                dataset_id=args.dataset_id,
                horizon=args.horizon,
                traj_dim=traj_dim,
                state_dim=state_dim,
                num_inference_steps=args.num_inference_steps,
                num_trajectories=args.visualize_trajectories,
                epoch=epoch,
                out_dir=args.out_dir,
                device=device,
            )
            
            if args.use_wandb:
                viz_path = os.path.join(args.out_dir, 'trajectory_viz', f'epoch_{epoch:03d}.png')
                if os.path.exists(viz_path):
                    wandb.log({"trajectories": wandb.Image(viz_path)}, step=global_step)
        
        if args.use_wandb:
            payload = {"train/epoch_completed": epoch + 1}
            if val_loss is not None:
                payload["val/loss_epoch_end"] = val_loss
            wandb.log(payload, step=global_step)

    final_path = os.path.join(args.out_dir, f"final_step_{global_step}.pt")
    ema_state = ema_model.state_dict() if ema_model is not None else None
    save_checkpoint(final_path, model, optimizer, global_step, args, ema_state=ema_state)
    print(f"[done] saved final: {final_path}")
    
    if args.use_wandb:
        wandb.save(final_path, base_path=os.path.dirname(final_path))
        wandb.finish()
        print("[wandb] run finished")


if __name__ == "__main__":
    main()

#example of run with MLP:
#python train_diffusion.py --epochs 20 --batch_size 256 --model_module diffusion_mlp_arch --model_class TrajectoryMLPDenoiser --use_wandb --visualize_trajectories 4 --wandb_group mlp_bringup
