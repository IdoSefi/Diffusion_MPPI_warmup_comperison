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
from dataclasses import asdict
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
from config import DATASET_ID, HORIZON, DEVICE, SEED
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

    return x.clamp(-1.0, 1.0)


def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, step: int, args: argparse.Namespace):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "args": vars(args),
    }
    torch.save(payload, path)


def load_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer | None = None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=True)
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    step = int(ckpt.get("step", 0))
    return step, ckpt


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
    
    model.train()


def main():
    parser = argparse.ArgumentParser()

    # Data / project defaults
    parser.add_argument("--dataset_id", type=str, default=DATASET_ID)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--seed", type=int, default=SEED)

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision (CUDA only).")

    # Diffusion scheduler (training)
    parser.add_argument("--num_diffusion_steps", type=int, default=100)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=2e-2)

    # Model loader (architecture-only)
    parser.add_argument("--model_module", type=str, default="diffusion_mlp_arch")
    parser.add_argument("--model_class", type=str, default="TrajectoryMLPDenoiser")
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--time_emb_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.0)

    # Checkpointing / logging
    parser.add_argument("--out_dir", type=str, default="./checkpoints_diffusion")
    parser.add_argument("--save_every_steps", type=int, default=2000)
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
    parser.add_argument("--num_inference_steps", type=int, default=50)
    
    # Trajectory visualization
    parser.add_argument("--visualize_trajectories", type=int, default=0, help="Number of trajectories to visualize per epoch (0 disables)")

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # Dataset / loader (your existing code)
    dataset = MinariDiffusionDataset(args.dataset_id, horizon_T=args.horizon)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    # Infer dims from one batch (keeps this script architecture-agnostic)
    first_batch = next(iter(loader))
    state_dim, action_dim, traj_dim, horizon = infer_dims_from_batch(first_batch)
    if horizon != args.horizon:
        print(f"[WARN] loader horizon={horizon} differs from args.horizon={args.horizon}. Using loader horizon.")
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

    # Diffusers scheduler (handles add_noise / step)
    scheduler = DDPMScheduler(
        num_train_timesteps=args.num_diffusion_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule="linear",
        clip_sample=False,  # we clamp actions manually when sampling
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
        global_step, ckpt = load_checkpoint(args.resume, model, optimizer)
        print(f"[RESUME] loaded step={global_step} from {args.resume}")

    model.train()
    t0 = time.time()

    for epoch in range(args.epochs):
        for batch in loader:
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
                model.eval()
                with torch.no_grad():
                    cond_small = state[:8]
                    samp = sample_sanity_check(
                        model=model,
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
                model.train()

            # Checkpoint
            if global_step % args.save_every_steps == 0:
                ckpt_path = os.path.join(args.out_dir, f"ckpt_step_{global_step}.pt")
                save_checkpoint(ckpt_path, model, optimizer, global_step, args)
                print(f"[ckpt] saved: {ckpt_path}")
                
                if args.use_wandb:
                    wandb.save(ckpt_path, base_path=os.path.dirname(ckpt_path))

        # End of epoch checkpoint
        ckpt_path = os.path.join(args.out_dir, f"ckpt_epoch_{epoch:03d}_step_{global_step}.pt")
        save_checkpoint(ckpt_path, model, optimizer, global_step, args)
        print(f"[ckpt] saved: {ckpt_path}")
        
        # Visualize trajectories
        if args.visualize_trajectories > 0:
            visualize_trajectories(
                model=model,
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
            wandb.log({"train/epoch_completed": epoch + 1}, step=global_step)
            wandb.save(ckpt_path, base_path=os.path.dirname(ckpt_path))
            # Log trajectory visualization if available
            if args.visualize_trajectories > 0:
                viz_path = os.path.join(args.out_dir, 'trajectory_viz', f'epoch_{epoch:03d}.png')
                if os.path.exists(viz_path):
                    wandb.log({"trajectories": wandb.Image(viz_path)}, step=global_step)

    final_path = os.path.join(args.out_dir, f"final_step_{global_step}.pt")
    save_checkpoint(final_path, model, optimizer, global_step, args)
    print(f"[done] saved final: {final_path}")
    
    if args.use_wandb:
        wandb.save(final_path, base_path=os.path.dirname(final_path))
        wandb.finish()
        print("[wandb] run finished")


if __name__ == "__main__":
    main()

#example of run with MLP:
#python train_diffusion.py --epochs 20 --batch_size 256 --model_module diffusion_mlp_arch --model_class TrajectoryMLPDenoiser --use_wandb --visualize_trajectories 4 --wandb_group mlp_bringup
