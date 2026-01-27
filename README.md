# Diffusion + MPPI for PointMaze

Diffusion-based trajectory proposal + MPPI refinement for PointMaze (Gymnasium Robotics / Minari D4RL).
The diffusion model predicts state-action trajectories; MPPI refines or replaces them at execution time.

## Highlights
- Conditional diffusion over state-action trajectories (planning horizon in `config.py`).
- MPPI controller with analytic double-integrator dynamics and collision-aware costs.
- Planning modes: MPPI-only, diffusion-only, and diffusion warm-started MPPI.
- Optional BFS distance field and 2D grid video visualization of planned horizons.

## Repo layout (key files)
- `config.py`: Central defaults (dataset/env IDs, horizons, MPPI/diffusion hyperparams, normalization stats).
- `run_diffusion_and_mppi.py`: Main evaluation/planning entry point.
- `diffusion/train_diffusion.py`: Trainer for diffusion models (MLP/CNN/Transformer).
- `diffusion/arch/`: Model architectures.
- `diffusion/diffusion_model_sampling.py`: Diffusion sampling + optional guidance.
- `data/minari_dataset.py`: Minari dataset loader + normalization for diffusion training.
- `MPPI/`: Dynamics, costs, grid viz, and MPPI controller wrapper.
- `TRAINING_AND_EVAL.md`: Expanded notes and examples.

## Setup
Install dependencies:
```bash
pip install -r requirements.txt
```

If you want GPU acceleration, install a CUDA-matching PyTorch build first, then install the rest.

## Dataset
Training downloads the dataset automatically, but evaluation uses `download=False`.
Pre-download if you only plan to evaluate:
```bash
python - <<'PY'
import minari
minari.load_dataset("D4RL/pointmaze/medium-v2", download=True)
PY
```

Defaults are in `config.py`:
- `DATASET_ID = "D4RL/pointmaze/medium-v2"`
- `ENV_ID = "PointMaze_Medium-v3"`
- `DIFFUSION_HORIZON = 100`
- `HORIZON = 30`

## Quick start (MPPI only)
```bash
python run_diffusion_and_mppi.py \
  --plan_method mppi_only \
  --episodes 10 \
  --logs_dir logs/mppi_only
```

## Train diffusion models
Minimal MLP example:
```bash
python -m diffusion.train_diffusion \
  --epochs 20 \
  --batch_size 256 \
  --model_module diffusion.arch.diffusion_mlp_arch \
  --model_class TrajectoryMLPDenoiser \
  --out_dir diffusion/checkpoints
```

Other architectures:
```bash
# CNN
python -m diffusion.train_diffusion \
  --model_module diffusion.arch.diffusion_cnn_arch \
  --model_class TrajectoryCNNDenoiser

# Transformer
python -m diffusion.train_diffusion \
  --model_module diffusion.arch.diffusion_transformer_arch \
  --model_class TrajectoryTransformerDenoiser
```

Useful knobs:
- `--dataset_id`, `--horizon` override config defaults.
- `--num_diffusion_steps`, `--beta_start`, `--beta_end` control the noise schedule.
- `--save_every_epochs`, `--save_every_steps` control checkpoint cadence.
- `--resume path/to/ckpt.pt` resumes training.
- `--use_wandb` enables W&B logging (optional).

Checkpoints are written to `--out_dir` (default is `./checkpoints_diffusion` in the trainer).

## Evaluation / planning
```bash
python run_diffusion_and_mppi.py [args...]
```

Planning methods (`--plan_method`):
- `mppi_only`
- `diffusion_only` (requires `--diff_ckpt`)
- `mppi_warmstart_by_diffusion` (requires `--diff_ckpt`)
- `diffusion_and_one_MPPI_refine` (requires `--diff_ckpt`)

Examples:
```bash
# Diffusion only
python run_diffusion_and_mppi.py \
  --plan_method diffusion_only \
  --diff_arch mlp \
  --diff_ckpt diffusion/checkpoints/final_step_XXXXX.pt \
  --episodes 10 \
  --logs_dir logs/diffusion_only

# Diffusion warm-started MPPI
python run_diffusion_and_mppi.py \
  --plan_method mppi_warmstart_by_diffusion \
  --apply_first_n_actions 3 \
  --warmstart_time_limit 1.0 \
  --diff_arch mlp \
  --diff_ckpt diffusion/checkpoints/val_best_1.pt \
  --episodes 10 \
  --logs_dir logs/warmstart
```

Optional flags:
- `--save_video` saves MuJoCo videos to `logs/videos/`.
- `--save_grid_video` saves a 2D grid overlay of the plan to `logs/grid_videos/`.
- `--plan_with_BFS` uses a BFS distance field for cost shaping.
- `--fixed_goal` disables random goal sampling.

## Outputs
Each evaluation run writes:
- `logs/run_args.json`
- `logs/vanilla_mppi_results.json`
- videos in `logs/videos/` if `--save_video`
- grid videos in `logs/grid_videos/` if `--save_grid_video`

## Sweeps and helpers
- `run_eval_sweep.py`: Batch evaluation across action-horizon settings (edit `DIFF_CKPTS` inside).
- `eval_sweep_denoise.py`: Additional sweep utilities (experimental).
- `A_helper_scripts/`: Misc. plotting and analysis helpers.

## Tips / troubleshooting
- Diffusion-based planning asserts that `--diff_ckpt` is provided.
- Evaluation uses `minari.load_dataset(..., download=False)`. Pre-download if needed.
- `run_diffusion_and_mppi.py` sets `MUJOCO_GL=egl` at import time. For on-screen rendering, change that line to `glfw` (or remove it) and use a display.
- Normalization stats (`NORM_MEAN`, `NORM_STD`) live in `config.py` and are used in training and sampling.
