## Diffusion + MPPI: training and evaluation

This repo’s main entry points are:
- Training: `python -m diffusion.train_diffusion`
- Evaluation: `python run_diffusion_and_mppi.py`

Note: Do not use `run_eval_sweep.py` or `eval_sweep_denoise.py` for the workflows below.

---

## Prereqs

1) Install dependencies (from repo root):
```bash
pip install -r requirements.txt
```

2) Ensure the Minari dataset is available.
Training downloads it automatically, but evaluation uses `download=False`.
If you only want to eval, you can pre-download:
```bash
python - <<'PY'
import minari
minari.load_dataset("D4RL/pointmaze/medium-v2", download=True)
PY
```

Defaults come from `config.py`:
- `DATASET_ID = "D4RL/pointmaze/medium-v2"`
- `ENV_ID = "PointMaze_Medium-v3"`
- `DIFFUSION_HORIZON = 100`
- `HORIZON = 30`

---

## Training diffusion model

### Minimal example (MLP)
```bash
python -m diffusion.train_diffusion \
  --epochs 20 \
  --batch_size 256 \
  --model_module diffusion.arch.diffusion_mlp_arch \
  --model_class TrajectoryMLPDenoiser \
  --out_dir diffusion/checkpoints
```

### Other architectures
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

### Useful knobs
- `--dataset_id` and `--horizon` override config defaults.
- `--num_diffusion_steps`, `--beta_start`, `--beta_end` control the noise schedule.
- `--save_every_epochs` and `--save_every_steps` control checkpoint cadence.
- `--resume path/to/ckpt.pt` resumes training.
- Checkpoints are written to `--out_dir` (default now `diffusion/checkpoints`):
  - `val_best_*.pt` (best-on-val checkpoints)
  - `ckpt_epoch_*.pt`, `ckpt_step_*.pt` (periodic)
  - `final_step_*.pt` (final)

---

## Evaluation / planning

Evaluation is done with:
```bash
python run_diffusion_and_mppi.py [args...]
```

Outputs go to `--logs_dir` (default `logs/`):
- `run_args.json`
- `vanilla_mppi_results.json`
- videos in `logs/videos/` if `--save_video`
- grid videos in `logs/grid_videos/` if `--save_grid_video`

### MPPI only (no diffusion checkpoint needed)
```bash
python run_diffusion_and_mppi.py \
  --plan_method mppi_only \
  --episodes 10 \
  --logs_dir logs/mppi_only
```

### Diffusion only (requires a trained checkpoint)
```bash
python run_diffusion_and_mppi.py \
  --plan_method diffusion_only \
  --diff_arch mlp \
  --diff_ckpt diffusion/checkpoints/final_step_XXXXX.pt \
  --episodes 10 \
  --logs_dir logs/diffusion_only
```

### MPPI warm-started by diffusion
```bash
python run_diffusion_and_mppi.py \
  --plan_method mppi_warmstart_by_diffusion \
  --apply_first_n_actions 3 \
  --warmstart_time_limit 1.0 \
  --diff_arch mlp \
  --diff_ckpt diffusion/checkpoints/val_best_1.pt \
  --episodes 10 \
  --logs_dir logs/warmstart
```

### Optional rendering / videos
```bash
# On-screen rendering (may require a display)
python run_diffusion_and_mppi.py --render 1

# Save MuJoCo video
python run_diffusion_and_mppi.py --save_video

# Save 2D grid visualization of the plan (requires maze map)
python run_diffusion_and_mppi.py --save_grid_video
```

---

## Common gotchas
- Diffusion-based planning requires `--diff_ckpt` (the script asserts this).
- Evaluation uses `minari.load_dataset(..., download=False)`, so pre-download if needed.
- `--plan_method` choices: `diffusion_only`, `mppi_only`, `mppi_warmstart_by_diffusion`, `diffusion_and_one_MPPI_refine`.
