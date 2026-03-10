# Diffusion based warm starting MPPI controller for robotic planning

### this project was done in Deep Learning course at the Technion: 046217
### authors: Ido Sefi 208008698, Yoav Vinov 208300954

Diffusion-based warm-start for an MPPI controller in robotic planning

This project was done as part of the Deep Learning course at the Technion (046217).
Authors: Ido Sefi (208008698), Yoav Vinov (208300954)


## Overview

This project tests whether using diffusion as a warm-start for MPPI improves the trade-off between trajectory quality
and planning latency, and compares denoiser backbones (MLP / CNN / Transformer).
In this project we study diffusion models for predicting fixed-horizon trajectories for
navigation in the D4RL PointMaze environment, and compare them to a classical sampling based planner, Model Predictive Path Integral (MPPI). We implement a DDPM-style diffusion model and evaluate three denoiser backbones: MLP, temporal CNN, and Transformer.
We focus on a hybrid setting where the diffusion model provides a warm start (initial trajectory guess) for MPPI under a fixed planning time budget. In our evaluation, MPPI-only
consistently achieved lower mean episode steps to reach the goal than diffusion warm-started
MPPI, while CNN and Transformer warm starts outperformed the MLP warm start across
most replanning settings.

[![Demo video](https://img.youtube.com/vi/Vm95qW2hwg8/0.jpg)](https://www.youtube.com/watch?v=Vm95qW2hwg8)

## 📌 Key Idea

- MPPI is robust but typically starts from a random trajectory → needs iterations.
- Diffusion proposes a full trajectory from offline data quickly, but can be imperfect.
- Warm-start = initialize MPPI with the diffusion proposal and refine (MPC loop).
- We compare diffusion denoiser backbones: MLP vs 1D CNN vs Transformer.

## MPPI warmstarting pipline:

<img width="1362" height="271" alt="image" src="https://github.com/user-attachments/assets/b610ef92-9d5f-43b4-9652-8af5d8f9e422" />
Fixed compute budget. We generate a trajectory proposal with a diffusion planner (trained offline on D4RL PointMaze), then optionally refine it with MPPI using the remaining planning budget. This warm-start MPPI trades diffusion sampling time for fewer MPPI iterations.


## Implementation notes
- Diffusion defines the iterative denoising process, while the backbone architecture (MLP/CNN/Transformer) executes the denoising steps.
- We incorporate positional embeddings into both CNN and Transformer denoisers to encode time information in the action sequence.
- Trajectory horizon for diffusion samples: 100 actions.

## Experiments (high-level)
- Baseline: MPPI only.
- Warm-start: sample one diffusion trajectory, use it to initialize MPPI (instead of a random guess), then run MPPI refinement until a fixed time budget.
- Evaluation: 80 episodes, 30 denoising steps.
- We also study replanning frequency by varying n_actions (how many actions are executed from each plan before replanning).

## Results (key findings)
- CNN and Transformer warm-starts produce better trajectories than the MLP warm-start (lower mean episode steps).
- However, under the same planning time budget, MPPI-only achieved the best overall performance in this PointMaze setup.
- Increasing n_actions (executing more actions per plan before replanning) generally degraded warm-start performance, especially for the MLP.
We suspect this is because PointMaze uses low-dimensional actions where classic MPPI refinement is very effective, and diffusion sampling consumes part of the available time budget.
<img width="2037" height="1131" alt="image" src="https://github.com/user-attachments/assets/fa281b9a-944d-4c4d-9fc5-7da6a59a1b43" />

## Denoiser Backbones (MLP / CNN / Transformer)

We train diffusion to generate a **horizon-100** trajectory. For CNN and Transformer we add **positional embeddings** to encode time along the trajectory.

| Denoiser | Depth | #Params | Notes |
|---|---:|---:|---|
| MLP | 4 | 1,438,850 | baseline |
| 1D CNN | 6 | 3,426,492 | + positional embedding |
| Transformer | 6 | 19,527,170 | + positional embedding |

the experiment was evaluated over 80 episode for each arch at each n_actions.
each arch was given 0.2 seconds to make the diffusion sampling + MPPI refinment steps with the leftover time. the MPPI only did MPPI iterations for 0.2 seconds.
each plan is for a trajectory of 100 actions. from it only the first n_actions are excecuted.

## Future work
- Test on higher-dimensional control tasks, where a strong learned proposal may provide larger gains.
- Explore stronger guidance and improved cost-aware sampling.

---

## Repo layout

```
.
├── run_diffusion_and_mppi.py          # main evaluation / planning entrypoint
├── config.py                          # default hyperparams (env, horizons, costs, normalization)
├── requirements.txt
├── data/
│   ├── minari_dataset.py              # Minari dataset loader + normalization for diffusion training
│   └── verify_dataset.py              # dataset sanity checks / utilities
├── diffusion/
│   ├── train_diffusion.py             # diffusion trainer (HuggingFace diffusers scheduler)
│   ├── diffusion_model_sampling.py    # sampling + first-state inpainting + optional guidance
│   ├── diffusion_model_factory.py     # model builder (mlp/cnn/transformer)
│   └── arch/                          # backbone architectures
└── MPPI/
    ├── mppi_controller.py             # MPPI wrapper (pytorch-mppi)
    ├── dynamics.py                    # analytic double integrator + collision handling hooks
    ├── costs.py                       # goal + collision + control + BFS shaping
    ├── env_utils.py                   # PointMaze parsing helpers
    └── grid_viz.py                    # 2D grid plan visualization video
```

---

## Setup
Install dependencies:
```bash
pip install -r requirements.txt
```


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

---

## Related work

this project was built upon "planning with diffusion": https://diffusion-planning.github.io/

