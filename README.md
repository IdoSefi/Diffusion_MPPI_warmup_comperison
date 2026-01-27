# Vanilla MPPI Baseline for PointMaze

This project implements a Vanilla MPPI controller for the `PointMaze_Large-v3` environment, using an analytic dynamics model (double integrator) and Minari for dataset/environment management.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the evaluation script:

```bash
python run_diffusion_and_mppi.py --episodes 10 --render 0
```

### Arguments
- `--episodes`: Number of episodes to run (default: 10).
- `--render`: 1 to enable rendering, 0 to disable (default: 0).
- `--seed`: Random seed (default: 42).

## Key Components
- `run_diffusion_and_mppi.py`: Main entry point. Loads Minari dataset and runs eval loop.
- `mppi_controller.py`: Wrapper around `pytorch_mppi` that uses our custom dynamics/costs.
- `dynamics.py`: Analytic double-integrator dynamics ($x_{t+1} \approx x_t + v_t \Delta t$).
- `costs.py`: Running cost function (Goal distance + Collision + Control).
- `env_utils.py`: Utilities to introspect the Gym environment and extract the maze grid for collision checking.
- `config.py`: Hyperparameters.

## Outputs
Results are saved to `logs/vanilla_mppi_results.json`.
