import torch

# Environment
ENV_ID = "PointMaze_Large-v3"
DATASET_ID = "D4RL/pointmaze/large-v2"

# Dynamics
DT = 0.1  # 10Hz
V_MIN = -5.0
V_MAX = 5.0
DAMPING = 0.5  # Approximate damping for PointMaze (needs tuning if behavior is off)
K_CONTROL = 1.0  # Approximate control gain

# MPPI Parameters
HORIZON = 50
NUM_SAMPLES = 500

NOISE_SIGMA = 0.7

LAMBDA = 0.01  # Temperature param for MPPI

# Cost Weights
W_GOAL = 10.0
W_BFS = 0.1

W_COLLISION = 500000.0
W_CTRL = 0.1

# Geometry / safety margin (world units)
AGENT_RADIUS_FALLBACK = 0.10
CLEARANCE_BUFFER = 0.2
HARD_CLEARANCE_BUFFER = 0.1

# Soft clearance cost weight (discourages getting close to walls)
W_CLEARANCE = 200.0

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Reproducibility
SEED = 42
