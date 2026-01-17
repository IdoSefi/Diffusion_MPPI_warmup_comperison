import torch

# Environment
ENV_ID = "PointMaze_Medium-v3"
DATASET_ID = "D4RL/pointmaze/medium-v2"

# Dynamics
DT = 0.02  
V_MIN = -5.0
V_MAX = 5.0

# Physics Parameters for "Snappy" Response
# Ratio K/Damping = 5.0 (Target Max Velocity)
# Higher magnitude = Faster response (less drift)
DAMPING = 2.0   
K_CONTROL = 10.0 

# MPPI Parameters
HORIZON = 30    
NUM_SAMPLES = 20

# Diffusion (planning horizon)
DIFFUSION_HORIZON = 100

NOISE_SIGMA = 1.0 # Increased noise to explore "force" space better

LAMBDA = 0.01    # Temperature param for MPPI

# Cost Weights
W_GOAL = 20.0
W_BFS = 4.0      

W_COLLISION = 1000.0
W_CTRL = 0.01    # Lower control cost to allow aggressive turns

# Geometry / safety margin (world units)
AGENT_RADIUS_FALLBACK = 0.15 
CLEARANCE_BUFFER = 0.3
HARD_CLEARANCE_BUFFER = 0.15

# Soft clearance cost weight
W_CLEARANCE = 50.0

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Reproducibility
SEED = 42

# Guided diffusion defaults (planning)
GUIDANCE_SCALE = 0.1
GUIDANCE_GAMMA = 0.997

# Diffusion training defaults (paper-aligned)
DIFFUSION_LR = 4e-5
DIFFUSION_BATCH_SIZE = 32
DIFFUSION_TRAIN_STEPS = 500_000
DIFFUSION_NUM_TRAIN_TIMESTEPS = 100
