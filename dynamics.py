import torch
import torch.nn as nn
from config import DT, DAMPING, K_CONTROL, V_MIN, V_MAX

class AnalyticDoubleIntegrator(nn.Module):
    def __init__(self, device="cpu", maze_handler=None):
        super().__init__()
        self.device = device
        self.maze_handler = maze_handler
        
        # Physics Parameters
        self.dt = DT
        self.damping = DAMPING
        self.k = K_CONTROL
        self.vmin = V_MIN
        self.vmax = V_MAX

    def forward(self, state, action):
        """
        state: (B, 6) -> [x, y, vx, vy, gx, gy]
        action: (B, 2) -> [ux, uy]
        """
        # Unpack state
        pos = state[:, 0:2]
        vel = state[:, 2:4]
        goal = state[:, 4:6]
        
        # 1. Clip action to [-1, 1] (Simulate Actuator Limits)
        action = torch.clamp(action, -1.0, 1.0)
        
        # 2. Dynamics: Acceleration = (Force - Damping * Velocity) / Mass
        # Assuming Mass = 1.0 for PointMaze
        acc = self.k * action - self.damping * vel
        
        # 3. Semi-Implicit Euler Integration (More stable for damped systems)
        # Update velocity first
        next_vel = vel + acc * self.dt
        
        # Clip velocity (Environment speed limit)
        next_vel = torch.clamp(next_vel, self.vmin, self.vmax)
        
        # Update position using the NEW velocity
        next_pos = pos + next_vel * self.dt
        
        # --- Collision Handling ---
        if self.maze_handler is not None:
            # Discrete Check (Target Position)
            dummy_next = torch.cat([next_pos, torch.zeros_like(next_pos), torch.zeros_like(next_pos)], dim=-1)
            is_collision = self.maze_handler.check_collision_batch(dummy_next)

            # Continuous Check (Midpoint to prevent tunneling)
            mid_pos = (pos + next_pos) / 2.0
            dummy_mid = torch.cat([mid_pos, torch.zeros_like(mid_pos), torch.zeros_like(mid_pos)], dim=-1)
            mid_collision = self.maze_handler.check_collision_batch(dummy_mid)
            
            is_collision = is_collision | mid_collision
            
            # Inelastic Collision Response (Stop at wall)
            mask = is_collision.unsqueeze(-1)
            next_pos = torch.where(mask, pos, next_pos) # Stay at previous pos
            next_vel = torch.where(mask, torch.zeros_like(next_vel), next_vel) # Kill velocity

        # Reassemble State
        next_state = torch.cat([next_pos, next_vel, goal], dim=1)
        return next_state