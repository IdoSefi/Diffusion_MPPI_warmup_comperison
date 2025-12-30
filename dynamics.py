import torch
import torch.nn as nn
from config import DT, DAMPING, K_CONTROL, V_MIN, V_MAX

class AnalyticDoubleIntegrator(nn.Module):
    def __init__(self, device="cpu", maze_handler=None):
        super().__init__()
        self.device = device
        self.maze_handler = maze_handler
        # Parameters (scalars, but could be tensors if needed)
        self.dt = DT
        self.damping = DAMPING
        self.k = K_CONTROL
        self.vmin = V_MIN
        self.vmax = V_MAX

    def forward(self, state, action):
        """
        state: (B, 6) -> [x, y, vx, vy, gx, gy]
        action: (B, 2) -> [ux, uy] (assumed in [-1, 1])
        
        Returns:
            next_state: (B, 6)
        """
        # Unpack state
        pos = state[:, 0:2]
        vel = state[:, 2:4]
        goal = state[:, 4:6]  # Goal is constant in dynamics
        
        # Dynamics: v_next = v + (k*u - damping*v) * dt
        # Note: action is already clipped to [-1, 1] by the controller usually, 
        # but we assume incoming action is raw and might need clipping if not handled elsewhere.
        # However, for analytic physics, we usually just apply the formula.
        
        # Acceleration
        acc = self.k * action - self.damping * vel
        
        next_vel = vel + acc * self.dt
        
        # Clip velocity
        next_vel = torch.clamp(next_vel, self.vmin, self.vmax)
        
        # Position update: p_next = p + v_next * dt (Symplectic Euler-ish or just explicit Euler)
        # Using next_vel for position update is more stable (semi-implicit Euler)
        next_pos = pos + next_vel * self.dt
        
        # --- Collision Handling ---
        if self.maze_handler is not None:
            # Discrete collision check at next state
            dummy_next_state = torch.cat(
                [next_pos, torch.zeros_like(next_pos), torch.zeros_like(next_pos)], dim=-1
            )
            is_collision = self.maze_handler.check_collision_batch(dummy_next_state)

            # Continuous collision along the segment (anti-tunneling): check a few interpolated points.
            alphas = torch.tensor([0.25, 0.5, 0.75], device=state.device, dtype=next_pos.dtype)
            seg_pos = pos.unsqueeze(1) + alphas.view(1, -1, 1) * (next_pos - pos).unsqueeze(1)
            dummy_seg = torch.cat(
                [seg_pos, torch.zeros_like(seg_pos), torch.zeros_like(seg_pos)], dim=-1
            )
            seg_collision = self.maze_handler.check_collision_batch(dummy_seg).any(dim=-1)

            is_collision = is_collision | seg_collision
            mask = is_collision.unsqueeze(-1)

            next_pos = torch.where(mask, pos, next_pos)
            next_vel = torch.where(mask, torch.zeros_like(next_vel), next_vel)

# Concat back
        next_state = torch.cat([next_pos, next_vel, goal], dim=1)
        return next_state
