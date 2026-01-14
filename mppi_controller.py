import torch
import numpy as np
from pytorch_mppi import MPPI
from config import (
    HORIZON, NUM_SAMPLES, NOISE_SIGMA, LAMBDA, 
    V_MIN, V_MAX, DEVICE
)
from dynamics import AnalyticDoubleIntegrator
from costs import mppi_running_cost

class MPPIController:
    def __init__(self, maze_handler=None, device="cpu", plan_iteration=1):
        self.device = device
        self.maze_handler = maze_handler
        self.plan_iteration = plan_iteration
        
        # Dynamics Model
        self.dynamics_model = AnalyticDoubleIntegrator(device=device, maze_handler=maze_handler)
        
        # Cost Function
        def cost_fn(state, action):
            return mppi_running_cost(state, action, maze_handler=self.maze_handler, distance_map=self.distance_map, device=self.device)
        
        self.cost_fn = cost_fn
        self.distance_map = None # Can be set per episode  
        # 3. Initialize MPPI
        # Action bounds: [-1, 1] usually for normalized envs. 
        # Initial action mean: 0
        
        self.mppi = MPPI(
            dynamics=self.dynamics_model.forward,
            running_cost=self.cost_fn,
            nx=6,
            num_samples=NUM_SAMPLES,
            horizon=HORIZON,
            noise_sigma=torch.tensor([[NOISE_SIGMA, 0.0], [0.0, NOISE_SIGMA]], device=self.device),
            lambda_=LAMBDA,
            device=self.device,
            u_min=torch.tensor([-1.0, -1.0], device=self.device),
            u_max=torch.tensor([1.0, 1.0], device=self.device)
        )
        
    def reset(self):
        """Resets MPPI internal state (action buffer)"""
        self.mppi.reset()
        
    def set_distance_map(self, distance_map):
        """Updates the distance map for the cost function."""
        if distance_map is not None:
             self.distance_map = distance_map.to(self.device)
        else:
             self.distance_map = None
        
    def get_action(self, current_state_np):
        """
        current_state_np: (6,) numpy array
        
        Returns:
            action: (2,) numpy array
        """
        # Convert to tensor
        state_t = torch.tensor(current_state_np, dtype=torch.float32, device=self.device)
        
        # MPPI command expects state
        action_t = self.mppi.command(state_t)
        for _ in range(self.plan_iteration-1):
            action_t = self.mppi.command(state_t, shift_nominal_trajectory=False)

        # --- Debug Check ---
        if self.maze_handler is not None:
            with torch.no_grad():
                # Reconstruct planned horizon: [current_action, *future_plan]
                # Assuming self.mppi.u holds the shifted action means (Horizon, 2)
                plan_actions = torch.cat([action_t.unsqueeze(0), self.mppi.U[:-1]], dim=0)
                
                curr_state = state_t.unsqueeze(0)
                trajectory_collisions = []
                
                for t in range(plan_actions.shape[0]):
                    u = plan_actions[t].unsqueeze(0)
                    
                    # 1. Predict candidate next state (simulating dynamics physics)
                    pos = curr_state[:, 0:2]
                    vel = curr_state[:, 2:4]
                    
                    acc = self.dynamics_model.k * u - self.dynamics_model.damping * vel
                    next_vel = vel + acc * self.dynamics_model.dt
                    next_vel = torch.clamp(next_vel, self.dynamics_model.vmin, self.dynamics_model.vmax)
                    next_pos = pos + next_vel * self.dynamics_model.dt
                    
                    # 2. Check collision on candidate
                    dummy_next = torch.cat([next_pos, torch.zeros_like(next_pos), torch.zeros_like(next_pos)], dim=-1)
                    is_col = self.maze_handler.check_collision_batch(dummy_next)
                    
                    if is_col.any():
                        trajectory_collisions.append(t)
                        
                    # 3. Update state using actual dynamics (which handles collision response)
                    curr_state = self.dynamics_model.forward(curr_state, u)

                    pass
        
        return action_t.cpu().detach().numpy()

    @torch.no_grad()
    def get_planned_xy(self, current_state_np): #TODO FOR DEBUG
        """Return the current nominal MPPI plan as world (x,y) points.

        This uses MPPI's internal nominal control sequence (self.mppi.U) and rolls the
        analytic dynamics forward (including collision handling if enabled there).

        Args:
            current_state_np: numpy array shape (6,) [x,y,vx,vy,gx,gy]

        Returns:
            xy: numpy array shape (H+1, 2)
        """
        if not hasattr(self, "mppi") or self.mppi is None:
            raise RuntimeError("MPPI is not initialized.")
        if not hasattr(self.mppi, "U") or self.mppi.U is None:
            raise RuntimeError("MPPI object has no nominal trajectory U.")
        state_t = torch.tensor(current_state_np, dtype=torch.float32, device=self.device).unsqueeze(0)

        # We want the full horizon that matches what MPPI just computed.
        # Prefer the plan sequence built in get_action() (starts with executed action),
        # then fall back to MPPI's nominal U.
        U = getattr(self, "_last_plan_actions", None)
        if U is None:
            U = self.mppi.U

        # Ensure shape is (H, nu) and length equals HORIZON.
        if U.dim() != 2:
            U = U.view(U.shape[0], -1)
        if U.shape[0] != HORIZON:
            if U.shape[0] > HORIZON:
                U = U[:HORIZON]
            else:
                pad = U[-1:].repeat(HORIZON - U.shape[0], 1)
                U = torch.cat([U, pad], dim=0)

        pts = [state_t[0, 0:2]]
        curr = state_t
        for t in range(U.shape[0]):
            u_t = U[t].unsqueeze(0)
            curr = self.dynamics_model.forward(curr, u_t)
            pts.append(curr[0, 0:2])
        xy = torch.stack(pts, dim=0)
        return xy.detach().cpu().numpy()