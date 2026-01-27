import torch
from config import W_GOAL, W_COLLISION, W_CTRL, W_BFS, W_CLEARANCE, CLEARANCE_BUFFER

def mppi_running_cost(state, action, maze_handler=None, distance_map=None, device="cpu"):
    """
    state: (B, 6) or (B, H, 6) -> [x, y, vx, vy, gx, gy]
    action: (B, 2) or (B, H, 2)
    maze_handler: MazeHandler object
    distance_map: torch.Tensor (H, W) for BFS dist (optional)
    
    Returns:
        cost: (B) or (B, H)
    """
    # 1. Goal Cost
    pos = state[..., 0:2]
    goal = state[..., 4:6] 
    
    # Always compute Euclidean (Weuc)
    dist_euc = torch.norm(pos - goal, dim=-1)
    goal_cost = W_GOAL * dist_euc
    
    if distance_map is not None and maze_handler is not None:
        # BFS Distance Field Lookup (Wbfs)
        # Ensure distance_map is on device
        if distance_map.device != state.device:
            distance_map = distance_map.to(state.device)
            
        x = pos[..., 0]
        y = pos[..., 1]
        
        h, w = distance_map.shape
        
        # Grid indices using helper
        ix, iy = maze_handler.world_to_grid(x, y)
        
        # Clamp to bounds to avoid indexing error
        ix = torch.clamp(ix, 0, w - 1)
        iy = torch.clamp(iy, 0, h - 1)
        
        # Lookup
        dist_bfs = distance_map[iy, ix]
        
        # Add to goal cost
        goal_cost += W_BFS * dist_bfs
    
    # 2. Control Cost: || u ||^2
    ctrl_cost = torch.sum(action ** 2, dim=-1)
    
    # 3. Collision Cost
    # Only if maze_handler is available
    if maze_handler is not None:
        is_collision = maze_handler.check_collision_batch(state)
        if is_collision.any():
            #print("Collision detected!")
            pass
        col_cost = is_collision.float() * W_COLLISION
    else:
        col_cost = 0.0
        

    # 4. Soft clearance cost (distance-to-wall hinge)
    if maze_handler is not None and getattr(maze_handler, "wall_dist_map", None) is not None:
        x = pos[..., 0]
        y = pos[..., 1]
        d_wall = maze_handler.wall_distance_world(x, y)
        
        # Base margin
        base_margin = float(getattr(maze_handler, "agent_radius", 0.0)) + float(CLEARANCE_BUFFER)
        
        # Dynamic margin: Inflate safety zone by speed
        # e.g., if speed is 5.0, add 0.5m to the buffer.
        vel = state[..., 2:4]
        speed = torch.norm(vel, dim=-1)
        k_braking = 0.15  # Tuning parameter: "seconds of lookahead"
        
        #dynamic_margin = base_margin + (k_braking * speed) #TODO disable dynamic margin for now
        dynamic_margin = base_margin
        
        # Use dynamic_margin instead of fixed margin
        clearance_cost = W_CLEARANCE * torch.relu(dynamic_margin - d_wall) ** 2
    else:
        print("NO clearence cost!!!!!!!")
        clearance_cost = 0.0
    
    

    # Total
    total_cost = goal_cost + (W_CTRL * ctrl_cost) + col_cost + clearance_cost
    
    return total_cost
