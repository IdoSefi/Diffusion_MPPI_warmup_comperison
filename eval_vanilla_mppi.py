import os
os.environ["MUJOCO_GL"] = "egl"
import argparse
import time
import json
import os
import numpy as np
import torch
import minari
import gymnasium as gym

from config import ENV_ID, DATASET_ID, SEED, DEVICE, DT, HORIZON, DIFFUSION_HORIZON, GUIDANCE_GAMMA
from env_utils import parse_obs, MazeHandler
from mppi_controller import MPPIController
from grid_viz import GridVideoWriter, GridVideoConfig

#diffusion models
from diffusers import DDPMScheduler
from diffusion_model_sampling import sample_state_action_trajectory
from diffusion_model_factory import build_denoiser 

def run_eval(args):
    # Create logs dir
    os.makedirs(args.logs_dir, exist_ok=True)
    print(f"Loading Minari dataset: {DATASET_ID}")
        
    dataset = minari.load_dataset(DATASET_ID, download=False)
    print("Recovering environment from dataset...")
    render_mode = "rgb_array" if args.save_video else ("human" if args.render else None)
    env = dataset.recover_environment(eval_env=True, render_mode=render_mode)
    
    if args.save_video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=f"{args.logs_dir}/videos",
            episode_trigger=lambda x: True,
            name_prefix="mppi_eval"
        )
    
    # Seeding
    # Note: gymnasium 0.26+ uses seed in reset, but we can seed numpy/torch globally
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Introspect maze for collision
    maze_handler = MazeHandler(env, device=DEVICE)
    if maze_handler.maze_map is None:
        print("Warning: Maze map not found, collision avoidance disabled.")
    else:
        print("maze_map shape:", maze_handler.maze_map.shape, "unique:", np.unique(maze_handler.maze_map))

    # Optional 2D grid video (no MuJoCo rendering): overlay planned MPPI horizon on the occupancy grid.
    grid_video_enabled = bool(getattr(args, "save_grid_video", False)) and (maze_handler.maze_map is not None)
    if grid_video_enabled:
        os.makedirs(f"{args.logs_dir}/grid_videos", exist_ok=True)
        grid_cfg = GridVideoConfig(
            fps=int(getattr(args, "grid_video_fps", 20)),
            cell_px=int(getattr(args, "grid_cell_px", 24)),
        )

    # --- Collision Sanity Check ---
    print("Running collision sanity check...")
    if maze_handler.maze_map is not None:
        walls = np.argwhere(maze_handler.maze_map == 1)
        free = np.argwhere(maze_handler.maze_map == 0)
        
        if len(walls) > 0:
            w_y, w_x = walls[0]
            test_x, test_y = maze_handler.grid_to_world(w_x, w_y)
            
            test_wall = torch.tensor([[test_x, test_y, 0, 0, 0, 0]], device=DEVICE)
            is_col = maze_handler.check_collision_batch(test_wall)
            print(f"  Test Wall (ix={w_x}, iy={w_y}) -> (x={test_x:.2f}, y={test_y:.2f}): Collision={is_col.item()} (Expected True)")
            assert is_col.item(), "Collision check failed on known wall!"
            
        if len(free) > 0:
            f_y, f_x = free[0]
            test_x, test_y = maze_handler.grid_to_world(f_x, f_y)
                 
            test_free = torch.tensor([[test_x, test_y, 0, 0, 0, 0]], device=DEVICE)
            is_col = maze_handler.check_collision_batch(test_free)
            print(f"  Test Free (ix={f_x}, iy={f_y}) -> (x={test_x:.2f}, y={test_y:.2f}): Collision={is_col.item()} (Expected False)")
            assert not is_col.item(), "Collision check failed on known free cell!"
    print("Collision sanity check passed.")
    # ------------------------------

    
    # Initialize implementation
    controller = MPPIController(maze_handler=maze_handler, device=DEVICE, plan_iteration=args.plan_iteration)

    #diffusion model initialization
    need_diffusion = args.use_diffusion_policy or args.plan_method in ["diffusion_only", "mppi_warmstart_by_diffusion"]
    if need_diffusion:
        assert args.diff_ckpt is not None, "--diff_ckpt is required for diffusion-based planning"

        action_dim = int(np.prod(env.action_space.shape))
        state_dim = 6  # parse_obs gives [x, y, vx, vy, gx, gy]
        traj_dim = state_dim + action_dim  # trajectory includes both state and action
        cond_dim = state_dim  # conditioning is the full state

        diff_model = build_denoiser(
            arch=args.diff_arch,
            horizon=DIFFUSION_HORIZON,
            traj_dim=traj_dim,
            cond_dim=cond_dim,
            state_dim=state_dim,
            action_dim=action_dim,
        ).to(DEVICE)

        ckpt = torch.load(args.diff_ckpt, map_location="cpu")
        state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        diff_model.load_state_dict(state_dict)
        diff_model.eval()

        diff_sched = DDPMScheduler(
            num_train_timesteps=args.diff_num_train_timesteps,
            beta_start=args.diff_beta_start,
            beta_end=args.diff_beta_end,
            beta_schedule=args.diff_beta_schedule,
            prediction_type="epsilon",   # match what you trained
            clip_sample=False,
        )
        print(f"diffusion model of arch {args.diff_arch} initialized on {DEVICE}")

    print(f"MPPI Controller initialized on {DEVICE}")
    print("Controller has maze handler:", controller.maze_handler is not None)
    print("Controller wall dist map exists:", getattr(controller.maze_handler, "wall_dist_map", None) is not None)
    print("Controller agent radius:", getattr(controller.maze_handler, "agent_radius", None))
    
    success_count = 0
    total_steps = 0
    latencies = []
    
    results = {
        "episodes": [],
        "summary": {}
    }
    
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep) # Deterministic per episode
        
        # Grid video writer for this episode (optional)
        grid_writer = None
        executed_rc = []
        if grid_video_enabled:
            out_path = f"{args.logs_dir}/grid_videos/mppi_grid_ep{ep+1}.mp4"
            grid_writer = GridVideoWriter(maze_handler.maze_map, out_path, cfg=grid_cfg)

        
        # --- BFS Setup for Episode ---
        if args.plan_with_BFS:
            # Extract goal from obs
            if isinstance(obs, dict) and 'desired_goal' in obs:
                 goal = obs['desired_goal'] # (2,)
                 goal_x, goal_y = goal[0], goal[1]
                 
                 # Compute BFS map
                 bfs_map = maze_handler.compute_bfs_map((goal_x, goal_y))
                 
                 controller.set_distance_map(bfs_map)
            else:
                 print("Warning: Could not extract goal for BFS planning. Using Euclidean.")
                 controller.set_distance_map(None)
        else:
             controller.set_distance_map(None)
        # -----------------------------
        
        controller.reset()
        
        done = False
        step = 0
        success = False
        collision_count = 0
        mppi_refinement_count = 0
        plan_count = 0
        
        ep_start_time = time.time()
        action_buffer = []
        
        while not done:
            if args.render and not args.save_video and hasattr(env, 'render'):
                env.render()
                
            # Parse state
            state_np = parse_obs(obs)
            
            # Plan
            t0 = time.time()
            if args.plan_method == "diffusion_only":
                traj = sample_state_action_trajectory(
                    model=diff_model,
                    scheduler=diff_sched,
                    cond=state_np,  # numpy (6,)
                    num_inference_steps=args.diff_num_inference_steps,
                    device=DEVICE,
                    return_numpy=True,
                )  # (DIFFUSION_HORIZON, traj_dim), numpy where traj_dim = state_dim + action_dim
                
                # Extract actions from trajectory (last action_dim dimensions)
                u_traj = traj[:, state_dim:]  # (DIFFUSION_HORIZON, action_dim)

                action = u_traj[0]  # first action of the sampled plan (receding horizon) 

            elif args.plan_method == "mppi_only":
                action = controller.get_action(state_np)

            elif args.plan_method == "mppi_warmstart_by_diffusion":
                if not action_buffer:
                    plan_count += 1
                    plan_start = time.time()
                    traj = sample_state_action_trajectory(
                        model=diff_model,
                        scheduler=diff_sched,
                        cond=state_np,
                        num_inference_steps=args.diff_num_inference_steps,
                        device=DEVICE,
                        return_numpy=True,
                    )
                    u_traj = traj[:, state_dim:]
                    u_tensor = torch.tensor(u_traj, device=DEVICE, dtype=torch.float32)
                    if u_tensor.shape[0] < HORIZON:
                        u_tensor = torch.cat([u_tensor, u_tensor[-1:].repeat(HORIZON - u_tensor.shape[0], 1)], dim=0)
                    controller.mppi.U = u_tensor[:HORIZON]
                    state_t = torch.tensor(state_np, dtype=torch.float32, device=DEVICE)
                    action_t = controller.mppi.command(state_t)
                    end_time = plan_start + args.warmstart_time_limit
                    refinement_iters = 0
                    while time.time() < end_time:
                        action_t = controller.mppi.command(state_t, shift_nominal_trajectory=False)
                        refinement_iters += 1
                    mppi_refinement_count += refinement_iters
                    plan_actions = torch.cat([action_t.unsqueeze(0), controller.mppi.U[:-1]], dim=0)
                    controller.mppi._last_plan_actions = plan_actions
                    n = max(1, args.apply_first_n_actions)
                    action_buffer = [a.cpu().numpy() for a in plan_actions[:n]]
                action = action_buffer.pop(0)

            else:
                raise ValueError(f"Unknown plan_method: {args.plan_method}")
            t1 = time.time()
            latencies.append((t1 - t0) * 1000) # ms

            # --- 2D Grid Visualization (planned MPPI trajectory) ---
            if grid_writer is not None:
                # Current agent and goal in world coordinates
                ax, ay = float(state_np[0]), float(state_np[1])
                gx, gy = float(state_np[4]), float(state_np[5])

                # Convert to continuous grid coordinates (row, col) to preserve sub-cell precision
                a_col_f, a_row_f = maze_handler.world_to_grid_float(ax, ay)
                g_col_f, g_row_f = maze_handler.world_to_grid_float(gx, gy)
                agent_rc = (float(a_row_f), float(a_col_f))
                goal_rc = (float(g_row_f), float(g_col_f))

                # Track executed path (in grid)
                executed_rc.append([agent_rc[0], agent_rc[1]])

                # Planned horizon: roll out nominal MPPI controls (H+1 points)

                try:
                    plan_xy = controller.get_planned_xy(state_np)  # (H+1,2) world (x,y)
                    p_col_f, p_row_f = maze_handler.world_to_grid_float(plan_xy[:, 0], plan_xy[:, 1])
                    planned_rc = np.stack([p_row_f, p_col_f], axis=1).astype(np.float32)
                except Exception:
                    planned_rc = None

                grid_writer.add_frame(
                    agent_rc=agent_rc,
                    goal_rc=goal_rc,
                    planned_rc=planned_rc,
                    executed_rc=np.asarray(executed_rc, dtype=np.float32),
                    step_idx=step,
                )
            # -------------------------------------------------------
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            
            # Check collision
            base_env = env.unwrapped
            is_hard_collision = False
            
            if hasattr(base_env, "data"):
                # 1. Get agent speed (magnitude of XY velocity)
                # We use this to define "Hard" collision.
                agent_vel = base_env.data.qvel[:2]
                speed = np.linalg.norm(agent_vel)
                
                # Thresholds
                SPEED_THRESHOLD = 0.5  # Adjust this: 0.5 m/s is a reasonable "hit"
                VERTICAL_TOLERANCE = 0.8 # Filter out floor (Z-component of normal)

                for i in range(base_env.data.ncon):
                    contact = base_env.data.contact[i]
                    normal_z = contact.frame[2]
                    
                    is_wall = abs(normal_z) < VERTICAL_TOLERANCE
                    
                    # 3. Register collision if it's a Wall AND we are moving fast enough
                    if is_wall and speed > SPEED_THRESHOLD:
                        is_hard_collision = True
                        break
            
            if is_hard_collision:
                collision_count += 1

            
            # Check success (PointMaze usually has 'is_success' in info)
            if info.get('is_success', False):
                success = True
            
            # Fallback success check (dist < 0.5 usually)
            # Fallback success check
            if not success:
                # Calculate dist manually
                s_curr = parse_obs(obs)
                pos_c = s_curr[:2]
                goal_c = s_curr[4:6]
                dist_g = np.linalg.norm(pos_c - goal_c)
                
                # DEBUG: Print distance periodically
                if step % 200 == 0:
                    print(f"  Ep {ep+1} Step {step}: Dist to goal = {dist_g:.3f}")
                    # print(f"  Info keys: {list(info.keys())}")
                
                if dist_g < 0.4: 
                    success = True
                    #print("  (Manual success triggered)")
                    break
            
        mean_mppi_refine_per_plan = mppi_refinement_count / plan_count if plan_count > 0 else 0.0
        print(f"Episode {ep+1}: Steps={step}, Success={success}, Mean Latency={np.mean(latencies[-step:]):.1f}ms, MPPI Refinements={mppi_refinement_count}, Plans={plan_count}, Mean Refine/Plan={mean_mppi_refine_per_plan:.1f}")
        
        results["episodes"].append({
            "episode": ep,
            "steps": step,
            "total_wall_clock_time_sec": time.time() - ep_start_time,
            "total_env_time_sec": info.get("time_env_steps_sec", None),
            "total_env_time_our_simple_sec": step * DT,
            "success": bool(success),
            "latency_mean": float(np.mean(latencies[-step:])),
            "collisions": collision_count,
            "mppi_refinement_steps": mppi_refinement_count,
            "plan_count": plan_count,
            "mean_mppi_refinement_per_plan": float(mppi_refinement_count / plan_count) if plan_count > 0 else 0.0
        })
        
        if success:
            success_count += 1
            
    # Summary
    summary = {
        "success_rate": success_count / args.episodes,
        "mean_steps": np.mean([e["steps"] for e in results["episodes"]]),
        "mean_latency_ms": np.mean(latencies) if latencies else 0.0
    }
    results["summary"] = summary
    
    print("\n=== Eval Summary ===")
    print(json.dumps(summary, indent=2))
    
    with open(f"{args.logs_dir}/vanilla_mppi_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {args.logs_dir}/vanilla_mppi_results.json")
    
    env.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", type=int, default=0, help="Enable human rendering (0 or 1)")
    parser.add_argument("--save_video", action="store_true", help="Save video of episodes")
    parser.add_argument("--save_grid_video", action="store_true", help="Save a 2D grid MPPI-plan video (grid + goal + agent + planned horizon)")
    parser.add_argument("--grid_video_fps", type=int, default=20)
    parser.add_argument("--grid_cell_px", type=int, default=24)
    parser.add_argument("--plan_with_BFS", action="store_true", help="Use BFS distance field for planning")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--plan_iteration", type=int, default=1)
    parser.add_argument("--logs_dir", type=str, default="logs", help="Directory to save logs and videos")
    parser.add_argument("--plan_method", type=str, default="diffusion_only", help="Planning method: diffusion_only, mppi_only, or mppi_warmstart_by_diffusion")
    parser.add_argument("--apply_first_n_actions", type=int, default=1, help="Number of planned actions to execute before replanning (warmstart mode)")
    parser.add_argument("--warmstart_time_limit", type=float, default=1.0, help="Seconds allocated to diffusion+MPPI planning in warmstart mode")

    ##diffusion:
    parser.add_argument("--use_diffusion_policy", action="store_true", help="If set, action comes from diffusion (first action of sampled trajectory)")
    parser.add_argument("--diff_arch", type=str, default="mlp", choices=["mlp", "cnn", "transformer"])
    parser.add_argument("--diff_ckpt", type=str, default=None, help="Path to diffusion checkpoint (.pt)")
    parser.add_argument("--diff_num_train_timesteps", type=int, default=100)
    parser.add_argument("--diff_num_inference_steps", type=int, default=100)
    parser.add_argument("--diff_beta_start", type=float, default=1e-4)
    parser.add_argument("--diff_beta_end", type=float, default=2e-2)
    parser.add_argument("--diff_beta_schedule", type=str, default="squaredcos_cap_v2", choices=["linear", "scaled_linear", "squaredcos_cap_v2"])

    args = parser.parse_args()
    
    run_eval(args)
