"""
Verification script to inspect and visualize the Minari dataset with 2D plots.
"""

import minari
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from config import DATASET_ID, DEVICE, HORIZON
from env_utils import MazeHandler
import gymnasium as gym
from grid_viz import GridVideoWriter, GridVideoConfig
from minari_dataset import MinariDiffusionDataset
from dynamics import AnalyticDoubleIntegrator
import torch
import os

# Create output folder
OUTPUT_DIR = "dataset_verify"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def verify_dataset_structure():
    """Inspect dataset basic properties."""
    print("=" * 60)
    print("DATASET STRUCTURE VERIFICATION")
    print("=" * 60)
    
    dataset = minari.load_dataset(DATASET_ID, download=False)
    print(f"Dataset ID: {DATASET_ID}")
    print(f"Total episodes: {len(dataset)}")
    
    # Inspect first episode
    first_episode = next(iter(dataset.iterate_episodes()))
    print(f"\nFirst Episode:")
    print(f"  Steps: {len(first_episode.actions)}")
    print(f"  Observation keys: {list(first_episode.observations.keys())}")
    print(f"  Observation shape: {first_episode.observations['observation'].shape}")
    print(f"  Desired goal shape: {first_episode.observations['desired_goal'].shape}")
    print(f"  Action shape: {first_episode.actions.shape}")
    print(f"  Action range: [{first_episode.actions.min():.3f}, {first_episode.actions.max():.3f}]")
    
    # Collect stats
    episode_lengths = []
    for episode in dataset.iterate_episodes():
        episode_lengths.append(len(episode.actions))
    
    print(f"\nEpisode Statistics:")
    print(f"  Min length: {min(episode_lengths)}")
    print(f"  Max length: {max(episode_lengths)}")
    print(f"  Mean length: {np.mean(episode_lengths):.1f}")
    print(f"  Median length: {np.median(episode_lengths):.1f}")


def plot_trajectories_2d(num_episodes: int = 6):
    """Plot 2D trajectories from dataset episodes."""
    print("\n" + "=" * 60)
    print("2D TRAJECTORY VISUALIZATION")
    print("=" * 60)
    
    dataset = minari.load_dataset(DATASET_ID, download=False)
    
    # Calculate grid dimensions
    num_cols = 3
    num_rows = (num_episodes + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    
    if num_episodes == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for ep_idx, episode in enumerate(dataset.iterate_episodes()):
        if ep_idx >= num_episodes:
            break
        
        # Extract positions (first 2 dims of observation)
        positions = episode.observations['observation'][:, :2]
        goals = episode.observations['desired_goal'][0, :2]  # Goal is constant per episode
        
        ax = axes[ep_idx]
        
        # Plot trajectory
        ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label='Trajectory', alpha=0.7)
        
        # Plot start
        ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=12, label='Start', zorder=5)
        
        # Plot end
        ax.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=12, label='End', zorder=5)
        
        # Plot goal
        ax.plot(goals[0], goals[1], 'y*', markersize=20, label='Goal', zorder=6)
        
        ax.set_title(f'Episode {ep_idx + 1} ({len(positions)} steps)', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.axis('equal')
    
    # Hide unused subplots
    for idx in range(ep_idx + 1, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/dataset_trajectories.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR}/dataset_trajectories.png")
    plt.close()


def plot_action_statistics():
    """Plot action distribution across dataset."""
    print("\n" + "=" * 60)
    print("ACTION STATISTICS")
    print("=" * 60)
    
    dataset = minari.load_dataset(DATASET_ID, download=False)
    
    all_actions = []
    for episode in dataset.iterate_episodes():
        all_actions.append(episode.actions)
    
    all_actions = np.vstack(all_actions)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram per action dimension
    for dim in range(all_actions.shape[1]):
        axes[0].hist(all_actions[:, dim], bins=30, alpha=0.6, label=f'Action {dim}')
    
    axes[0].set_xlabel('Action Value')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Action Value Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot([all_actions[:, i] for i in range(all_actions.shape[1])])
    axes[1].set_xlabel('Action Dimension')
    axes[1].set_ylabel('Value')
    axes[1].set_title('Action Range per Dimension')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/action_statistics.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR}/action_statistics.png")
    plt.close()
    
    print(f"Total actions collected: {len(all_actions)}")
    for dim in range(all_actions.shape[1]):
        print(f"  Action {dim}: [{all_actions[:, dim].min():.3f}, {all_actions[:, dim].max():.3f}]")


def visualize_with_maze(num_episodes: int = 3):
    """Visualize trajectories overlaid on the maze map using GridVideoWriter."""
    print("\n" + "=" * 60)
    print("MAZE-OVERLAID TRAJECTORY VISUALIZATION")
    print("=" * 60)
    
    # Recover environment and maze
    dataset = minari.load_dataset(DATASET_ID, download=False)
    env = dataset.recover_environment(eval_env=True, render_mode=None)
    maze_handler = MazeHandler(env, device=DEVICE)
    
    if maze_handler.maze_map is None:
        print("⚠ Warning: Maze map not available. Skipping maze visualization.")
        env.close()
        return
    
    os.makedirs(f"{OUTPUT_DIR}/dataset_videos", exist_ok=True)
    
    for ep_idx, episode in enumerate(dataset.iterate_episodes()):
        if ep_idx >= num_episodes:
            break
        
        # Extract trajectory
        positions = episode.observations['observation'][:, :2]
        goal = episode.observations['desired_goal'][0, :2]
        
        # Convert to grid coordinates
        agent_grid = np.array([maze_handler.world_to_grid_float(pos[0], pos[1]) for pos in positions])
        goal_grid = maze_handler.world_to_grid_float(goal[0], goal[1])
        
        # Create video
        out_path = f"{OUTPUT_DIR}/dataset_videos/episode_{ep_idx + 1}.mp4"
        cfg = GridVideoConfig(fps=10, cell_px=20)
        
        with GridVideoWriter(maze_handler.maze_map, out_path, cfg=cfg) as writer:
            for step_idx, agent_pos in enumerate(agent_grid):
                # agent_pos is (col, row), convert to (row, col) for GridVideoWriter
                agent_rc = (agent_pos[1], agent_pos[0])
                goal_rc = (goal_grid[1], goal_grid[0])
                executed_rc = agent_grid[:step_idx + 1][:, [1, 0]]  # Swap to (row, col)
                
                writer.add_frame(
                    agent_rc=agent_rc,
                    goal_rc=goal_rc,
                    executed_rc=executed_rc,
                    step_idx=step_idx
                )
        
        print(f"✓ Saved: {out_path}")
    
    env.close()


def verify_action_shapes():
    """Verify that all actions in dataset have shape (Horizon, 2)."""
    print("\n" + "=" * 60)
    print("ACTION SHAPE VERIFICATION")
    print("=" * 60)
    
    dataset = MinariDiffusionDataset(DATASET_ID, horizon_T=HORIZON)
    print(f"Dataset created with horizon: {HORIZON}")
    print(f"Total valid windows: {len(dataset)}")
    
    # Check shapes for multiple samples
    print(f"\nSampling {min(10, len(dataset))} windows to verify shapes...")
    for idx in range(min(10, len(dataset))):
        sample = dataset[idx]
        state_shape = sample['state'].shape
        action_shape = sample['action_window'].shape
        print(f"  Sample {idx}: state {state_shape}, actions {action_shape}")
        
        # Verify action shape
        assert action_shape == (HORIZON, 2), f"Expected ({HORIZON}, 2), got {action_shape}"
    
    print(f"✓ All action windows have shape ({HORIZON}, 2)")
    return dataset


def visualize_action_trajectories(num_images: int = 6, horizon_limit: int = None, episode_index: int = 0):
    """Visualize action trajectories from different starting states in a specific episode.
    
    Args:
        num_images: Number of images to generate (different starting states)
        horizon_limit: Maximum number of steps to rollout. If None, uses all available actions.
        episode_index: Zero-based index of the episode to visualize.
    """
    print("\n" + "=" * 60)
    print("ACTION TRAJECTORY VISUALIZATION WITH DYNAMICS")
    print("=" * 60)
    
    # Initialize dynamics model
    dynamics = AnalyticDoubleIntegrator(device=DEVICE)
    
    # Get maze handler and dataset
    minari_dataset = minari.load_dataset(DATASET_ID, download=False)
    env = minari_dataset.recover_environment(eval_env=True, render_mode=None)
    maze_handler = MazeHandler(env, device=DEVICE)
    dynamics.maze_handler = maze_handler
    
    # Select requested episode (fallback to first if out of range)
    selected_episode = None
    for i, ep in enumerate(minari_dataset.iterate_episodes()):
        if i == episode_index:
            selected_episode = ep
            break
    if selected_episode is None:
        selected_episode = next(iter(minari_dataset.iterate_episodes()))

    observations = selected_episode.observations['observation']  # Shape: (T, 6)
    actions = selected_episode.actions  # Shape: (T, 2)
    goal = selected_episode.observations['desired_goal'][0, :2]
    
    print(f"\nSelected Episode: {episode_index + 1}")
    print(f"  Total steps: {len(actions)}")
    print(f"  Action range: [{actions.min():.3f}, {actions.max():.3f}]")
    print(f"  Number of images: {num_images}")
    print(f"  Horizon limit: {horizon_limit if horizon_limit else 'None (use all)'}")
    
    # Setup figure grid
    num_cols = 3
    num_rows = (num_images + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    
    if num_images == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Calculate starting indices for each image
    if num_images > 1:
        step_size = len(actions) // (num_images + 1)
    else:
        step_size = 0
    start_indices = [i * step_size for i in range(num_images)]
    
    # Plot each starting state
    for plot_idx, start_idx in enumerate(start_indices):
        if start_idx >= len(actions):
            break
        
        print(f"\nImage {plot_idx + 1}: Starting from state {start_idx}")
        
        # Get starting state
        initial_state = torch.from_numpy(observations[start_idx]).float()
        
        # Determine how many actions to use
        remaining_actions = len(actions) - start_idx
        if horizon_limit:
            num_steps = min(horizon_limit, remaining_actions)
        else:
            num_steps = remaining_actions
        
        print(f"  Rolling out {num_steps} steps")
        
        # Rollout trajectory using dynamics
        trajectory = [initial_state.numpy().copy()]
        current_state = initial_state.unsqueeze(0).to(DEVICE)  # (1, 6)
        
        for t in range(num_steps):
            action_idx = start_idx + t
            action = torch.from_numpy(actions[action_idx]).float().unsqueeze(0).to(DEVICE)  # (1, 2)
            next_state = dynamics(current_state, action)
            trajectory.append(next_state[0].cpu().detach().numpy())
            current_state = next_state
        
        trajectory = np.array(trajectory)  # (num_steps+1, 6)
        rollout_positions = trajectory[:, :2]  # (num_steps+1, 2)
        actual_positions = observations[start_idx:start_idx + num_steps + 1, :2]
        
        # Plot
        ax = axes[plot_idx]
        
        # Plot actual trajectory from dataset
        ax.plot(actual_positions[:, 0], actual_positions[:, 1], 'b-', linewidth=2, 
                label='Dataset trajectory', alpha=0.7)
        
        # Plot rollout trajectory with gradient color (dark red to bright pink)
        from matplotlib.collections import LineCollection
        from matplotlib.colors import LinearSegmentedColormap
        
        # Create gradient colormap from dark red to bright pink
        cmap = LinearSegmentedColormap.from_list('red_to_pink', 
                                                  ['#8B0000', '#FF1493'])  # Dark red to deep pink
        
        # Create line segments
        points = rollout_positions.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create line collection with gradient colors
        lc = LineCollection(segments, cmap=cmap, linewidth=2.5, alpha=0.8)
        lc.set_array(np.linspace(0, 1, len(segments)))
        line = ax.add_collection(lc)
        
        # Add proxy artist for legend
        ax.plot([], [], color='#8B0000', linewidth=2.5, alpha=0.8, 
                label='Dynamics rollout (red→pink)', linestyle='--')
        
        # Plot start, end, goal
        ax.plot(actual_positions[0, 0], actual_positions[0, 1], 'go', markersize=12, 
                label='Start', zorder=5)
        ax.plot(actual_positions[-1, 0], actual_positions[-1, 1], 'ro', markersize=12, 
                label='End', zorder=5)
        ax.plot(goal[0], goal[1], 'y*', markersize=20, label='Goal', zorder=6)
        
        # Plot action vectors at regular intervals
        arrow_interval = max(1, num_steps // 10)
        arrows_plotted = False
        for t in range(0, num_steps, arrow_interval):
            action_idx = start_idx + t
            action_val = actions[action_idx]
            scale = 0.3
            arrow = ax.arrow(actual_positions[t, 0], actual_positions[t, 1], 
                    action_val[0] * scale, action_val[1] * scale,
                    head_width=0.15, head_length=0.1, fc='purple', ec='purple', 
                    alpha=0.6, linewidth=1.5)
            if not arrows_plotted:
                arrow.set_label('Actions (scaled)')
                arrows_plotted = True
        
        ax.set_title(f'Start state {start_idx} ({num_steps} steps rollout)', 
                    fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.axis('equal')
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
    
    # Hide unused subplots
    for idx in range(plot_idx + 1, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/action_trajectories.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {OUTPUT_DIR}/action_trajectories.png")
    plt.close()
    
    env.close()


def plot_state_dimensions():
    """Visualize individual state dimensions (position and goal)."""
    print("\n" + "=" * 60)
    print("STATE DIMENSION VISUALIZATION")
    print("=" * 60)
    
    dataset = minari.load_dataset(DATASET_ID, download=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    positions_x = []
    positions_y = []
    goals_x = []
    goals_y = []
    
    for episode in dataset.iterate_episodes():
        obs = episode.observations['observation'][:, :2]
        goal = episode.observations['desired_goal'][0, :2]
        
        positions_x.extend(obs[:, 0])
        positions_y.extend(obs[:, 1])
        goals_x.extend([goal[0]] * len(obs))
        goals_y.extend([goal[1]] * len(obs))
    
    # X position
    axes[0, 0].hist(positions_x, bins=40, alpha=0.7, color='blue')
    axes[0, 0].axvline(np.mean(goals_x), color='red', linestyle='--', linewidth=2, label=f'Goal mean: {np.mean(goals_x):.2f}')
    axes[0, 0].set_xlabel('X Position')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Agent X Position Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Y position
    axes[0, 1].hist(positions_y, bins=40, alpha=0.7, color='blue')
    axes[0, 1].axvline(np.mean(goals_y), color='red', linestyle='--', linewidth=2, label=f'Goal mean: {np.mean(goals_y):.2f}')
    axes[0, 1].set_xlabel('Y Position')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Agent Y Position Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Goal X vs Goal Y scatter
    axes[1, 0].scatter(goals_x[:1000], goals_y[:1000], alpha=0.5, s=10)
    axes[1, 0].set_xlabel('Goal X')
    axes[1, 0].set_ylabel('Goal Y')
    axes[1, 0].set_title('Goal Positions (sample)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axis('equal')
    
    # Agent X vs Y scatter (sample)
    axes[1, 1].scatter(positions_x[::10], positions_y[::10], alpha=0.3, s=5, label='Agent positions')
    axes[1, 1].scatter(goals_x[::10], goals_y[::10], alpha=0.5, s=20, color='red', label='Goals')
    axes[1, 1].set_xlabel('X')
    axes[1, 1].set_ylabel('Y')
    axes[1, 1].set_title('Agent vs Goal Positions (sample)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axis('equal')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/state_dimensions.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR}/state_dimensions.png")
    plt.close()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("MINARI DATASET VERIFICATION SCRIPT")
    print("=" * 60)
    
    # Run all verifications
    verify_dataset_structure()
    plot_trajectories_2d(num_episodes=6)
    plot_action_statistics()
    plot_state_dimensions()
    visualize_with_maze(num_episodes=3)
    
    # Verify action shapes and visualize trajectories
    dataset = verify_action_shapes()
    visualize_action_trajectories(num_images=15, horizon_limit=HORIZON, episode_index=5)
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
    print("\nGenerated files in dataset_verify/:")
    print("  - dataset_trajectories.png: 2D trajectories from episodes")
    print("  - action_statistics.png: Action value distributions")
    print("  - state_dimensions.png: State dimension analysis")
    print("  - action_trajectories.png: Rollouts using dynamics model")
    print("  - dataset_videos/: Maze-overlaid trajectory videos")
