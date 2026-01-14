import minari
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv
from config import DATASET_ID, HORIZON


class MinariDiffusionDataset(Dataset):
    def __init__(self, dataset_id, horizon_T):
        """
        Args:
            dataset_id (str): Minari dataset name (e.g., 'D4RL/pointmaze/large-v2')
            horizon_T (int): The prediction horizon for actions
        """
        self.horizon = horizon_T
        
        # 1. Load the Minari Dataset
        self.minari_dataset = minari.load_dataset(dataset_id)
        
        # 2. Build an index of valid windows
        # We need a list of tuples: (episode_index, start_timestep)
        self.indices = []
        
        # We cache the data in RAM for speed (PointMaze is small enough)
        self.episode_data = []
        
        print("Pre-loading episodes and building indices...")
        for ep_idx, episode in enumerate(self.minari_dataset.iterate_episodes()):
            # Episode lengths
            n_steps = len(episode.actions)
            
            # Skip episodes that are too short for the horizon
            if n_steps < self.horizon:
                continue
                
            # Store data. Note: Minari observations are (N+1), actions are (N)
            # We assume you want to condition on Goal + Observation
            # Flattening dict: we concat 'observation' and 'desired_goal'
            obs = episode.observations['observation'][:-1]  # Remove last obs to match actions
            goal = episode.observations['desired_goal'][:-1]
            
            # Concatenate state for easier conditioning: (N, obs_dim + goal_dim)
            combined_state = np.concatenate([obs, goal], axis=-1)
            
            self.episode_data.append({
                'state': combined_state,
                'actions': episode.actions
            })
            
            # Calculate valid start indices for this episode
            # If length is L and horizon is T, last start index is L-T
            for t in range(n_steps - self.horizon + 1):
                self.indices.append((len(self.episode_data) - 1, t))
                
        print(f"Dataset ready. Found {len(self.indices)} valid windows.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep_idx, start_t = self.indices[idx]
        data = self.episode_data[ep_idx]
        
        end_t = start_t + self.horizon
        
        # 1. Get the conditioning state at time t (current state)
        state_at_t = data['state'][start_t]
        
        # 2. Get the Horizon of T actions
        action_window = data['actions'][start_t : end_t]
        
        # Convert to Torch Tensors
        return {
            'state': torch.from_numpy(state_at_t).float(),
            'action_window': torch.from_numpy(action_window).float()  # Shape: (T, action_dim)
        }


if __name__ == "__main__":
    # --- Usage Example ---
    
    # Define your Horizon T (e.g., predict next 16 steps)
    T = HORIZON
    
    # Initialize with DATASET_ID from config
    dataset = MinariDiffusionDataset(DATASET_ID, horizon_T=T)
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # Test a batch
    batch = next(iter(dataloader))
    print(f"State Batch Shape: {batch['state'].shape}")           # (256, 6) -> 4 obs + 2 goal
    print(f"Action Window Shape: {batch['action_window'].shape}") # (256, 16, 2) -> 16 steps, 2 dims
