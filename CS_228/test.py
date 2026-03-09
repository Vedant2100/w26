import gymnasium as gym
import minigrid
import pandas as pd
import time
from pathlib import Path

import argparse

# Paths to the experiment data
BASE_DIR = Path("CS 228/outputs/20260309_063547_bot_sweep")
EPISODES_PATH = BASE_DIR / "episodes.csv"
ACTIONS_PATH = BASE_DIR / "actions.csv"

# Set up argument parsing
parser = argparse.ArgumentParser(description="Replay successful episodes.")
parser.add_argument("--env-type", type=str, choices=["lava", "doorkey", "empty"], 
                    help="Filter by environment type (lava, doorkey, empty)")
args = parser.parse_args()

# Load data
df_episodes = pd.read_csv(EPISODES_PATH)
df_actions = pd.read_csv(ACTIONS_PATH)

# Filter for successful episodes
successes = df_episodes[df_episodes['success'] == True].copy()

# Apply environment type filter if provided
if args.env_type:
    successes = successes[successes['env'].str.contains(args.env_type, case=False)]


# Action mapping for MiniGrid (parsing internal strings back to ints)
ACTION_MAP = {
    "forward": 2,
    "turn_left": 0,
    "turn_right": 1,
    "pickup": 3,
    "drop": 4,
    "toggle": 5,
    "done": 6
}

def replay_episode(row):
    env_id = row['env']
    episode_idx = int(row['episode'])
    model = row['model']
    config = f"Buf={row['buffer_size']}, K={row['history_window']}"
    
    # Identify the steps for this specific episode
    # Note: We filter by episode, model, env, buffer_size, and history_window to ensure uniqueness
    steps = df_actions[
        (df_actions['env'] == env_id) & 
        (df_actions['episode'] == episode_idx) & 
        (df_actions['model'] == model) &
        (df_actions['buffer_size'] == row['buffer_size']) &
        (df_actions['history_window'] == row['history_window'])
    ]
    
    print(f"\n--- Replaying SUCCESS: {env_id} ---")
    print(f"Model: {model} | Config: {config} | Episode: {episode_idx}")
    
    # Create env and reset with seed (42 + i)
    env = gym.make(env_id, render_mode="human")
    obs, _ = env.reset(seed=42 + episode_idx)
    env.render()
    time.sleep(0.5) # initial pause
    
    for _, step in steps.iterrows():
        action_str = step['parsed_action']
        action_int = ACTION_MAP.get(action_str, 2) # default to forward if unknown
        
        obs, rew, terminated, truncated, info = env.step(action_int)
        env.render()
        time.sleep(0.1) # speed up playback for batching
        
        if terminated or truncated:
            print(f"DONE. Reward: {rew}")
            time.sleep(0.5)
            break
            
    env.close()

# Iterate through all successes and play them
if successes.empty:
    print("No successful episodes found to replay.")
else:
    print(f"Found {len(successes)} successful episodes. Starting gallery replay...")
    for _, row in successes.iterrows():
        replay_episode(row)
