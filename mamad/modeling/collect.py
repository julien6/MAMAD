import gym
import numpy as np
import pygame
import pickle
import os
import random

from gym.spaces import Dict as GymDict, Discrete, Box
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import Agent, RandomAgent
from tqdm import trange

# Step 1.1 - Environment and wrapper

# Choosing the layout
layout_mdp = OvercookedGridworld.from_layout_name("asymmetric_advantages")
core_env = OvercookedEnv.from_mdp(layout_mdp, horizon=200)
config_dict = {'base_env': core_env,
               'featurize_fn': core_env.featurize_state_mdp}

# Creating the environment
env = gym.make('Overcooked-v0', **config_dict)

# Creating two random agents
agent1 = RandomAgent(all_actions=True)
agent2 = RandomAgent(all_actions=True)

# Number episodes to collect
nb_episodes = 100
episode_length = 200

# Data storage folder
output_dir = "./overcooked_data"
os.makedirs(output_dir, exist_ok=True)

agents = [f'agent_{i}' for i in range(0,env.env.env.agent_idx + 1)]

# Step 1.2 - Collect observations
def collect_data(env, agent1, agent2, nb_episodes=100, episode_length=200, save_every=10, output_dir="./overcooked_data"):
    all_data = []

    for episode in trange(nb_episodes):
        obs = env.reset()
        traj = []

        for t in range(episode_length):
            # Get joint action
            action1 = random.randint(0,5)
            action2 = random.randint(0,5)
            joint_action = [action1, action2]
            
            obs_next, reward, done, info = env.step(joint_action)
            traj.append({
                "observations": obs["both_agent_obs"],
                "actions": joint_action,
                "next_observations": obs_next["both_agent_obs"]
            })

            obs = obs_next

            if done:
                break

        all_data.extend(traj)

        if (episode + 1) % save_every == 0:
            file_path = os.path.join(output_dir, f"data_ep{episode + 1}.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(traj, f)
            print(f"Saved: {file_path}")

    return all_data


# Step 1.3 - Launching the collect
data = collect_data(env, agent1, agent2, nb_episodes=100, episode_length=200)
