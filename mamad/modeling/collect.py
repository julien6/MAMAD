import gym
import numpy as np
import pygame
import pickle
import os
import random
import os
import gym
import random
import tensorflow as tf
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from gym.spaces import Dict as GymDict, Discrete, Box
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import Agent, RandomAgent
from tqdm import trange
from typing import Any, Dict, List, Tuple, Union
from pettingzoo.butterfly import pistonball_v4
from PIL import Image
from typing import Dict, List
from typing import Dict
from tqdm import tqdm
from torch.nn.functional import cosine_similarity
from skimage.metrics import structural_similarity as ssim
from math import log10
from torch.utils.data import DataLoader, TensorDataset
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from omarle.warehouse_management.warehouse_management_v0 import parallel_env as wm_env


Trajectory = Dict[str, Dict[str, Dict[str, List]]]

Trajectories = Dict[str, Trajectory]

# Example of collected trajectories saved in the JSON file
#
# {
# "episode_0":{
# "step_0": {"joint_observation": {"agent_0": [0,...], "agent_1": [0,...]}, "joint_action": {"agent_0": [0,...], "agent_1": [0,...]}},
# "step_1": {"joint_observation": {"agent_0": [0,...], "agent_1": [0,...]}, "joint_action": {"agent_0": [0,...], "agent_1": [0,...]}},
# "step_2": {"joint_observation": {"agent_0": [0,...], "agent_1": [0,...]}, "joint_action": {"agent_0": [0,...], "agent_1": [0,...]}},
# ...
# },
# "episode_1":{
# "step_0": {"joint_observation": {"agent_0": [0,...], "agent_1": [0,...]}, "joint_action": {"agent_0": [0,...], "agent_1": [0,...]}},
# "step_1": {"joint_observation": {"agent_0": [0,...], "agent_1": [0,...]}, "joint_action": {"agent_0": [0,...], "agent_1": [0,...]}},
# "step_2": {"joint_observation": {"agent_0": [0,...], "agent_1": [0,...]}, "joint_action": {"agent_0": [0,...], "agent_1": [0,...]}},
# ...
# },
# ...
# }


class ActionSampler:

    def sample(self, observation: Any) -> Union[Dict, List, np.ndarray]:
        raise NotImplementedError()


def collect_trajectories(env, action_sampler: ActionSampler = None, num_episodes=100, max_steps=1000, file_path="trajectories.json", max_categorical_observation: int = None):
    """
    Collects and saves trajectories from the Pistonball environment in an incremental JSON file.
    A trajectory is a list of pairs (observations, joint_action) where each element
    is a dictionary {agent: List} that, for each agent, associates an observation/action
    flattened into a list.

    Parameters:
        env (gym.Env): Gym environment
        action_sampler (ActionSampler): Action sampler
        num_episodes (int): Number of episodes to simulate.
        max_steps (int): Maximum number of steps per episode.
        file_path (str): Path to the JSON file where the trajectories are stored.

    Returns:
        str: The path to the JSON file containing the collected trajectories, respecting the "Trajectories" format.
    """

    # Adapt to the directory name of the script
    file_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), file_path)
    if os.path.exists(file_path):
        os.remove(file_path)

    # Create the file
    f = open(file_path, 'w+')
    f.close()

    with open(file_path, 'a') as f:
        f.write('{\n')  # Start the JSON file
        for episode in tqdm(range(num_episodes)):

            joint_obs: Dict[str, List] = env.reset()

            f.write('"episode_' + str(episode) + '":{\n')

            for step in tqdm(range(max_steps)):
                joint_action = None
                if action_sampler is None:
                    joint_action = {agent: env.action_space(
                        agent).sample() for agent in env.agents}
                else:
                    joint_action = action_sampler.sample(joint_obs)

                next_joint_obs, rewards, dones, infos = env.step(joint_action)

                if not isinstance(joint_action, Dict):
                    agents = [f'agent_{i}'.format(i)
                              for i in range(0, len(joint_action))]
                    joint_obs = {agent: joint_obs["both_agent_obs"][index].tolist(
                    ) for index, agent in enumerate(agents)}
                    joint_action = {agent: joint_action[index]
                                    for index, agent in enumerate(agents)}

                if type(list(joint_obs.values())[0]) == np.ndarray:
                    joint_obs = {agent: obs.tolist()
                                 for agent, obs in joint_obs.items()}

                if max_categorical_observation is not None:
                    joint_obs = {
                        agent: np.concatenate([np.eye(max_categorical_observation, max_categorical_observation)[cell] for cell in obs]).tolist() for agent, obs in joint_obs.items()
                    }

                trajectory_step = {
                    "joint_observation": joint_obs, "joint_action": joint_action}

                # Écrire la trajectoire de l'épisode dans le fichier
                f.write(f'"step_{str(step)}": {json.dumps(trajectory_step)}')
                if (step < max_steps - 1) and not list(dones.values())[0]:
                    f.write(',\n')  # Séparer chaque step par une virgule

                joint_obs = next_joint_obs

                if list(dones.values())[0]:
                    break

            if episode < num_episodes - 1:
                f.write('},\n')  # Séparer chaque épisode par une virgule
            print(f"Episode {episode} completed.")

        f.write('}\n}')  # Fin du fichier JSON

    return file_path


def load_episode_step_data(file_path: str, episode_idx: int, step_idx: int):
    """
    Loads the trajectory of a given episode, extracting the joint_action and observations for a specific step.

    Parameters:
        file_path (str): Path to the JSON file containing the trajectories.
        episode_idx (int): The index of the episode to load.
        step_idx (int): The step number of the episode for which the observations will be concatenated.
    """
    # Variables pour suivre l'épisode et le pas actuels
    current_episode = -1
    found_episode = False
    step_data = None

    with open(file_path, 'r') as f:
        # Aller à l'ouverture de la liste des épisodes dans le JSON
        f.seek(1)  # Skip the first "[" character

        # Lecture ligne par ligne de chaque épisode
        for line in f:
            if not found_episode and f"episode_{episode_idx}" not in line:
                continue
            found_episode = True

            if found_episode:
                if f"step_{step_idx}" not in line:
                    continue
                step_data = line
                break

    if step_data is None:
        print(
            f"The trajectory of episode {episode_idx} for step {step_idx} was not found in the file.")
        return None

    step_data = step_data.replace(f'"step_{step_idx}": ', "")
    if step_data[-4:] == "}}}\n":
        step_data = step_data[:-2]
    if step_data[-5:] == "}}},\n":
        step_data = step_data[:-3]
    if step_data[-5:] == "]}}}\n":
        step_data = step_data[:-2]
    if step_data[-5:] == "]}},\n":
        step_data = step_data[:-2]
    if step_data[-2:] == ",\n":
        step_data = step_data[:-2]
    step_data = json.loads(step_data)
    return step_data


if __name__ == '__main__':

    class OvercookedActionSampler(ActionSampler):

        def __init__(self) -> None:
            super().__init__()

        def sample(self, observations: Any):
            return [random.randint(0, 5), random.randint(0, 5)]

    class WMActionSampler(ActionSampler):

        def __init__(self) -> None:
            super().__init__()

        def sample(self, observations: Any):
            return {"agent_0": random.randint(0, 6), "agent_1": random.randint(0, 6), "agent_2": random.randint(0, 6)}

    # Step 1.1 - Initialize the Overcooked-AI environment
    # layout_mdp = OvercookedGridworld.from_layout_name("asymmetric_advantages")
    # core_env = OvercookedEnv.from_mdp(layout_mdp, horizon=400)
    # config = {'base_env': core_env,
    #           'featurize_fn': core_env.featurize_state_mdp}
    # env = gym.make('Overcooked-v0', **config)

    # Step 1.1 - Initialize the Warehouse Management environment
    env = wm_env()

    output_file_path = collect_trajectories(
        env, num_episodes=1000, action_sampler=WMActionSampler())  # , max_categorical_observation=14)

    # output_file_path = "trajectories.json"

    from prepare_data import infer_metadata_from_json

    print(infer_metadata_from_json(output_file_path))
