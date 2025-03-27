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

# Une trajectoire ("Trajectory") est une liste de couples (observations, actions) où chaque élément est un dictionnaire {agent: List}
Trajectory = Dict[str, Dict[str, Dict[str, List]]]

Trajectories = Dict[str, Trajectory]

# Exemple de trajectoires collectées dans le fichier JSON
#
# {
# "episode_0":{
# "step_0": {"frame": [58,...], "actions": {"agent_0": [0,...], "agent_1": [0,...]}},
# "step_1": {"frame": [58,...], "actions": {"agent_0": [0,...], "agent_1": [0,...]}},
# "step_2": {"frame": [58,...], "actions": {"agent_0": [0,...], "agent_1": [0,...]}},
# ...
# },
# "episode_1":{
# "step_0": {"frame": [58,...], "actions": {"agent_0": [0,...], "agent_1": [0,...]}},
# "step_1": {"frame": [58,...], "actions": {"agent_0": [0,...], "agent_1": [0,...]}},
# "step_2": {"frame": [58,...], "actions": {"agent_0": [0,...], "agent_1": [0,...]}},
# ...
# },
# ...
# }


class ActionSampler:

    def sample(self, observation: Any) -> Union[Dict, List, np.ndarray]:
        raise NotImplementedError()


def collect_trajectories(env, action_sampler: ActionSampler = None, num_episodes=100, max_steps=100, file_path="trajectories.json"):
    """
    Collecte et enregistre les trajectoires de l'environnement Pistonball dans un fichier JSON incrémental.
    Une trajectoire est une liste de couples (observations, actions) où chaque élément
    est un dictionnaire {agent: List} qui pour chaque agent associe une observation/action
    applatie sous forme de liste.

    Parameters:
        env (gym.Env): Environnement Gym
        action_sampler (ActionSampler): Echantilloneur d'action
        num_episodes (int): Nombre d'épisodes à simuler.
        max_steps (int): Nombre maximal de pas par épisode.
        file_path (str): Chemin du fichier JSON où stocker les trajectoires.

    Returns:
        str: Le chemin vers le fichier JSON contenant les trajectoires collectées respctant le format "Trajectories".
    """

    # Ensure RGB rendering is supported
    env.reset()
    env.render(mode="rgb_array")

    # Adapt to the directory name of the script
    file_path = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), file_path)
    # Supprime le fichier existant si nécessaire
    if os.path.exists(file_path):
        os.remove(file_path)

    with open(file_path, 'a') as f:
        f.write('{\n')  # Début du fichier JSON (liste de trajectoires)
        for episode in tqdm(range(num_episodes)):

            obs: Dict[str, List] = env.reset()
            frame = env.render(mode="rgb_array")

            f.write('"episode_' + str(episode) + '":{\n')

            for step in tqdm(range(max_steps)):
                actions = None
                if action_sampler is None:
                    actions = {agent: env.action_space(
                        agent).sample() for agent in env.agents}
                else:
                    actions = action_sampler.sample(obs)
                obs, rewards, dones, infos = env.step(actions)

                if not isinstance(actions, Dict):
                    agents = [f'agent_{i}'.format(i)
                              for i in range(0, len(actions))]
                    actions = {agent: actions[i]
                               for i, agent in enumerate(agents)}

                trajectory_step = {
                    "frame": frame.tolist(), "actions": actions}

                frame = env.render(mode="rgb_array")

                # Écrire la trajectoire de l'épisode dans le fichier
                f.write(f'"step_{str(step)}": {json.dumps(trajectory_step)}')
                if step < max_steps - 1:
                    f.write(',\n')  # Séparer chaque step par une virgule

            if episode < num_episodes - 1:
                f.write('},\n')  # Séparer chaque épisode par une virgule
            print(f"Episode {episode} completed.")

        f.write('}\n}')  # Fin du fichier JSON

    return file_path


def load_episode_step_data(file_path: str, episode_idx: int, step_idx: int):
    """
    Charge la trajectoire d'un épisode donné, extrait les actions et observations d'un pas spécifique.

    Parameters:
        file_path (str): Chemin du fichier JSON contenant les trajectoires.
        episode_idx (int): L'indice de l'épisode à charger.
        step_idx (int): Le numéro du pas de l'épisode pour lequel les observations seront concaténées.
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
            f"La trajectoire de l'épisode {episode_idx} pour l'étape {step_idx} n'a pas été trouvé dans le fichier.")
        return

    step_data = step_data.replace(f'"step_{step_idx}": ', "")
    if step_data[-5:] == "}}},\n":
        step_data = step_data[:-3]
    if step_data[-5:] == "]}}}\n":
        step_data = step_data[:-2]
    if step_data[-5:] == "]}},\n":
        step_data = step_data[:-2]

    print(step_data)

    step_data = json.loads(step_data)
    return step_data


if __name__ == '__main__':

    class OvercookedActionSampler(ActionSampler):

        def __init__(self) -> None:
            super().__init__()

        def sample(self, observations: Any):
            return [random.randint(0, 5), random.randint(0, 5)]

    # Step 1.1 - Initialize the Overcooked-AI environment
    layout_mdp = OvercookedGridworld.from_layout_name("asymmetric_advantages")
    core_env = OvercookedEnv.from_mdp(layout_mdp, horizon=200)
    config = {'base_env': core_env,
              'featurize_fn': core_env.featurize_state_mdp}
    env = gym.make('Overcooked-v0', **config)

    # output_file_path = collect_trajectories(
    #     env, num_episodes=1, max_steps=30, action_sampler=OvercookedActionSampler())

    output_file_path = "trajectories.json"

    print(load_episode_step_data(output_file_path, 0, 0))
