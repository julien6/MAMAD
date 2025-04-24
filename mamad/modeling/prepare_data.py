import pickle
import os
import torch.nn as nn
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from collect import load_episode_step_data
from typing import Dict


def infer_metadata_from_json(file_path: str, max_episodes_to_check: int = 20) -> Dict:
    """
    Infers agent order, number of actions, and observation dimensions automatically
    by sampling early steps from episodes using load_episode_step_data().
    """

    # Étape 1 : trouver un step valide
    episode_idx = 0
    step_data = load_episode_step_data(file_path, episode_idx, 0)

    if step_data is None:
        raise ValueError("Impossible de charger la step 0 de l'épisode 0.")

    # Étape 2 : extraire l'ordre des agents
    agent_order = list(step_data["joint_action"].keys())

    # Étape 3 : calculer la taille d'observation par agent
    sample_obs = step_data["joint_observation"][agent_order[0]]
    obs_dim_per_agent = len(sample_obs)

    # Étape 4 : parcourir plusieurs steps pour détecter les bornes min/max des actions
    min_action = float("inf")
    max_action = float("-inf")

    max_step = None
    max_episode = None
    old_step_idx = None

    for ep in range(max_episodes_to_check):
        step_idx = 0
        step = None
        while True:
            step = load_episode_step_data(file_path, ep, step_idx)

            if step_idx == 0 and max_step is None and old_step_idx is not None:
                max_step = old_step_idx

            if step is None:
                break
            actions = step["joint_action"]
            for agent in agent_order:
                act = actions[agent]
                if isinstance(act, int):
                    min_action = min(min_action, act)
                    max_action = max(max_action, act)
            step_idx += 1
            if max_step is None:
                old_step_idx = step_idx
            elif step_idx >= max_step:
                break

        if step is None and step_idx == 0:
            if max_episode is None:
                max_episode = ep
            break

    num_actions = max_action - min_action + 1

    return {
        "agent_order": agent_order,
        "obs_dim_per_agent": obs_dim_per_agent,
        "num_actions": num_actions,
        "min_action": min_action,
        "max_action": max_action,
        "max_step": max_step,
        "max_episode": max_episode
    }


def one_hot_encode_action(action_idx: int, num_actions: int = 6) -> np.ndarray:
    """One-hot encode a single action index."""
    vec = np.zeros(num_actions)
    vec[action_idx] = 1.0
    return vec


def flatten_joint_observation(joint_obs: dict, agent_order: List[str]) -> np.ndarray:
    """Concatenate agent observations into a single vector."""
    return np.concatenate([np.array(joint_obs[agent]) for agent in agent_order])


def flatten_joint_action(joint_action: dict, agent_order: List[str], num_actions: int) -> np.ndarray:
    """Concatenate one-hot encoded agent actions."""
    return np.concatenate([
        one_hot_encode_action(joint_action[agent], num_actions)
        for agent in agent_order
    ])


def load_episode_as_sequence_lazy(file_path: str, episode_idx: int,
                                  agent_order: List[str], num_actions: int, nb_step=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a full episode as a sequence using lazy loading from disk, step by step.
    Returns input sequence and target sequence for LSTM training.
    """
    input_seq = []
    target_seq = []

    step = None
    next_step = None
    i = 0
    while True:

        if step is None and next_step is None:
            step = load_episode_step_data(file_path, episode_idx, i)
        else:
            step = next_step

        next_step = load_episode_step_data(file_path, episode_idx, i + 1)

        if step is None or next_step is None:
            break

        obs = flatten_joint_observation(step["joint_observation"], agent_order)
        act = flatten_joint_action(
            step["joint_action"], agent_order, num_actions)
        next_obs = flatten_joint_observation(
            next_step["joint_observation"], agent_order)

        x = np.concatenate([obs, act])
        y = next_obs

        input_seq.append(x)
        target_seq.append(y)
        i += 1

        if nb_step is not None and i + 1 >= nb_step:
            break

    return np.stack(input_seq), np.stack(target_seq)


class OvercookedSequenceLSTMDatasetLazy(torch.utils.data.Dataset):
    """
    Dataset that loads episodes lazily one-by-one and returns full sequences for LSTM training.
    """

    def __init__(self, file_path: str, episode_indices: List[int],
                 agent_order: List[str], num_actions: int, nb_step: int = None):
        self.file_path = file_path
        self.episode_indices = episode_indices
        self.agent_order = agent_order
        self.num_actions = num_actions
        self.nb_step = nb_step

    def __len__(self):
        return len(self.episode_indices)

    def __getitem__(self, idx):
        ep_idx = self.episode_indices[idx]
        x_seq, y_seq = load_episode_as_sequence_lazy(
            self.file_path,
            ep_idx,
            self.agent_order,
            self.num_actions,
            self.nb_step)
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_seq, dtype=torch.float32)


if __name__ == "__main__":
    metadata = infer_metadata_from_json("trajectories.json")
    print("Agent order:", metadata["agent_order"])
    print("Observation size per agent:", metadata["obs_dim_per_agent"])
    print("Number of actions:", metadata["num_actions"])
    print("Action range:", metadata["min_action"],
          "to", metadata["max_action"])
    print("Episode number:", metadata["max_episode"])
    print("Step number:", metadata["max_step"])

    # === TEST : Chargement d'une trajectoire pour entraînement LSTM ===
    print("\n=== Loading a trajectory ===")
    episode_idx = 0
    x_seq, y_seq = load_episode_as_sequence_lazy(
        file_path="trajectories.json",
        episode_idx=episode_idx,
        agent_order=metadata["agent_order"],
        num_actions=metadata["num_actions"],
        nb_step=metadata["max_step"]
    )

    # (nb_step, obs_dim*n_agents + action_dim*n_agents)
    print(f"x_seq shape (input to LSTM): {x_seq.shape}")
    # (nb_step, obs_dim*n_agents)
    print(f"y_seq shape (target): {y_seq.shape}")

    # Exemple : affichage premier pas
    print("\n--- Exemple t=0 ---")
    print("x[0][:10] =", x_seq[0][:10])
    print("y[0][:10] =", y_seq[0][:10])
