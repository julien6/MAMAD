# jax_evaluate.py

import numpy as np
import jax
import jax.numpy as jnp
import pickle
import json
import matplotlib.pyplot as plt

from typing import Dict, List
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
from prepare_data import infer_metadata_from_json, one_hot_encode_action
from flax import linen as nn
from jax_train import SimpleLSTM
from collect import load_episode_step_data


class ODF_LSTM_Runner_JAX:
    def __init__(self, model_path: str, trajectories_path: str):

        with open(model_path, 'rb') as f:
            checkpoint = pickle.load(f)

        self.params = checkpoint["params"]
        self.model_hyperparams = checkpoint["model_hyperparams"]
        self.metadata = infer_metadata_from_json(trajectories_path)

        self.model = SimpleLSTM(
            hidden_size=self.model_hyperparams["hidden_size"],
            output_size=self.metadata["obs_dim_per_agent"] *
            len(self.metadata["agent_order"]),
            num_layers=self.model_hyperparams["num_layers"]
        )
        self.rng = jax.random.PRNGKey(0)  # Dummy RNG

    def predict_next_obs(self, joint_obs: Dict[str, List[float]], joint_action: Dict[str, int]) -> Dict[str, List[float]]:
        obs_vec = np.array([])
        act_vec = np.array([])

        for agent in self.metadata["agent_order"]:
            obs_vec = np.concatenate([obs_vec, joint_obs[agent]])
            if agent in joint_action:
                act_vec = np.concatenate([act_vec, one_hot_encode_action(
                    joint_action[agent], self.metadata["num_actions"]
                )])
            else:
                # If missing, assume a "no-op" action (all zeros)
                act_vec = np.concatenate(
                    [act_vec, np.array([0.0] * self.metadata["num_actions"])])

        input_vec = np.concatenate([obs_vec, act_vec])
        input_vec = jnp.array(input_vec)
        # (batch_size=1, seq_len=1, input_dim)
        input_vec = input_vec[None, None, :]

        pred = self.model.apply({'params': self.params}, input_vec)
        pred = np.array(pred)[0]  # remove batch dimension

        # Split predicted vector back into agents
        obs_dim = self.metadata["obs_dim_per_agent"]
        agent_order = self.metadata["agent_order"]
        result = {}
        idx = 0
        for agent in agent_order:
            result[agent] = pred[idx:idx + obs_dim].tolist()
            idx += obs_dim

        return result


def transition_to_vector(joint_obs: Dict[str, List[float]],
                         joint_action: Dict[str, int],
                         next_joint_obs: Dict[str, List[float]],
                         agent_order: List[str],
                         num_actions: int) -> np.ndarray:
    obs = np.concatenate([np.array(joint_obs[agent]) for agent in agent_order])

    act = np.array([])
    for agent in agent_order:
        if agent in joint_action:
            act = np.concatenate([act, one_hot_encode_action(
                joint_action[agent], num_actions)])
        else:
            act = np.concatenate([act, np.array([0.0] * num_actions)])
    act = np.array(act)

    next_obs = np.concatenate([np.array(next_joint_obs[agent])
                              for agent in agent_order])

    return np.concatenate([obs, act, next_obs])


def generate_exact_and_predicted_trajectories(file_path: str, ep_idx: int,
                                              odf_runner: ODF_LSTM_Runner_JAX,
                                              max_step: int,
                                              agent_order: List[str],
                                              num_actions: int):
    exact_trajectory = []
    predicted_trajectory = []

    # Initial step
    step = load_episode_step_data(file_path, ep_idx, 0)
    if step is None:
        raise ValueError("Unable to load initial step.")

    obs = step["joint_observation"]
    act = step["joint_action"]

    for t in range(max_step - 1):
        next_step = load_episode_step_data(file_path, ep_idx, t + 1)
        if next_step is None:
            break
        true_next_obs = next_step["joint_observation"]

        # Exact
        exact_vec = transition_to_vector(
            obs, act, true_next_obs, agent_order, num_actions)
        exact_trajectory.append(exact_vec)

        # Predicted
        predicted_next_obs = odf_runner.predict_next_obs(obs, act)
        predicted_vec = transition_to_vector(
            obs, act, predicted_next_obs, agent_order, num_actions)
        predicted_trajectory.append(predicted_vec)

        # Update
        obs = predicted_next_obs
        act = next_step["joint_action"]

    return np.array(exact_trajectory), np.array(predicted_trajectory)


def plot_transitions_PCA(exact_traj, predicted_traj, ep_idx, save_path="transitions_PCA.png"):
    X = np.vstack([exact_traj, predicted_traj])
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    n_obs = len(exact_traj)

    plt.figure(figsize=(12, 7))
    plt.plot(X_2d[:n_obs, 0], X_2d[:n_obs, 1], '-o',
             color='blue', label="Exact trajectory")
    for i in range(n_obs):
        plt.text(X_2d[i, 0], X_2d[i, 1], str(i), fontsize=12, color='blue')

    plt.plot(X_2d[n_obs:, 0], X_2d[n_obs:, 1], '-o',
             color='orange', label="Predicted trajectory")
    for i in range(n_obs):
        plt.text(X_2d[n_obs + i, 0], X_2d[n_obs + i, 1],
                 str(i), fontsize=12, color='orange')

    plt.title(f"Transitions PCA with Step Index (episode {ep_idx})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='Exact obs'),
        Line2D([0], [0], color='orange', lw=2, label='Predicted obs')
    ]
    plt.legend(handles=legend_elements)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_observations_PCA(exact_traj, predicted_traj, obs_dim: int, agent_count: int, num_actions: int, ep_idx, save_path="observations_PCA.png"):
    def extract_obs_vectors(traj):
        obs_size = obs_dim * agent_count
        act_size = num_actions * agent_count
        obs_vectors = []
        obs = None
        next_obs = None
        for row in traj:
            if obs is None and next_obs is None:
                obs = row[:obs_size]
                obs_vectors.append(obs)
            next_obs = row[obs_size + act_size:]
            obs_vectors.append(next_obs)
        return np.array(obs_vectors)

    exact_obs = extract_obs_vectors(exact_traj)
    pred_obs = extract_obs_vectors(predicted_traj)

    X = np.vstack([exact_obs, pred_obs])
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    n_obs = len(exact_obs)

    plt.figure(figsize=(14, 8))
    for i in range(n_obs):
        plt.scatter(X_2d[i, 0], X_2d[i, 1], color='blue')
        plt.text(X_2d[i, 0], X_2d[i, 1], str(i), fontsize=10, color='blue')
    for i in range(len(pred_obs)):
        j = n_obs + i
        plt.scatter(X_2d[j, 0], X_2d[j, 1], color='orange')
        plt.text(X_2d[j, 0], X_2d[j, 1], str(i), fontsize=10, color='orange')

    for i in range(0, len(exact_obs) - 1):
        plt.plot(X_2d[i:i+2, 0], X_2d[i:i+2, 1],
                 color='blue', linestyle='--', linewidth=0.5)
    for i in range(n_obs, len(X_2d) - 1):
        plt.plot(X_2d[i:i+2, 0], X_2d[i:i+2, 1],
                 color='orange', linestyle='--', linewidth=0.5)

    plt.title(f"Observations PCA with Step Index (episode {ep_idx})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='Exact obs'),
        Line2D([0], [0], color='orange', lw=2, label='Predicted obs')
    ]
    plt.legend(handles=legend_elements)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


if __name__ == '__main__':
    model_path = "trained_model.pkl"
    trajectories_path = "trajectories.json"
    runner = ODF_LSTM_Runner_JAX(model_path, trajectories_path)
    ep_idx = 0
    print(f"▶️  Selected trajectory: episode {ep_idx}")

    exact_traj, pred_traj = generate_exact_and_predicted_trajectories(
        file_path=trajectories_path,
        ep_idx=ep_idx,
        odf_runner=runner,
        max_step=runner.metadata["max_step"],
        agent_order=runner.metadata["agent_order"],
        num_actions=runner.metadata["num_actions"]
    )

    plot_transitions_PCA(exact_traj, pred_traj, ep_idx)
    plot_observations_PCA(
        exact_traj, pred_traj,
        runner.metadata["obs_dim_per_agent"],
        len(runner.metadata["agent_order"]),
        runner.metadata["num_actions"],
        ep_idx
    )
