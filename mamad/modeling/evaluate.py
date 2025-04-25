# Re-import required modules after code execution environment reset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random
import torch

from typing import Dict, List
from collect import load_episode_step_data
from prepare_data import one_hot_encode_action, infer_metadata_from_json
from train import ODF_LSTM_runner
from matplotlib.lines import Line2D


def transition_to_vector(joint_obs: Dict[str, List[float]],
                         joint_action: Dict[str, int],
                         next_joint_obs: Dict[str, List[float]],
                         agent_order: List[str],
                         num_actions: int) -> np.ndarray:
    obs = np.concatenate([np.array(joint_obs[agent]) for agent in agent_order])
    act = np.concatenate([one_hot_encode_action(
        joint_action[agent], num_actions) for agent in agent_order])
    next_obs = np.concatenate([np.array(next_joint_obs[agent])
                              for agent in agent_order])
    return np.concatenate([obs, act, next_obs])


def generate_exact_and_predicted_trajectories(file_path: str, ep_idx: int,
                                              odf_runner: ODF_LSTM_runner,
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
        predicted_next_obs = odf_runner.run_ODF_LSTM(obs, act)
        predicted_vec = transition_to_vector(
            obs, act, predicted_next_obs, agent_order, num_actions)
        predicted_trajectory.append(predicted_vec)

        # Update for next iteration
        obs = predicted_next_obs
        act = next_step["joint_action"]

    return np.array(exact_trajectory), np.array(predicted_trajectory)


def plot_transitions_PCA(exact_traj, predicted_traj, ep_idx, save_path="transitions_PCA.png"):
    X = np.vstack([exact_traj, predicted_traj])
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    n_obs = len(exact_traj)

    plt.figure(figsize=(12, 7))
    # Exact
    plt.plot(X_2d[:n_obs, 0], X_2d[:n_obs, 1], '-o',
             color='blue', label="Exact trajectory")
    for i in range(n_obs):
        plt.text(X_2d[i, 0], X_2d[i, 1], str(i), fontsize=12, color='blue')

    # Predicted
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

    def extract_obs_vectors_from_trajectory(traj: np.ndarray, obs_dim: int, agent_count: int, num_actions: int) -> List[np.ndarray]:
        """
        Given a trajectory (N transitions), returns 2N observation vectors:
        the initial and next observation for each transition.
        """
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

    exact_obs = extract_obs_vectors_from_trajectory(
        exact_traj, obs_dim, agent_count, num_actions)
    pred_obs = extract_obs_vectors_from_trajectory(
        predicted_traj, obs_dim, agent_count, num_actions)

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

    # Lier les points
    for i in range(0, len(exact_obs)-1):
        plt.plot(X_2d[i:i+2, 0], X_2d[i:i+2, 1],
                 color='blue', linestyle='--', linewidth=0.5)
    for i in range(n_obs, len(X_2d)-1):
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
    return save_path


if __name__ == '__main__':
    odf_runner = ODF_LSTM_runner()
    metadata = odf_runner.checkpoint["metadata"]
    ep_idx = 0  # random.randint(0, metadata["max_episode"] - 1)
    print(f"▶️  Selected trajectory: episode {ep_idx}")

    exact_traj, pred_traj = generate_exact_and_predicted_trajectories(
        file_path="trajectories.json",
        ep_idx=ep_idx,
        odf_runner=odf_runner,
        max_step=metadata["max_step"],
        agent_order=metadata["agent_order"],
        num_actions=metadata["num_actions"]
    )

    plot_transitions_PCA(exact_traj, pred_traj, ep_idx)
    plot_observations_PCA(exact_traj, pred_traj,
                          metadata["obs_dim_per_agent"],
                          len(metadata["agent_order"]),
                          metadata["num_actions"],
                          ep_idx)
