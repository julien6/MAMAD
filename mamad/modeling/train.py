import pickle
import os
import torch.nn as nn
import numpy as np
import torch
import optuna
import random

from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from collect import load_episode_step_data
from typing import Dict
from prepare_data import OvercookedSequenceLSTMDatasetLazy, infer_metadata_from_json, load_episode_as_sequence_lazy, one_hot_encode_action


class ODF_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim)
        return self.fc(out)    # (batch, seq_len, output_dim)


def sample_eval_episodes(episode_indices: List[int], ratio: float = 0.2, seed: int = 42) -> List[int]:
    """
    Samples a given proportion of episode indices for evaluation.

    Args:
        episode_indices: Full list of available episode indices
        ratio: Proportion to use for evaluation (e.g., 0.2 for 20%)
        seed: Seed for reproducibility

    Returns:
        List of indices selected for evaluation
    """
    assert 0 < ratio < 1, "The ratio must be within (0, 1)"
    eval_size = max(1, int(len(episode_indices) * ratio))
    random.seed(seed)
    return random.sample(episode_indices, eval_size)


def load_lstm_model(checkpoint_path: str) -> Tuple[ODF_LSTM, Dict]:
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    model = ODF_LSTM(
        input_dim=checkpoint['input_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        output_dim=checkpoint['output_dim'],
        num_layers=checkpoint['num_layers']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✅ ODF LSTM model restored from {checkpoint_path}")
    return model, checkpoint['metadata']


def train_odf_lstm(file_path: str, metadata: Dict,
                   hidden_dim: int = 128, num_layers: int = 1,
                   num_epochs: int = 10, batch_size: int = 1, lr: float = 1e-3, save_path: str = "checkpoints/lstm_checkpoint.pth"):
    """
    Initializes and trains an ODF_LSTM model from the provided trajectory file and metadata.
    """
    # Sample example to obtain dimensions
    x_seq, y_seq = load_episode_as_sequence_lazy(
        file_path=file_path,
        episode_idx=0,
        agent_order=metadata["agent_order"],
        num_actions=metadata["num_actions"],
        nb_step=metadata["max_step"]
    )
    input_dim = x_seq.shape[-1]
    output_dim = y_seq.shape[-1]

    dataset = OvercookedSequenceLSTMDatasetLazy(
        file_path=file_path,
        episode_indices=list(range(metadata["max_episode"])),
        agent_order=metadata["agent_order"],
        num_actions=metadata["num_actions"],
        nb_step=metadata["max_step"]
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = ODF_LSTM(input_dim, hidden_dim, output_dim, num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print("\n=== Training LSTM model ===")
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for x_batch, y_batch in dataloader:
            pred = model(x_batch)
            loss = criterion(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"[Epoch {epoch + 1}/{num_epochs}] Loss: {epoch_loss:.6f}")

    # === Saving LSTM model with hyperparameter ===
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'output_dim': output_dim,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'lr': lr,
        'metadata': metadata
    }, save_path)
    print(f"✅ Model saved in {save_path}")

    return model


def get_search_intervals(metadata: Dict, hp_list: List[str]) -> Dict[str, Tuple[float, float]]:
    """
    Determines the relevant search intervals for each hyperparameter to be optimized.
    """
    obs_size = metadata["obs_dim_per_agent"] * len(metadata["agent_order"])
    action_size = metadata["num_actions"] * len(metadata["agent_order"])
    input_dim = obs_size + action_size

    intervals = {}

    if "hidden_dim" in hp_list:
        # rule of thumb: 1x to 5x the input size
        intervals["hidden_dim"] = (input_dim, input_dim * 5)

    if "num_layers" in hp_list:
        # in general, between 1 and 3 are sufficient
        intervals["num_layers"] = (1, 3)

    if "batch_size" in hp_list:
        intervals["batch_size"] = (1, 8)  # LSTM batch-wise training

    if "lr" in hp_list:
        intervals["lr"] = (1e-4, 1e-2)  # standard rule

    if "num_epochs" in hp_list:
        # we can adjust by early stopping then
        intervals["num_epochs"] = (5, 20)

    return intervals


def run_hpo(file_path: str, metadata: Dict, hp_list: List[str], n_trials: int = 20,
            eval_ratio: float = 0.2) -> Dict:
    """
    Runs an Optuna optimization to find the best hyperparameters within the provided ranges.
    """

    search_space = get_search_intervals(metadata, hp_list)
    all_episodes = list(range(metadata["max_episode"]))
    eval_episodes = sample_eval_episodes(all_episodes, ratio=eval_ratio)

    def objective(trial):
        # Suggest hyperparams
        hidden_dim = int(trial.suggest_int("hidden_dim", int(
            search_space["hidden_dim"][0]), int(search_space["hidden_dim"][1])))
        num_layers = int(trial.suggest_int("num_layers", int(
            search_space["num_layers"][0]), int(search_space["num_layers"][1])))
        batch_size = int(trial.suggest_int("batch_size", int(
            search_space["batch_size"][0]), int(search_space["batch_size"][1])))
        lr = trial.suggest_float(
            "lr", search_space["lr"][0], search_space["lr"][1], log=True)
        num_epochs = int(trial.suggest_int("num_epochs", int(
            search_space["num_epochs"][0]), int(search_space["num_epochs"][1])))

        # Training on ALL trajectories
        model = train_odf_lstm(file_path, metadata,
                               hidden_dim=hidden_dim,
                               num_layers=num_layers,
                               num_epochs=num_epochs,
                               batch_size=batch_size,
                               lr=lr)

        # Evaluation on the sample of trajectories
        model.eval()
        criterion = nn.MSELoss()
        total_loss = 0.0
        count = 0
        for ep_idx in eval_episodes:
            x_seq, y_seq = load_episode_as_sequence_lazy(
                file_path, episode_idx=ep_idx,
                agent_order=metadata["agent_order"],
                num_actions=metadata["num_actions"],
                nb_step=metadata["max_step"]
            )
            x_tensor = torch.tensor(
                x_seq[None, :, :], dtype=torch.float32)  # add batch dim
            y_tensor = torch.tensor(y_seq[None, :, :], dtype=torch.float32)
            with torch.no_grad():
                pred = model(x_tensor)
                loss = criterion(pred, y_tensor)
                total_loss += loss.item()
                count += 1

        return total_loss / count

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("\n=== Best hyperparameters found ===")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")

    return study.best_params


def prepare_input(obs_dict: Dict[str, List[float]], act_dict: Dict[str, int],
                  agent_order: List[str], num_actions: int) -> torch.Tensor:
    obs = np.concatenate([np.array(obs_dict[agent]) for agent in agent_order])
    act = np.concatenate([one_hot_encode_action(
        act_dict[agent], num_actions) for agent in agent_order])
    x = np.concatenate([obs, act])[None, None, :]
    return torch.tensor(x, dtype=torch.float32)


def unpack_output(output: torch.Tensor, agent_order: List[str], obs_dim_per_agent: int) -> Dict[str, List[float]]:
    output = output.squeeze(0).squeeze(0).detach().numpy()
    obs_dict = {}
    for i, agent in enumerate(agent_order):
        start = i * obs_dim_per_agent
        end = (i + 1) * obs_dim_per_agent
        obs_dict[agent] = output[start:end].tolist()
    return obs_dict


class ODF_LSTM_runner:

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_path = os.path.join(
            checkpoint_dir, "lstm_checkpoint.pth")
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"Aucun checkpoint trouvé dans {self.checkpoint_path}")

        self.checkpoint = torch.load(
            self.checkpoint_path, map_location=torch.device("cpu"))
        self.model = ODF_LSTM(
            input_dim=self.checkpoint["input_dim"],
            hidden_dim=self.checkpoint["hidden_dim"],
            output_dim=self.checkpoint["output_dim"],
            num_layers=self.checkpoint["num_layers"]
        )
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.eval()

    def run_ODF_LSTM(self, obs_dict: Dict[str, List[float]], act_dict: Dict[str, int]) -> Dict[str, List[float]]:
        input_tensor = prepare_input(
            obs_dict, act_dict, self.checkpoint["metadata"]["agent_order"], self.checkpoint["metadata"]["num_actions"])

        with torch.no_grad():
            output_tensor = self.model(input_tensor)

        predicted_obs = unpack_output(
            output_tensor, self.checkpoint["metadata"]["agent_order"], self.checkpoint["metadata"]["obs_dim_per_agent"])
        return predicted_obs


if __name__ == '__main__':

    metadata = infer_metadata_from_json("trajectories.json")

    # === TRAIN ===

    # best_hparams = run_hpo(
    #     file_path="trajectories.json",
    #     metadata=metadata,
    #     hp_list=["hidden_dim", "num_layers", "batch_size", "lr", "num_epochs"],
    #     n_trials=20,
    #     eval_ratio=0.2  # test on 20% of trajectories
    # )

    # print("Using best hyper-parameters for the final training")
    # model = train_odf_lstm("trajectories.json", metadata, **best_hparams)

    # === FINAL TEST : Run model on random (obs, action) pair ===
    print("\n=== Testing run_ODF_LSTM on a random transition ===")
    random_episode = random.randint(0, metadata["max_episode"] - 1)
    random_step = random.randint(
        0, metadata["max_step"] - 2)  # pour que step+1 existe

    current_step = load_episode_step_data(
        "trajectories.json", random_episode, random_step)
    next_step = load_episode_step_data(
        "trajectories.json", random_episode, random_step + 1)

    if current_step is not None and next_step is not None:
        obs = current_step["joint_observation"]
        act = current_step["joint_action"]
        true_next_obs = next_step["joint_observation"]

        odf_runner = ODF_LSTM_runner()
        predicted_next_obs = odf_runner.run_ODF_LSTM(obs, act)

        print(f"Episode {random_episode}, Step {random_step}")
        print("\n[Agent 0] True next obs (first 10):")
        print(true_next_obs["agent_0"][:10])
        print("[Agent 0] Predicted next obs (first 10):")
        print(np.round(predicted_next_obs["agent_0"][:10], 2))
