from typing import Union
import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
import os

from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from torch.nn.functional import cosine_similarity
from math import log10
from collect import load_episode_step_data
from reduction import load_autoencoder, save_frame_and_reconstruct, get_frame_shape, load_episode_step_data


class LatentPredictorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LatentPredictorMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)


def train_predictive_mlp(
    latent_trajectory_file: str,
    num_agents: int,
    num_actions: int,
    hidden_dim: int = 128,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    model_path: str = "predictive_mlp.pth"
):
    with open(latent_trajectory_file, 'r') as f:
        data = json.load(f)

    X, Y = [], []

    for episode in data.values():
        steps = list(episode.values())
        for i in range(len(steps) - 1):
            curr = steps[i]
            nxt = steps[i + 1]

            latent_frame = np.array(curr["latent_frame"])
            next_latent_frame = np.array(nxt["latent_frame"])

            # One-hot encode all actions
            action_vector = np.zeros(num_agents * num_actions)
            for j in range(num_agents):
                a = curr["joint_action"][f"agent_{j}"]
                action_vector[j * num_actions + a] = 1

            input_vector = np.concatenate([latent_frame, action_vector])
            X.append(input_vector)
            Y.append(next_latent_frame)

    X = torch.tensor(np.array(X), dtype=torch.float32)
    Y = torch.tensor(np.array(Y), dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True)

    model = LatentPredictorMLP(
        input_dim=X.shape[1], hidden_dim=hidden_dim, output_dim=Y.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        total_loss = 0.0
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"Predictive model saved at {model_path}")
    return model, {"input_dim": X.shape[1], "hidden_dim": hidden_dim, "output_dim": Y.shape[1]}


def save_image(image_array, image_file_path, trajectory_file_path):
    # Save reconstructed frame
    frame_shape = get_frame_shape(trajectory_file_path)
    img = image_array.reshape(frame_shape)
    img = Image.fromarray(image_array.astype(np.uint8))
    img.save(image_file_path)


def decode_image(latent_frame, autoencoder, trajectory_file_path):
    latent_tensor = torch.tensor(
        latent_frame, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        reconstructed_frame = autoencoder.decoder(
            latent_tensor).squeeze(0).numpy()
    height, width, channels = get_frame_shape(trajectory_file_path)
    reconstructed_frame = (reconstructed_frame * 255).astype(np.uint8).reshape(height, width, channels)
    return reconstructed_frame

def encode_image(image_array, autoencoder):
    latent_frame = image_array / 255.0
    frame_tensor = torch.tensor(
        latent_frame.reshape(1, -1), dtype=torch.float32)
    with torch.no_grad():
        latent_vector = autoencoder.encoder(
            frame_tensor).squeeze(0).numpy().tolist()
    return latent_vector


def load_predictive_mlp(model_path: str, input_dim: int, hidden_dim: int = 128, output_dim: int = 2) -> LatentPredictorMLP:
    model = LatentPredictorMLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode
    print(f"Model successfully loaded from {model_path}")
    return model

if __name__ == "__main__":
    # Path to the latent trajectories file
    trajectory_file_path = "trajectories.json"
    latent_trajectory_file = "latent_trajectories.json"
    autoencoder_path = "autoencoder_model.pth"

    # Parameters
    num_agents = 2  # Adapt this if using more than 2 agents
    num_actions = 6  # Number of discrete actions per agent in Overcooked
    hidden_dim = 128
    epochs = 30
    batch_size = 32
    learning_rate = 1e-3
    model_path = "predictive_mlp.pth"

    if os.path.exists(autoencoder_path):
        autoencoder = load_autoencoder(autoencoder_path, trajectory_file_path)
        autoencoder.eval()
    else:
        print("Autoencoder not found. Skipping reconstruction.")

    if os.path.exists(model_path):
        model = load_predictive_mlp(model_path, 76, 128, 64)
    else:
        model, model_shape = train_predictive_mlp(
            latent_trajectory_file=latent_trajectory_file,
            num_agents=num_agents,
            num_actions=num_actions,
            hidden_dim=hidden_dim,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            model_path=model_path
        )

    # Randomly select an episode and a step
    random_episode = 0 # random.choice(episode_keys)
    step_idx = 0 # random.randint(0, len(steps) - 2)

    step_data = load_episode_step_data(file_path=trajectory_file_path, episode_idx= random_episode, step_idx= step_idx)
    actions = step_data["joint_action"]
    print("Played actions: ", actions)
    frame = np.array(step_data["frame"])
    print("===> ", frame.shape)
    save_image(frame, "true_frame.png", trajectory_file_path)

    step_data = load_episode_step_data(file_path=trajectory_file_path, episode_idx= random_episode, step_idx= step_idx + 1)
    next_frame = np.array(step_data["frame"])
    save_image(next_frame, "true_next_frame.png", trajectory_file_path)

    latent_frame = encode_image(frame, autoencoder)

    # One-hot encode actions
    action_vector = np.zeros(num_agents * num_actions)
    for j in range(num_agents):
        a = actions[f"agent_{j}"]
        action_vector[j * num_actions + a] = 1

    # Predict the next latent frame
    input_vector = np.concatenate([latent_frame, action_vector])
    input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        predicted_next_latent = model(input_tensor).squeeze(0).numpy()

    # === Optionally, decode and save the frame (requires autoencoder) ===
    reconstructed_frame = decode_image(predicted_next_latent, autoencoder, trajectory_file_path)

    # Save reconstructed frame
    save_image(reconstructed_frame, "predicted_next_frame.png",
               trajectory_file_path)
    print("Predicted frame saved as predicted_next_frame.png")
