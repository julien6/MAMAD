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


class Autoencoder(nn.Module):
    def __init__(self, input_shape):
        super(Autoencoder, self).__init__()

        # Déterminer la taille d'entrée de l'image (hauteur * largeur * canaux)
        height, width, channels = input_shape

        # Taille de l'entrée
        input_dim = height * width * channels

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            # La sortie doit être de la même taille que l'entrée
            nn.Linear(512, input_dim),
            nn.Sigmoid()  # pour que les valeurs soient entre 0 et 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_autoencoder(file_path: str, num_episodes: int, max_steps: int, epochs: int = 5, batch_size: int = 16, learning_rate: float = 1e-3):
    # Get the shape of a single frame from the first episode and step
    step_data = load_episode_step_data(file_path, 0, 0)
    if step_data is None:
        print("Failed to load step data for determining frame dimensions.")
        return None

    # Get the shape of the frame (height, width, channels)
    frame = np.array(step_data['frame'])
    height, width, channels = frame.shape

    # Initialize the autoencoder with the dynamic input shape
    autoencoder = Autoencoder((height, width, channels))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

    autoencoder.train()

    # Track epoch losses
    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for episode_idx in tqdm(range(num_episodes), desc=f"Epoch {epoch + 1}/{epochs}"):
            for step_idx in tqdm(range(max_steps), desc=f"Episode {episode_idx} Steps"):
                # Load the current episode's step data
                step_data = load_episode_step_data(
                    file_path, episode_idx, step_idx)
                if step_data is None:
                    continue  # Skip if step data is unavailable

                # Convert the frame to a 1D flattened array and prepare for training
                frame = np.array(step_data['frame']) / \
                    255.0  # Normalize the frame
                frame = frame.reshape(1, -1)  # Flatten the frame into 1D

                # Convert actions (not necessary for autoencoder, but you can use them for supervision)
                actions = step_data['actions']

                # Convert frame to tensor and move to the correct device
                frame_tensor = torch.tensor(frame, dtype=torch.float32)

                # Training step: Forward pass, calculate loss, and backpropagate
                optimizer.zero_grad()
                # Get the reconstructed frame
                reconstructed = autoencoder(frame_tensor)
                # Calculate the reconstruction loss
                loss = criterion(reconstructed, frame_tensor)

                loss.backward()  # Backpropagation
                optimizer.step()  # Optimizer step

                total_loss += loss.item()  # Track total loss
                num_batches += 1

        # Average loss for the epoch
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    # Save the trained autoencoder model
    model_path = "autoencoder_model.pth"
    torch.save(autoencoder.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Save epoch loss statistics
    stats = {
        "epoch_losses": epoch_losses,
        "final_loss": epoch_losses[-1] if epoch_losses else None
    }
    stats_path = "autoencoder_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f)
    print(f"Statistics saved to {stats_path}")

    return autoencoder, stats


def calculate_psnr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    mse = nn.functional.mse_loss(reconstructed, original)
    if mse == 0:
        return float("inf")
    max_pixel = 1.0  # Supposons que les pixels sont normalisés entre 0 et 1
    psnr = 20 * log10(max_pixel / torch.sqrt(mse))
    return psnr


def calculate_ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
    return ssim(original, reconstructed, win_size=5, channel_axis=-1, data_range=1)


def calculate_variance_explained(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    var_original = torch.var(original)
    var_error = torch.var(original - reconstructed)
    ve = 1 - (var_error / var_original)
    return ve.item()


def calculate_cosine_similarity(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    original_flat = original.view(-1)
    reconstructed_flat = reconstructed.view(-1)
    return cosine_similarity(original_flat.unsqueeze(0), reconstructed_flat.unsqueeze(0)).item()


def analyze_latent_distribution(autoencoder, data_loader):
    latents = []
    for batch in data_loader:
        with torch.no_grad():
            latent = autoencoder.encoder(batch).numpy()
            latents.append(latent)
    latents = np.concatenate(latents, axis=0)
    variance_per_dimension = np.var(latents, axis=0)
    return variance_per_dimension


def plot_reconstruction_error_histogram(original: torch.Tensor, reconstructed: torch.Tensor):
    errors = ((original - reconstructed) ** 2).mean(dim=1).sqrt()
    plt.hist(errors.detach().cpu().numpy(), bins=50)
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Histogram of Reconstruction Errors')
    plt.show()


def get_frame_shape(trajectory_file_path: str) -> np.shape:
    step_data = load_episode_step_data(trajectory_file_path, 0, 0)
    if step_data is None:
        print(
            f"Les données pour l'épisode {0} et le pas {0} n'ont pas pu être chargées.")
        return
    return np.array(step_data["frame"]).shape


def save_frame_and_reconstruct(autoencoder, file_path: str, episode_idx: int, step_idx: int, original_output_path: str, reconstructed_output_path: str):
    """
    Sélectionne une frame, la passe dans l'autoencodeur et sauvegarde la frame originale
    et la reconstruction en PNG.

    Parameters:
        autoencoder (torch.nn.Module): Le modèle d'autoencodeur entraîné.
        file_path (str): Chemin du fichier JSON contenant les trajectoires.
        episode_idx (int): L'indice de l'épisode à charger.
        step_idx (int): Le numéro du pas de l'épisode pour lequel les frames seront utilisées.
        original_output_path (str): Chemin pour sauvegarder la frame originale en PNG.
        reconstructed_output_path (str): Chemin pour sauvegarder la reconstruction en PNG.
    """
    # Charger les données de l'épisode et du step
    step_data = load_episode_step_data(file_path, episode_idx, step_idx)
    if step_data is None:
        print(
            f"Les données pour l'épisode {episode_idx} et le pas {step_idx} n'ont pas pu être chargées.")
        return

    frame_shape = get_frame_shape(trajectory_file)

    # Concaténer toutes les frames et choisir une au hasard
    original_image = np.array(step_data["frame"]) / 255
    # Flatten the original rendered image
    processed_frame = original_image.reshape(-1, np.prod(frame_shape))

    # Sauvegarder la frame originale en PNG
    # Convertir en 0-255 si nécessaire
    original_img = Image.fromarray((original_image * 255).astype(np.uint8))
    original_img.save(original_output_path)
    print(f"Frame originale sauvegardée sous {original_output_path}")

    # Passer la frame dans l'autoencodeur pour obtenir la reconstruction
    autoencoder.eval()  # S'assurer que le modèle est en mode évaluation
    with torch.no_grad():
        input_tensor = torch.tensor(processed_frame, dtype=torch.float32).unsqueeze(
            0)  # Ajouter la dimension batch
        reconstructed_tensor = autoencoder(input_tensor).squeeze(
            0)  # Supprimer la dimension batch
        reconstructed_image = reconstructed_tensor.numpy().reshape(frame_shape)

    # Sauvegarder la reconstruction en PNG
    reconstructed_img = Image.fromarray(
        (reconstructed_image * 255).astype(np.uint8))
    reconstructed_img.save(reconstructed_output_path)
    print(f"Image reconstruite sauvegardée sous {reconstructed_output_path}")


def load_autoencoder(model_path: str, trajectory_file_path: str):

    # Get the shape of the frame (height, width, channels)
    frame_shape = get_frame_shape(trajectory_file_path)

    # Initialiser l'autoencodeur avec les mêmes dimensions
    autoencoder = Autoencoder(frame_shape)

    # Charger les poids sauvegardés dans le modèle
    autoencoder.load_state_dict(torch.load(model_path))
    autoencoder.eval()  # Mettre le modèle en mode évaluation
    print(f"Autoencodeur chargé avec succès depuis {model_path}")
    return autoencoder


def validate_autoencoder(autoencoder: Union[Autoencoder, str], file_path: str, num_episodes: int, max_steps: int):

    if isinstance(autoencoder, str):
        autoencoder = load_autoencoder(autoencoder, file_path)

    autoencoder.eval()
    psnr_list, ssim_list, ve_list, cosine_sim_list = [], [], [], []

    # Iterate through the episodes and steps to calculate metrics
    for episode_idx in tqdm(range(num_episodes), desc="Validating Episodes"):
        for step_idx in tqdm(range(max_steps), desc=f"Episode {episode_idx} Steps"):
            step_data = load_episode_step_data(
                file_path, episode_idx, step_idx)
            if step_data is None:
                continue

            # Normalize and flatten the observations (frames)
            observations = np.array(step_data["frame"]) / 255

            observations_shape = observations.shape

            observations = observations.reshape(-1,
                                                np.prod(observations_shape))
            observations = torch.tensor(observations, dtype=torch.float32)

            # Reconstruct the frames using the autoencoder
            with torch.no_grad():
                reconstructed = autoencoder(observations)

            # Calculate metrics for each observation
            original_frame = observations.reshape(observations_shape).numpy()
            reconstructed_frame = reconstructed.reshape(
                observations_shape).numpy()

            # PSNR, SSIM, Variance Explained, and Cosine Similarity
            psnr_list.append(calculate_psnr(
                observations, reconstructed))
            ssim_list.append(calculate_ssim(
                original_frame, reconstructed_frame))
            ve_list.append(calculate_variance_explained(
                observations, reconstructed))
            cosine_sim_list.append(calculate_cosine_similarity(
                observations, reconstructed))

    # Compute the average of each metric
    stats = {
        "PSNR": float(np.mean(psnr_list)),
        "SSIM": float(np.mean(ssim_list)),
        "Variance Explained": float(np.mean(ve_list)),
        "Cosine Similarity": float(np.mean(cosine_sim_list))
    }

    # Print the validation statistics
    print("Validation Metrics:", stats)

    # # Optionally, you can plot some additional metrics like histograms of reconstruction errors
    # plot_reconstruction_error_histogram(observations, reconstructed)

    return stats


def convert_to_latent_trajectories(autoencoder: torch.nn.Module, input_file: str, num_episodes: int, max_steps: int, output_file: str = "latent_trajectories.json"):
    autoencoder.eval()
    if os.path.exists(output_file):
        os.remove(output_file)

    with open(output_file, 'a') as f_out:
        f_out.write("{")

        for episode_idx in tqdm(range(num_episodes)):

            f_out.write(f'\n"episode_{episode_idx}": {{')

            for step_idx in tqdm(range(max_steps), desc=f"Episode {episode_idx} Steps"):
                # Load the current episode's step data
                step_data = load_episode_step_data(
                    input_file, episode_idx, step_idx)
                if step_data is None:
                    continue  # Skip if step data is unavailable

                # Convert frame to latent vector
                frame = np.array(step_data["frame"]) / 255.0
                frame_tensor = torch.tensor(
                    frame.reshape(1, -1), dtype=torch.float32)
                with torch.no_grad():
                    latent_vector = autoencoder.encoder(
                        frame_tensor).squeeze(0).numpy().tolist()

                # Build new step with latent representation
                latent_step = {
                    "latent_frame": latent_vector,
                    "joint_action": step_data["joint_action"]
                }

                f_out.write(f'\n"step_{step_idx}": {json.dumps(latent_step)}')

                if step_idx < max_steps - 1:
                    f_out.write(",")

            f_out.write("}")
            if episode_idx < num_episodes - 1:
                f_out.write(",")
        f_out.write("\n}")
    print(f"Latent trajectories saved to {output_file}")
    return output_file


if __name__ == "__main__":

    # Path to the collected trajectory file (frames + actions)
    trajectory_file = "trajectories.json"

    # # Step 1: Train the autoencoder
    # print("Training autoencoder on collected frame trajectories...")
    # autoencoder, training_stats = train_autoencoder(
    #     file_path=trajectory_file,
    #     num_episodes=5,     # Adjust this number depending on your dataset
    #     max_steps=100,
    #     epochs=10,
    #     batch_size=32,
    #     learning_rate=1e-3
    # )

    autoencoder = load_autoencoder("autoencoder_model.pth", trajectory_file)

    # # Step 2: Validate the autoencoder on the same data
    # print("\nValidating the trained autoencoder...")
    # validation_stats = validate_autoencoder(
    #     autoencoder,
    #     file_path=trajectory_file,
    #     num_episodes=5,
    #     max_steps=100
    # )

    # # Save validation stats
    # with open("autoencoder_validation_stats.json", "w") as f:
    #     json.dump(validation_stats, f, indent=4)

    # # Step 3: Convert full trajectories (frames → latent vectors)
    # print("\nConverting trajectories to latent representation...")
    # latent_output_file = "latent_trajectories.json"

    # convert_to_latent_trajectories(
    #     autoencoder, trajectory_file, num_episodes=5, max_steps=100, output_file=latent_output_file)

    # print("\n✅ Full pipeline complete: training, validation, and conversion.")

    # save_frame_and_reconstruct(autoencoder, trajectory_file, episode_idx=0, step_idx=50,
    #                                      original_output_path="original_image.png", reconstructed_output_path="reconstructed_image.png")
