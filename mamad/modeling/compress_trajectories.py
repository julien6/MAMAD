import json
import pickle
from typing import List, Tuple
import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from tqdm import trange
from vae_model import VAE


def load_vae_model(pkl_path="vae_model.pkl") -> Tuple[VAE, FrozenDict]:
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data['model_def'], data['params']


def one_hot_encode_observation(obs: List[int], num_categories=15, max_cells=9) -> jnp.ndarray:
    obs = obs[:max_cells]  # ⚠️ Limite à 9 cellules comme à l'entraînement
    obs_array = jnp.array(obs, dtype=int)
    return jax.nn.one_hot(obs_array, num_categories).reshape(-1)


def compress_trajectories_with_vae(input_path: str, output_path: str, vae_model: VAE, params: FrozenDict, max_cells=9):
    with open(input_path, 'r') as f:
        raw_data = json.load(f)

    compressed_data = {}

    for i in trange(len(raw_data)):
        ep_id = list(raw_data.keys())[i]
        episode = raw_data[ep_id]
        compressed_data[ep_id] = {}
        for step_id, step_data in episode.items():
            compressed_obs = {}
            for agent, obs in step_data["joint_observation"].items():
                one_hot_obs = one_hot_encode_observation(
                    obs, max_cells=max_cells)
                mean, _ = vae_model.apply(
                    {'params': params}, one_hot_obs[None, :], method=vae_model.encode
                )
                compressed_obs[agent] = mean.squeeze().tolist()

            compressed_data[ep_id][step_id] = {
                "joint_observation": compressed_obs,
                "joint_action": step_data["joint_action"]
            }

    with open(output_path, 'w') as f:
        json.dump(compressed_data, f, indent=2)

    print(f"✅ Compressed trajectories saved to {output_path}")


def test_encode_decode(obs: List[int], vae_model: VAE, params: FrozenDict, num_categories: int = 15, max_cells=9):
    obs = obs[:max_cells]  # ⚠️ Couper ici aussi
    one_hot_obs = one_hot_encode_observation(
        obs, num_categories=num_categories)
    print("Original non one-hot encoded input:")
    print(obs)

    print("Original one-hot encoded input:")
    print(one_hot_obs)

    # Encode
    mu, _ = vae_model.apply(
        {'params': params}, one_hot_obs[None, :], method=vae_model.encode)
    print("\nEncoded latent vector:")
    print(mu.squeeze())

    # Decode
    recon = vae_model.apply({'params': params}, mu, method=vae_model.decode)
    recon_reshaped = recon.reshape(-1, num_categories)
    decoded = jnp.argmax(recon_reshaped, axis=-1)

    print("\nDecoded observation (one-hot):")
    print(recon.tolist())

    print("\nDecoded observation (de-one-hot):")
    print(decoded.tolist())


if __name__ == "__main__":
    # Chargement du modèle VAE entraîné
    vae_model, params = load_vae_model("vae_model.pkl")

    # Compression du fichier JSON
    # compress_trajectories_with_vae(
    #     input_path="trajectories.json",
    #     output_path="latent_trajectories.json",
    #     vae_model=vae_model,
    #     params=params
    # )

    # Test visuel avec une observation brute
    test_obs = [0, 7, 1, 1, 1, 1, 1, 0, 8]  # 9 cellules
    test_encode_decode(test_obs, vae_model, params)
