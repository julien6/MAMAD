import json
import os
import numpy as np
import jax
import jax.numpy as jnp
import optax
import pickle
import optuna

from flax import linen as nn
from flax.training import train_state
from typing import Dict, List, Tuple
import psutil
from sklearn.model_selection import train_test_split
from tqdm import trange


def get_hardware_info():
    n_devices = jax.local_device_count()
    devices = jax.devices()

    hardware_info = {
        "num_devices": n_devices,
        "device_type": devices[0].device_kind if devices else "Unknown",
        "num_cpu_cores": psutil.cpu_count(logical=True),
        "total_ram_gb": psutil.virtual_memory().total / (1024**3)
    }

    # Approximate VRAM info only if CUDA visible (optional fallback)
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        hardware_info["total_memory_gb"] = mem_info.total / (1024**3)
        pynvml.nvmlShutdown()
    except Exception as e:
        hardware_info["total_memory_gb"] = None  # Cannot detect

    return hardware_info


def get_hyperparameter_search_space(hardware_info):
    search_space = {}

    # Rule: if you have a lot of VRAM you can afford bigger LSTM
    if hardware_info["total_memory_gb"] and hardware_info["total_memory_gb"] >= 16:
        search_space["hidden_dim"] = [
            x for x in range(128, 1024, 64)]  # large models
    else:
        search_space["hidden_dim"] = [
            x for x in range(64, 512, 64)]    # smaller models

    # usually between 1 and 4 layers for stability
    search_space["num_layers"] = (1, 4)
    search_space["learning_rate"] = (1e-4, 1e-2)  # classic interval for Adam
    search_space["batch_size"] = [x for x in range(
        hardware_info["num_devices"], hardware_info["num_devices"]*8, 16)]
    search_space["num_epochs"] = (20, 100)
    search_space["latent_dim"] = [16, 32, 64, 128]

    return search_space

# ----------------------
# VAE Definition
# ----------------------


class Encoder(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        mean_x = nn.Dense(self.latent_dim)(x)
        logvar_x = nn.Dense(self.latent_dim)(x)
        return mean_x, logvar_x


class Decoder(nn.Module):
    original_dim: int

    @nn.compact
    def __call__(self, z):
        z = nn.Dense(64)(z)
        z = nn.relu(z)
        return nn.Dense(self.original_dim)(z)


class VAE(nn.Module):
    latent_dim: int
    original_dim: int

    def setup(self):
        self.encoder = Encoder(self.latent_dim)
        self.decoder = Decoder(self.original_dim)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mean, logvar, key):
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(key, std.shape)
        return mean + eps * std

    def __call__(self, x, key):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar, key)
        x_recon = self.decode(z)
        return x_recon, mean, logvar


# ----------------------
# Utils
# ----------------------

def make_vae_loss_fn(model):
    """Returns a JIT-compiled loss function bound to a specific model."""
    @jax.jit
    def loss_fn(params, batch, key):
        recon_x, mean, logvar = model.apply({'params': params}, batch, key)
        recon_loss = jnp.mean((batch - recon_x) ** 2)
        kl_div = -0.5 * jnp.mean(1 + logvar - mean**2 - jnp.exp(logvar))
        return recon_loss + kl_div
    return loss_fn


def one_hot_encode_observation(obs: List[int], num_categories=15) -> np.ndarray:
    return np.concatenate([np.eye(num_categories)[val] for val in obs])


def extract_all_observations(json_data: Dict, num_categories=15, max_cells=9) -> np.ndarray:
    """
    Transform raw observations into a dataset of one-hot encoded vectors.
    """
    all_obs = []
    for episode in json_data.values():
        for step in episode.values():
            for obs in step["joint_observation"].values():
                if len(obs) >= max_cells:
                    one_hot = one_hot_encode_observation(
                        obs[:max_cells], num_categories)
                    all_obs.append(one_hot)
    return np.array(all_obs)


# ----------------------
# Training
# ----------------------

def create_train_state(rng, model, learning_rate=1e-3):
    key, subkey = jax.random.split(rng)
    params = model.init({'params': key}, jnp.ones(
        [1, model.original_dim]), subkey)['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def make_train_step(model):
    @jax.jit
    def train_step(state, batch, key):
        def loss_fn(params):
            recon_x, mean, logvar = model.apply({'params': params}, batch, key)
            recon_loss = jnp.mean((batch - recon_x) ** 2)
            kl_div = -0.5 * jnp.mean(1 + logvar - mean**2 - jnp.exp(logvar))
            return recon_loss + kl_div
        grads = jax.grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads)
    return train_step


def save_vae_model(save_path, vae_model, state):
    with open(save_path, 'wb') as f:
        pickle.dump({
            'model_def': vae_model,
            'params': state.params
        }, f)


def load_vae_model(save_path: str) -> Tuple[VAE, train_state.TrainState, int]:
    with open(save_path, 'rb') as f:
        checkpoint = pickle.load(f)
    model_def = checkpoint['model_def']
    params = checkpoint['params']

    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, model_def)
    state = state.replace(params=params)

    return model_def, state


def train_vae(model, data: np.ndarray, num_epochs=50, batch_size=32, lr=1e-3, save_interval: int = 5, save_path: str = "vae_model.pkl") -> train_state.TrainState:
    rng = jax.random.PRNGKey(0)
    key = rng
    state = create_train_state(rng, model, lr)

    dataset_size = data.shape[0]
    num_batches = int(np.ceil(dataset_size / batch_size))

    train_step_fn = make_train_step(model)

    for epoch in trange(num_epochs):
        key, subkey = jax.random.split(key)
        perm = np.random.permutation(dataset_size)
        for i in range(num_batches):
            batch_idx = perm[i * batch_size: (i + 1) * batch_size]
            batch = jnp.array(data[batch_idx])
            key, subkey = jax.random.split(key)
            state = train_step_fn(state, batch, subkey)

        if epoch % save_interval == 0:
            save_vae_model(save_path, model, state)

    return state


# ----------------------
# Pipeline Function
# ----------------------

def run_full_vae_pipeline(input_json_path: str, latent_dim=32, save_path="vae_model.pkl", save_interval: int = 5, batch_size: int = 32, num_epochs: int = 100) -> Tuple[VAE, train_state.TrainState]:
    with open(input_json_path, 'r') as f:
        raw_data = json.load(f)

    obs_data = jnp.array(extract_all_observations(raw_data))
    original_dim = obs_data.shape[1]

    if os.path.exists(save_path):
        print("🔁 Reprise depuis le dernier checkpoint...")
        vae_model, state = load_vae_model(save_path)
    else:
        vae_model = VAE(latent_dim=latent_dim, original_dim=original_dim)

    trained_state = train_vae(
        vae_model, obs_data, num_epochs=num_epochs, batch_size=batch_size, save_interval=save_interval)

    save_vae_model(save_path, vae_model, trained_state)

    return vae_model, trained_state


def run_hpo_with_optuna(input_json_path: str, num_trials: int = 20, custom_hpo_intervals: dict = None):

    hardware_info = get_hardware_info()
    search_space = get_hyperparameter_search_space(hardware_info)
    search_space.update(custom_hpo_intervals)

    print(search_space)

    with open(input_json_path, 'r') as f:
        raw_data = json.load(f)

    obs_data = extract_all_observations(raw_data)
    original_dim = obs_data.shape[1]

    def objective(trial):
        # Hyperparameters to optimize
        hidden_dim = trial.suggest_categorical(
            'hidden_dim', search_space['hidden_dim'])
        num_layers = trial.suggest_int(
            'num_layers', *search_space['num_layers'])
        learning_rate = trial.suggest_loguniform(
            'learning_rate', *search_space['learning_rate'])
        batch_size = trial.suggest_categorical(
            'batch_size', search_space['batch_size'])
        num_epochs = trial.suggest_int(
            'num_epochs', *search_space['num_epochs'])
        latent_dim = trial.suggest_categorical(
            'latent_dim', search_space['latent_dim'])

        class CustomEncoder(nn.Module):
            latent_dim: int
            hidden_dim: int
            num_layers: int

            @nn.compact
            def __call__(self, x):
                for _ in range(self.num_layers):
                    x = nn.Dense(self.hidden_dim)(x)
                    x = nn.relu(x)
                mean_x = nn.Dense(self.latent_dim)(x)
                logvar_x = nn.Dense(self.latent_dim)(x)
                return mean_x, logvar_x

        class CustomDecoder(nn.Module):
            original_dim: int
            hidden_dim: int
            num_layers: int

            @nn.compact
            def __call__(self, z):
                for _ in range(self.num_layers):
                    z = nn.Dense(self.hidden_dim)(z)
                    z = nn.relu(z)
                return nn.Dense(self.original_dim)(z)

        class CustomVAE(nn.Module):
            latent_dim: int
            original_dim: int
            hidden_dim: int
            num_layers: int

            def setup(self):
                self.encoder = CustomEncoder(
                    self.latent_dim, self.hidden_dim, self.num_layers)
                self.decoder = CustomDecoder(
                    self.original_dim, self.hidden_dim, self.num_layers)

            def encode(self, x):
                return self.encoder(x)

            def decode(self, z):
                return self.decoder(z)

            def reparameterize(self, mean, logvar, key):
                std = jnp.exp(0.5 * logvar)
                eps = jax.random.normal(key, std.shape)
                return mean + eps * std

            def __call__(self, x, key):
                mean, logvar = self.encode(x)
                z = self.reparameterize(mean, logvar, key)
                x_recon = self.decode(z)
                return x_recon, mean, logvar

        vae_model = CustomVAE(
            latent_dim=latent_dim,
            original_dim=original_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )

        loss_fn = make_vae_loss_fn(vae_model)
        train_step_fn = make_train_step(vae_model)

        rng = jax.random.PRNGKey(0)
        state = create_train_state(rng, vae_model, learning_rate)

        dataset_size = obs_data.shape[0]
        num_batches = int(np.ceil(dataset_size / batch_size))
        key = rng

        for epoch in trange(num_epochs):
            key, subkey = jax.random.split(key)
            perm = np.random.permutation(dataset_size)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size: (i + 1) * batch_size]
                batch = jnp.array(obs_data[batch_idx])
                key, subkey = jax.random.split(key)
                state = train_step_fn(state, batch, subkey)

        # Compute final loss on full dataset
        full_batch = jnp.array(obs_data)
        key, _ = jax.random.split(key)
        final_loss = float(loss_fn(state.params, full_batch, key))
        return final_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=num_trials, show_progress_bar=True)

    print("✅ Best trial:")
    print(study.best_trial)


if __name__ == '__main__':

    # ----------------------
    # Execution
    # ----------------------

    custom_intervals = {
        "latent_dim": [64],
        "num_layers": (2, 2),
        "num_epochs": (20, 20),
        "batch_size": [128],
        # "hidden_dim": [256],
        # "learning_rate": (1e-2, 1e-2)
    }

    run_hpo_with_optuna("trajectories.json", num_trials=20,
                        custom_hpo_intervals=custom_intervals)

    # input_path = "trajectories.json"
    # vae_model, trained_state = run_full_vae_pipeline(
    #     input_path, latent_dim=64, save_interval=20, batch_size=256, num_epochs=1000)
    # print("✅ VAE trained and saved.")
