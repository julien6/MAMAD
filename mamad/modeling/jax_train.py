import json
import pickle
from typing import Dict
import jax
import jax.numpy as jnp
import optax
import jax
import psutil
import optuna
import optuna.visualization as vis
import matplotlib.pyplot as plt

from flax import linen as nn
from flax.training import train_state
from tqdm import trange
from flax.training import common_utils
from flax.training import checkpoints
from flax import jax_utils


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
        search_space["hidden_size"] = (128, 1024)  # large models
    else:
        search_space["hidden_size"] = (64, 512)    # smaller models

    # usually between 1 and 4 layers for stability
    search_space["num_layers"] = (1, 4)
    search_space["learning_rate"] = (1e-4, 1e-2)  # classic interval for Adam
    search_space["batch_size"] = (
        hardware_info["num_devices"], hardware_info["num_devices"]*8)
    search_space["num_epochs"] = (20, 100)

    return search_space


def stepwise_loss_fn(params, model, inputs, targets):
    def step_fn(carry, input_target):
        x_t, y_t = input_target
        x_t = x_t[None, None, :]  # shape (1, 1, input_dim)
        y_pred, carry = model.apply({'params': params}, x_t, carry=carry)
        loss = jnp.mean((y_pred[0, 0] - y_t) ** 2)
        return carry, loss

    carry = [
        nn.OptimizedLSTMCell.initialize_carry(
            jax.random.PRNGKey(i), (1,), model.hidden_size
        ) for i in range(model.num_layers)
    ]

    _, losses = jax.lax.scan(step_fn, carry, (inputs, targets))
    return jnp.mean(losses)


def evaluate_stepwise_loss(params, model, X, Y):
    batched_stepwise_loss = jax.vmap(
        lambda x, y: stepwise_loss_fn(params, model, x, y),
        in_axes=(0, 0)
    )
    losses = batched_stepwise_loss(X, Y)
    return jnp.mean(losses)

# Simple LSTM model definition


class SimpleLSTM(nn.Module):
    hidden_size: int
    output_size: int
    num_layers: int = 1

    class StackedLSTM(nn.Module):
        hidden_size: int
        num_layers: int

        def setup(self):
            self.layers = [
                nn.OptimizedLSTMCell() for _ in range(self.num_layers)
            ]

        def __call__(self, carry, x_t):
            new_carry = []
            for i, layer in enumerate(self.layers):
                carry[i], x_t = layer(carry[i], x_t)
                new_carry.append(carry[i])
            return new_carry, x_t

    def setup(self):
        # Stack multiple layers into a single scanned module
        self.stacked_lstm = nn.scan(
            SimpleLSTM.StackedLSTM,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1, out_axes=1
        )(hidden_size=self.hidden_size, num_layers=self.num_layers)

        self.dense = nn.Dense(self.output_size)

    @nn.compact
    def __call__(self, x, carry=None):
        batch_size = x.shape[0]
        if carry is None:
            carry = [
                nn.OptimizedLSTMCell.initialize_carry(
                    jax.random.PRNGKey(i), (batch_size,), self.hidden_size
                )
                for i in range(self.num_layers)
            ]
        carry, y = self.stacked_lstm(carry, x)
        preds = self.dense(y)
        return preds, carry

# Create a train state


def create_train_state(rng, model, learning_rate, input_shape):
    params = model.init(rng, jnp.ones(input_shape))['params']

    # Learning rate schedule (cosine decay)
    lr_schedule = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=1000,    # adjust depending on dataset size
        alpha=0.1            # final LR will be 0.1 * init_value
    )

    # Chained optimizer with gradient clipping
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),  # clip gradients if norm > 1.0
        optax.adam(lr_schedule)
    )

    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# Loss function comparing prediction to target at every timestep


jit_stepwise_loss_fn = jax.jit(stepwise_loss_fn, static_argnames=["model"])

# Single training step


def build_train_step(pmap_enabled, model):

    def _single_device_train_step(state, batch_inputs, batch_targets):

        def loss(params):
            losses = []
            for ep in range(batch_inputs.shape[0]):
                loss = jit_stepwise_loss_fn(params, model,
                                            batch_inputs[ep],
                                            batch_targets[ep])
                losses.append(loss)
            return jnp.mean(jnp.array(losses))

        grads = jax.grad(loss)(state.params)
        return state.apply_gradients(grads=grads)

    def _multi_device_train_step(state, batch_inputs, batch_targets):

        def loss(params):
            losses = []
            for ep in range(batch_inputs.shape[0]):
                loss = jit_stepwise_loss_fn(params, model,
                                            batch_inputs[ep],
                                            batch_targets[ep])
                losses.append(loss)
            return jnp.mean(jnp.array(losses))

        grads = jax.grad(loss)(state.params)
        grads = jax.lax.pmean(grads, axis_name='batch')
        return state.apply_gradients(grads=grads)

    if pmap_enabled:
        return jax.pmap(_multi_device_train_step, axis_name='batch')
    else:
        return jax.jit(_single_device_train_step)


def one_hot_encode_action(action_idx, num_actions):
    vec = [0.0] * num_actions
    vec[action_idx] = 1.0
    return vec


def load_trajectories(path, agent_order, num_actions):
    with open(path, 'r') as f:
        data = json.load(f)

    inputs = []
    targets = []

    for episode in data.values():
        steps = list(episode.values())
        episode_inputs = []
        episode_targets = []
        for t in range(len(steps) - 1):
            current_step = steps[t]
            next_step = steps[t + 1]

            obs_vec = []
            act_vec = []

            for agent in agent_order:
                obs_vec += current_step['joint_observation'][agent]
                act_vec += one_hot_encode_action(
                    current_step['joint_action'][agent], num_actions)

            input_vec = obs_vec + act_vec
            target_vec = []

            for agent in agent_order:
                target_vec += next_step['joint_observation'][agent]

            episode_inputs += [input_vec]
            episode_targets += [target_vec]

        inputs += [episode_inputs]
        targets += [episode_targets]

    X = jnp.array(inputs)
    Y = jnp.array(targets)

    return X, Y


def create_batches(X, Y, batch_size, rng=None, n_devices=1):
    """Splits the full dataset into batches of complete episodes."""
    n_episodes = X.shape[0]
    indices = jnp.arange(n_episodes)
    if rng is not None:
        indices = jax.random.permutation(rng, indices)

    X = X[indices]
    Y = Y[indices]

    for start_idx in range(0, n_episodes, batch_size):
        end_idx = min(start_idx + batch_size, n_episodes)
        batch_X = X[start_idx:end_idx]
        batch_Y = Y[start_idx:end_idx]

        if n_devices > 1:
            # reshape batch to [n_devices, batch_per_device, ...]
            batch_X = batch_X.reshape(
                (n_devices, batch_X.shape[0] // n_devices) + batch_X.shape[1:])
            batch_Y = batch_Y.reshape(
                (n_devices, batch_Y.shape[0] // n_devices) + batch_Y.shape[1:])

        yield batch_X, batch_Y


def build_one_epoch(train_step, batch_size, n_devices):
    @jax.jit
    def _one_epoch(state_rng):
        state, rng = state_rng
        rng, rng_epoch = jax.random.split(rng)

        for batch_X, batch_Y in create_batches(X, Y, batch_size, rng_epoch, n_devices):
            state = train_step(state, batch_X, batch_Y)

        return state, rng

    return _one_epoch


def train_one_model(X, Y, hidden_size, num_layers, learning_rate, batch_size, num_epochs, trial=None, save_model=False, target_loss=None):
    """Train a model for a few epochs and return validation loss (for HPO)."""
    n_devices = jax.local_device_count()
    pmap_enabled = (n_devices > 1)

    # Adjust for batch_size
    if batch_size % n_devices != 0:
        batch_size = n_devices * max(1, batch_size // n_devices)

    model = SimpleLSTM(hidden_size=hidden_size,
                       output_size=Y.shape[2], num_layers=num_layers)
    rng = jax.random.PRNGKey(0)
    rng, rng_init = jax.random.split(rng)

    state = create_train_state(
        rng_init, model, learning_rate, (batch_size, X.shape[1], X.shape[2]))

    if pmap_enabled:
        state = jax_utils.replicate(state)

    train_step = build_train_step(pmap_enabled, model)

    def one_epoch(state_rng):
        state, rng = state_rng
        rng, rng_epoch = jax.random.split(rng)

        for batch_X, batch_Y in create_batches(X, Y, batch_size, rng_epoch, n_devices):
            state = train_step(state, batch_X, batch_Y)

        return state, rng

    rng = jax.random.PRNGKey(42)

    for epoch in range(num_epochs):
        # 1 epoch training
        state, rng = one_epoch((state, rng))

        # Intermediate evaluation after this epoch
        params = jax_utils.unreplicate(
            state).params if pmap_enabled else state.params
        current_loss = evaluate_stepwise_loss(params, model, X, Y)

        if target_loss is not None and current_loss < target_loss:
            print(f"🎯 Target loss reached at epoch {epoch}: {current_loss}")
            break

        # Report intermediate value to Optuna
        if trial is not None:
            trial.report(float(current_loss), step=epoch)

            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()

    # Final evaluation
    params = jax_utils.unreplicate(
        state).params if pmap_enabled else state.params
    final_loss = evaluate_stepwise_loss(params, model, X, Y)

    if save_model:
        save_dict = {
            "params": state.params,
            "model_hyperparams": {
                "input_size": X.shape[2],
                "hidden_size": hidden_size,
                "output_size": Y.shape[2],
                "num_layers": num_layers
            }
        }
        with open('trained_model.pkl', 'wb') as f:
            pickle.dump(save_dict, f)

    return float(final_loss)  # Optuna excepts a float


def run_hpo(X, Y, n_trials=30, custom_intervals: dict = None, target_loss=None, max_epochs=None):

    hardware_info = get_hardware_info()
    search_space = get_hyperparameter_search_space(hardware_info)
    search_space.update(custom_intervals)

    def objective(trial):
        hidden_size = trial.suggest_int(
            'hidden_size', *search_space['hidden_size'])
        num_layers = trial.suggest_int(
            'num_layers', *search_space['num_layers'])
        learning_rate = trial.suggest_loguniform(
            'learning_rate', *search_space['learning_rate'])
        batch_size = trial.suggest_categorical('batch_size',
                                               [search_space['batch_size'][0],
                                                (search_space['batch_size'][0] +
                                                   search_space['batch_size'][1]) // 2,
                                                   search_space['batch_size'][1]]
                                               )
        num_epochs = trial.suggest_int(
            'num_epochs', *search_space['num_epochs'])
        # Pass the trial for pruning
        loss = train_one_model(X, Y, hidden_size, num_layers,
                               learning_rate, batch_size, num_epochs, trial=trial, target_loss=target_loss)

        return loss

    study = optuna.create_study(
        direction="minimize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials)

    print("✅ Best trial:")
    print(study.best_trial.params)

    return study


# Training loop
if __name__ == "__main__":

    agent_order = ['agent_0', 'agent_1']
    num_actions = 6

    X, Y = load_trajectories('trajectories.json', agent_order, num_actions)

    # results = run_hpo(X, Y, n_trials=100, custom_intervals={
    #                   "num_epochs": (1000, 1000)})

    # print(results.best_value)
    # print(results.best_params)

    # # Plot graphs
    # param_importances_graph = vis.plot_param_importances(results)
    # optimization_history_graph = vis.plot_optimization_history(results)

    # # Save graphs
    # param_importances_graph.write_image("hyperparameter_importance.png")
    # optimization_history_graph.write_image("optimization_history.png")

    # print("✅ HPO Importances Graphs saved as PNG file.")

    # {'hidden_size': 475, 'num_layers': 2, 'learning_rate': 0.00822465146199599, 'batch_size': 4, 'num_epochs': 1000}

    final_loss = train_one_model(
        X, Y, 500, 3, 0.00822465146199599, 1, 1000, save_model=True, target_loss=1.2e-10)
    print(f"Final loss: {final_loss}")
