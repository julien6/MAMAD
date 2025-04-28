import time
from typing import Dict, List, Tuple
import jax
import json
import jax.numpy as jnp
import optax
import pickle
import optuna

from tqdm import trange, tqdm
from flax import linen as nn
from flax.training import train_state
from prepare_data import infer_metadata_from_json


intervals = {
    "hidden_size": (64, 512),
    "num_layers": (1, 3),
    "learning_rate": (1e-5, 1e-2),
    "batch_size": (64, 64),
    "num_epochs": (5, 5)
}


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
        intervals["num_epochs"] = (5, 20)

    return intervals


metadata = infer_metadata_from_json("trajectories.json")

intervals.update(get_search_intervals(metadata, [
                 "hidden_size", "num_layers", "learning_rate", "batch_size"]))


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

            inputs.append(input_vec)
            targets.append(target_vec)

    X = jnp.array(inputs)
    Y = jnp.array(targets)
    return X, Y


agent_order = ['agent_0', 'agent_1']
num_actions = 6

X, Y = load_trajectories('trajectories.json', agent_order, num_actions)

output_size = Y.shape[1]
input_size = X.shape[1]


class LSTMCell(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, carry, x):
        lstm_cell = nn.OptimizedLSTMCell()
        return lstm_cell(carry, x)


class LSTMModel(nn.Module):
    hidden_size: int
    output_size: int
    num_layers: int = 1

    @nn.compact
    def __call__(self, x):
        batch_size = x.shape[0]

        carry_list = []
        y = x  # initial input

        rng = jax.random.PRNGKey(0)
        rngs = jax.random.split(rng, self.num_layers)

        # Building stacked LSTM
        for layer_idx in range(self.num_layers):
            lstm_layer = nn.scan(
                LSTMCell,
                variable_broadcast="params",
                split_rngs={"params": False},
                in_axes=1,
                out_axes=1
            )(self.hidden_size)

            carry = nn.OptimizedLSTMCell.initialize_carry(
                rngs[layer_idx], (batch_size,), self.hidden_size)  # Use different rng for each layer

            carry, y = lstm_layer(carry, y)
            carry_list.append(carry)

        last_hidden = y[:, -1]

        output = nn.Dense(self.output_size)(last_hidden)

        return output


def create_train_state(rng, model, learning_rate, input_shape):
    params = model.init(rng, jnp.ones(input_shape))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def single_loss_fn(params, apply_fn, x, y):
    pred = apply_fn({'params': params}, x)
    loss = jnp.mean((pred[0] - y) ** 2)
    return loss


batched_loss_fn = jax.vmap(single_loss_fn, in_axes=(None, None, 0, 0))


@jax.jit
def train_step(state, batch):
    inputs, targets = batch
    # Compute the loss for every batch of a coup
    losses = batched_loss_fn(state.params, state.apply_fn, inputs, targets)
    mean_loss = jnp.mean(losses)
    grads = jax.grad(lambda params: mean_loss)(state.params)
    return state.apply_gradients(grads=grads), mean_loss


def create_batches(X, Y, batch_size, rng=None):
    n_samples = X.shape[0]
    indices = jnp.arange(n_samples)
    if rng is not None:
        indices = jax.random.permutation(
            rng, indices)  # shuffle if rng is given
    X = X[indices]
    Y = Y[indices]

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        yield X[start_idx:end_idx], Y[start_idx:end_idx]


def train_model(state, X, Y, batch_size, num_epochs, rng):
    total_time = 0.0
    total_samples = 0

    for epoch in trange(num_epochs, desc="Training epochs"):
        start_time = time.time()
        epoch_rng = jax.random.fold_in(rng, epoch)

        for batch_X, batch_Y in create_batches(X, Y, batch_size, epoch_rng):
            state, batch_loss = train_step(state, (batch_X, batch_Y))
            total_samples += batch_X.shape[0]

        epoch_time = time.time() - start_time
        total_time += epoch_time
        fps = total_samples / total_time

        print(
            f"[Epoch {epoch+1}/{num_epochs}] Time: {epoch_time:.2f}s | Samples seen: {total_samples} | Avg FPS: {fps:.2f}")

    print(f"\n✅ Training completed in {total_time:.2f} seconds total.")
    print(f"✅ Average samples/sec (FPS): {fps:.2f}")

    return state


@jax.pmap
def train_step_pmap(state, batch):
    grads = jax.grad(batched_loss_fn)(state.params, state.apply_fn, batch)
    return state.apply_gradients(grads=grads)


@jax.jit
def evaluate_model(state, inputs, targets):
    losses = batched_loss_fn(state.params, state.apply_fn, inputs, targets)
    return jnp.mean(losses)


def batched_evaluate_model(state, inputs, targets, batch_size):
    total_loss = 0.0
    total_samples = 0

    for batch_inputs, batch_targets in create_batches(inputs, targets, batch_size):
        loss = evaluate_model(state, batch_inputs, batch_targets)
        total_loss += loss * batch_inputs.shape[0]
        total_samples += batch_inputs.shape[0]

    avg_loss = total_loss / total_samples
    return avg_loss


def objective(trial):
    # 1. Suggérer des hyperparamètres
    hidden_size = trial.suggest_int(
        'hidden_size', intervals["hidden_size"][0], intervals["hidden_size"][1])
    learning_rate = trial.suggest_loguniform(
        'learning_rate', intervals["learning_rate"][0], intervals["learning_rate"][1])
    num_epochs = trial.suggest_int(
        'num_epochs', intervals["num_epochs"][0], intervals["num_epochs"][1])
    num_layers = trial.suggest_int(
        'num_layers', intervals["num_layers"][0], intervals["num_layers"][1])
    batch_size = trial.suggest_int(
        'batch_size', intervals["batch_size"][0], intervals["batch_size"][1])

    # 2. Initialize model and state
    model = LSTMModel(hidden_size=hidden_size,
                      output_size=output_size, num_layers=num_layers)
    rng = jax.random.PRNGKey(0)
    state = create_train_state(
        rng, model, learning_rate, (X_seq.shape[0], 1, X_seq.shape[2]))

    # 3. Train
    state = train_model(state, X_seq, Y, batch_size, num_epochs, rng)

    # 4. Evaluate
    eval_loss = batched_evaluate_model(state, X_seq, Y, batch_size)

    return float(eval_loss)


if __name__ == '__main__':

    # 1 - Run HPO

    # Adapter X pour simuler une séquence de longueur 1 :
    X_seq = X[:, None, :]  # (batch_size, 1, input_dim)

    # Lancer l'optimisation
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=2, show_progress_bar=True)

    print("Best hyperparameters found:")
    print(study.best_params)
    print(f"Best evaluation loss: {study.best_value}")

    # 2 - Create the model with best hyperparameters for further training

    model = LSTMModel(
        hidden_size=study.best_params["hidden_size"], output_size=output_size, num_layers=study.best_params["num_layers"])

    rng = jax.random.PRNGKey(0)
    # batch_size, input_dim
    state = create_train_state(
        rng, model, study.best_params["learning_rate"], (1, X.shape[0], X.shape[1]))

    state = train_model(
        state, X_seq, Y, study.best_params["batch_size"], study.best_params["num_epochs"], rng)

    save_dict = {
        "params": state.params,
        "model_hyperparams": {
            "hidden_size": study.best_params["hidden_size"],
            "output_size": output_size,
            "num_layers": study.best_params["num_layers"]
        }
    }
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(save_dict, f)

    # Evaluate the model
    eval_loss = batched_evaluate_model(
        state, X_seq, Y, study.best_params["batch_size"])

    print(f"Evaluation Loss on training set: {eval_loss:.6f}")
