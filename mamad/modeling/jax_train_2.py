import json
import pickle
import jax
import jax.numpy as jnp
import optax

from flax import linen as nn
from flax.training import train_state
from tqdm import trange
from flax.training import common_utils
from flax.training import checkpoints
from flax import jax_utils

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
    def __call__(self, x):
        batch_size = x.shape[0]

        # Initialize carry for each layer
        carry = [
            nn.OptimizedLSTMCell.initialize_carry(
                jax.random.PRNGKey(i), (batch_size,), self.hidden_size
            )
            for i in range(self.num_layers)
        ]

        carry, y = self.stacked_lstm(carry, x)
        preds = self.dense(y)
        return preds


# Create a train state


def create_train_state(rng, model, learning_rate, input_shape):
    params = model.init(rng, jnp.ones(input_shape))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# Loss function comparing prediction to target at every timestep


def single_loss_fn(params, apply_fn, input_sample, target_sample):
    # [None, ...] to fake batch axis
    preds = apply_fn({'params': params}, input_sample[None, ...])
    preds = preds[0]  # remove fake batch axis
    loss = jnp.mean((preds - target_sample) ** 2)
    return loss


batched_loss_fn = jax.vmap(single_loss_fn, in_axes=(None, None, 0, 0))


# Single training step
@jax.jit
def train_step(state, batch_inputs, batch_targets):
    def loss(params):
        losses = batched_loss_fn(
            params, state.apply_fn, batch_inputs, batch_targets)
        return jnp.mean(losses)  # average over the batch

    grads = jax.grad(loss)(state.params)
    return state.apply_gradients(grads=grads)


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


def create_batches(X, Y, batch_size, rng=None):
    """Splits the full dataset (trajectories) into batches of complete episodes."""
    n_episodes = X.shape[0]
    indices = jnp.arange(n_episodes)
    if rng is not None:
        indices = jax.random.permutation(rng, indices)  # optional shuffle

    X = X[indices]
    Y = Y[indices]

    for start_idx in range(0, n_episodes, batch_size):
        end_idx = min(start_idx + batch_size, n_episodes)
        yield X[start_idx:end_idx], Y[start_idx:end_idx]


def one_epoch(epoch_idx, state_rng):
    state, rng = state_rng

    rng, rng_epoch = jax.random.split(rng)
    for batch_X, batch_Y in create_batches(X, Y, batch_size, rng_epoch):
        state = train_step(state, batch_X, batch_Y)

    return state, rng


# Training loop
if __name__ == "__main__":

    agent_order = ['agent_0', 'agent_1']
    num_actions = 6

    X, Y = load_trajectories('trajectories.json', agent_order, num_actions)

    # Config
    seq_len = X.shape[1]         # transitions in an episode (2)
    input_dim = X.shape[2]       # observation + action (204)
    output_dim = Y.shape[2]      # next observation (196)
    hidden_size = 256
    learning_rate = 1e-3
    num_epochs = 20

    batch_size = 1  # number of episodes per batch

    # Create model and state
    model = SimpleLSTM(hidden_size=hidden_size,
                       output_size=output_dim, num_layers=2)

    rng = jax.random.PRNGKey(42)
    rng, rng_init = jax.random.split(rng)

    state = create_train_state(
        rng_init, model, learning_rate, (batch_size, seq_len, input_dim))

    # Train over multiple epochs
    # for epoch in trange(num_epochs, desc="Training epochs"):
    #     rng, rng_epoch = jax.random.split(rng)
    #     for batch_X, batch_Y in create_batches(X, Y, batch_size, rng_epoch):
    #         state = train_step(state, batch_X, batch_Y)
    state, rng = jax.lax.fori_loop(0, num_epochs, one_epoch, (state, rng))

    print("\n✅ Training finished successfully.")

    save_dict = {
        "params": state.params,
        "model_config": {
            "hidden_size": hidden_size,
            "output_size": output_dim,
            "num_layers": 2
        }
    }

    with open("trained_lstm.pkl", "wb") as f:
        pickle.dump(save_dict, f)

    print("✅ Model saved.")
