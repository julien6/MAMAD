import os
import gym
import random
import pickle
import numpy as np
from tqdm import trange
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import RandomAgent
import pygame

# ---------- Setup environment ----------
layout_name = "asymmetric_advantages"
mdp = OvercookedGridworld.from_layout_name(layout_name)
base_env = OvercookedEnv.from_mdp(mdp, horizon=200)
env = gym.make("Overcooked-v0", base_env=base_env,
               featurize_fn=base_env.featurize_state_mdp)

agent1 = RandomAgent(all_actions=True)
agent2 = RandomAgent(all_actions=True)

output_dir = "./overcooked_frames"
os.makedirs(output_dir, exist_ok=True)

# ---------- Collect data ----------


def collect_visual_data(env, agent1, agent2, num_episodes=100, episode_length=200, save_every=10, output_dir="./overcooked_frames"):
    all_data = []

    for ep in trange(num_episodes, desc="Collecting"):
        obs = env.reset()["both_agent_obs"]
        frame = env.render(mode="rgb_array")  # Initial frame
        traj = []

        for t in range(episode_length):
            a1 = random.randint(0, 5)  # agent1.action(obs[0])
            a2 = random.randint(0, 5)  # agent2.action(obs[1])
            joint_action = [a1, a2]

            next_obs, _, done, _ = env.step(joint_action)
            next_frame = env.render(mode="rgb_array")

            traj.append({
                "frame_t": frame,
                "actions_joint_t": joint_action,
                "frame_t_plus_1": next_frame
            })

            frame = next_frame
            obs = next_obs["both_agent_obs"]

            if done:
                break

        all_data.extend(traj)

        if (ep + 1) % save_every == 0:
            with open(os.path.join(output_dir, f"frames_ep{ep+1}.pkl"), "wb") as f:
                pickle.dump(traj, f)
            print(f"Saved episode {ep+1} to disk.")

    return all_data


if __name__ == "__main__":
    data = collect_visual_data(env, agent1, agent2, num_episodes=100)
