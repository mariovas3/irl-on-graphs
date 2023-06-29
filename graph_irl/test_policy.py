import policy
from tqdm import tqdm
import numpy as np
import pickle
import random
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from vis_utils import ci_plot
from pathlib import Path

TEST_OUTPUTS_PATH = Path(".").absolute().parent / "test_output"
if not TEST_OUTPUTS_PATH.exists():
    TEST_OUTPUTS_PATH.mkdir()


ENV_NAME = "Hopper-v2"

configs = {
    "Hopper-v2": {"hiddens": [26, 26]},
    "MountainCarContinuous-v0": {"hiddens": [4, 4]},
}

env = gym.make(ENV_NAME)
num_episodes = 2000
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
seeds = range(5)
lr, discount = 1e-3, 0.99
T = 200


def get_returns_over_seeds(
    env,
    agent_make,
    seeds,
    T,
    num_episodes,
    save_pkl_to: Path = None,
    **agent_kwargs,
):
    returns_over_seeds = []
    for seed in tqdm(seeds):
        # fix all seeds;
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        agent = agent_make(**agent_kwargs)
        returns_over_episodes = []

        for ep in tqdm(range(num_episodes)):
            # start env;
            obs, info = env.reset(seed=seed)
            done = False
            t = 0
            while not done and t < T:
                # sample action;
                action = agent.sample_action(torch.FloatTensor(obs).view(-1))
                # apply action;
                obs, reward, terminated, truncated, info = env.step(
                    action.numpy()
                )
                # update rewards and baseline;
                agent.update_rewards(reward)
                agent.update_baseline()
                done = truncated or terminated
                # t += 1  # let episode terminate or truncate;
            # sum the rewards over the episode;
            returns_over_episodes.append(np.sum(agent.rewards))
            # update params of policy;
            agent.update()
            if (ep + 1) % 100 == 0:
                print(
                    f"Episode {ep + 1} undiscounted return: {returns_over_episodes[-1]:.3f}"
                )
        # add episodic undiscounted returns to returns over seeds;
        returns_over_seeds.append(returns_over_episodes)
        # save episodic returns to file for later use;
        if save_pkl_to:
            file_name = agent.name + f"-{env.spec.id}-seed-{seed}.pkl"
            file_name = save_pkl_to / file_name
            with open(file_name, "wb") as f:
                pickle.dump(returns_over_episodes, f)
    return returns_over_seeds


def test_seeds_mcc():
    agent_kwargs = {
        "name": f"PGGauss-b1-lr-{lr}",
        "obs_dim": state_dim,
        "action_dim": action_dim,
        "policy": policy.GaussPolicy,
        "with_baseline": True,
        "lr": 1e-4,
        "discount": 0.99,
        "hiddens": configs[ENV_NAME]["hiddens"],
        "with_layer_norm": False,
    }
    png_file_name = agent_kwargs['name'] + f'-{env.spec.id}.png'
    png_file_name = TEST_OUTPUTS_PATH / png_file_name
    returns_over_seeds = get_returns_over_seeds(
        env,
        policy.PGGauss,
        seeds,
        T,
        num_episodes,
        save_pkl_to=TEST_OUTPUTS_PATH,
        **agent_kwargs,
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    avg_returns = np.mean(returns_over_seeds, 0)
    for s, r in zip(seeds, returns_over_seeds):
        ax.plot(r, label=s)
    ax.plot(avg_returns, label="avg returns")
    plt.ylabel("episodic undiscounted return")
    plt.xlabel("episode")
    plt.title(f"{agent_kwargs['name']}")
    plt.legend()
    fig.tight_layout()
    plt.savefig(png_file_name)
