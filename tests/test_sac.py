import sys
from pathlib import Path

p = Path(__file__).absolute().parent.parent
sys.path.append(str(p))

from graph_irl.sac import train_sac, TEST_OUTPUTS_PATH, SACAgentMuJoCo
from graph_irl.policy import GaussPolicy
from graph_irl.vis_utils import see_one_episode
import gymnasium as gym

import random
import numpy as np
import torch

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


if __name__ == "__main__":
    print(TEST_OUTPUTS_PATH)

    # path to save logs;
    if not TEST_OUTPUTS_PATH.exists():
        TEST_OUTPUTS_PATH.mkdir()

    T = 3000  # max path length;
    num_epochs = 1
    num_iters = 100  # iterations per epoch;

    env = gym.make("Ant-v2", max_episode_steps=T)

    agent_policy_kwargs = {
        "agent_kwargs": {
            "name": "SACAgentMuJoCo",
            "policy": GaussPolicy,
            "policy_lr": 3e-4,
            "entropy_lb": None,
            "temperature_lr": 3e-4,
        },
        "policy_kwargs": {
            "obs_dim": env.observation_space.shape[0],
            "action_dim": env.action_space.shape[0],
            "hiddens": [256, 256],
            "with_layer_norm": True,
        },
    }

    Q1, Q2, agent = train_sac(
        env,
        SACAgentMuJoCo,
        num_iters,
        qfunc_hiddens=[256, 256],
        qfunc_layer_norm=True,
        qfunc_lr=3e-4,
        buffer_len=int(1e5),
        batch_size=250,
        discount=0.99,
        tau=0.1,
        seed=0,
        save_returns_to=TEST_OUTPUTS_PATH,
        num_steps_to_sample=2 * T,
        num_eval_steps_to_sample=T,
        num_grad_steps=500,
        num_epochs=num_epochs,
        min_steps_to_presample=T,
        UT_trick=False,
        with_entropy=False,
        for_graph=False,
        **agent_policy_kwargs,
    )

    print(f"temperature is: {agent.log_temperature.exp()}")

    env = gym.make(
        "Ant-v2", max_episode_steps=max(T, 5000), render_mode="human"
    )
    see_one_episode(env, agent, seed=0, save_to=TEST_OUTPUTS_PATH / "bob.pkl")
