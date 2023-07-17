"""
TODO:
    (1) Finish this file by essentially testing the
        functionality of the GraphBuffer class that 
        should be written in buffer_v2 or some other place.
"""
import sys
from pathlib import Path

p = Path(__file__).absolute().parent.parent
sys.path.append(str(p))

from graph_irl.graph_rl_utils import GraphEnv
from graph_irl.policy import TwoStageGaussPolicy, GCN
from graph_irl.buffer_v2 import GraphBuffer, sample_eval_path_graph
from graph_irl.sac import SACAgentGraph

import torch


if __name__ == "__main__":
    n_nodes = 10
    x = torch.eye(n_nodes)  # 10 nodes;

    encoder_hiddens = [20, 20]
    encoder = GCN(x.shape[-1], encoder_hiddens, with_layer_norm=True)

    # env setup:
    max_episode_steps = 80
    num_expert_steps = 16
    max_repeats = 50
    env = GraphEnv(
        x,
        lambda data: data.x.mean(),
        max_episode_steps,
        num_expert_steps,
        max_repeats,
        drop_repeats_or_self_loops=False,
    )

    # policy setup for TwoStageGaussPolicy;
    policy_kwargs = {
        "obs_dim": encoder_hiddens[-1],
        "action_dim": encoder_hiddens[-1],
        "hiddens1": [30, 30],
        "hiddens2": [40, 40],
        "encoder": encoder,
        "with_layer_norm": True,
    }

    # make agent;
    agent = SACAgentGraph(
        name="SACAgentGraph",
        policy=TwoStageGaussPolicy,
        policy_lr=3e-4,
        entropy_lb=encoder_hiddens[-1],
        temperature_lr=3e-4,
        **policy_kwargs,
    )

    # replay buffer params;
    max_size = 1_000
    num_steps_to_collect = 200
    batch_size = 100
    buffer = GraphBuffer(max_size, x, seed=0)

    # collect graph path;
    buffer.collect_path(env, agent, num_steps_to_collect)
    for graph in buffer.obs_t[:17]:
        print(graph)
    print(buffer.terminal_tp1[:17])

    obs_t, action_t, reward_t, obs_tp1, terminated_tp1 = buffer.sample(
        batch_size
    )
