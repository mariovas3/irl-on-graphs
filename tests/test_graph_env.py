import sys
from pathlib import Path

p = Path(__file__).absolute().parent.parent
sys.path.append(str(p))

from graph_irl.graph_rl_utils import GraphEnv
from graph_irl.policy import *
from graph_irl.buffer_v2 import GraphBuffer, sample_eval_path_graph
from graph_irl.sac import *

import random
import numpy as np
import torch
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


if __name__ == "__main__":
    n_nodes = 10
    x = torch.eye(n_nodes)  # 10 nodes;

    encoder_hiddens = [20, 20]
    encoder = GCN(x.shape[-1], encoder_hiddens, with_layer_norm=True)

    # define reward fn
    def reward_fn(data, first, second):
        diff = abs(first - second)
        if diff != 1:
            return -100
        return 1

    # env setup:
    max_episode_steps = 80
    num_expert_steps = 16
    max_repeats = 50
    env = GraphEnv(
        x,
        reward_fn,
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

    # policy setup for GaussPolicy like GraphOpt;
    gauss_policy_kwargs = {
        'obs_dim': encoder_hiddens[-1],
        'action_dim': encoder_hiddens[-1],
        'hiddens': [30, 30],
        'with_layer_norm': True,
        'encoder': encoder,
        'two_action_vectors': True,
    }
    
    # make agent;
    agent = SACAgentGraph(
        name="SACAgentGraph",
        policy=GaussPolicy,
        policy_lr=3e-4,
        entropy_lb=encoder_hiddens[-1],
        temperature_lr=3e-4,
        **gauss_policy_kwargs,
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

    # collect eval path;
    obs, actions, rewards, code = sample_eval_path_graph(100, env, agent, 0)
    print("eval path:", len(rewards), sep='\n')
    for g in obs:
        print(g)

    # sample batch;
    obs_t, action_t, reward_t, obs_tp1, terminated_tp1 = buffer.sample(
        batch_size
    )

    # test policy loss and temperature loss
    # and q func losses;
    encoder1 = GCN(x.shape[-1], encoder_hiddens, with_layer_norm=True)
    encoder2 = GCN(x.shape[-1], encoder_hiddens, with_layer_norm=True)
    Q1 = Qfunc(3 * encoder_hiddens[-1], [31, 31], with_layer_norm=True, encoder=encoder1)
    Q2 = Qfunc(3 * encoder_hiddens[-1], [31, 31], with_layer_norm=True, encoder=encoder2)
    Q1t = Qfunc(3 * encoder_hiddens[-1], [31, 31], with_layer_norm=True)
    Q2t = Qfunc(3 * encoder_hiddens[-1], [31, 31], with_layer_norm=True)


    # test policy loss and temperature loss;
    agent.get_policy_loss_and_temperature_loss(
        obs_t, Q1, Q2, UT_trick=False, with_entropy=False
    )
    print(agent.policy_loss, agent.temperature_loss, sep='\n', end='\n\n')
    
    l1, l2 = get_q_losses(
        Q1, Q2, Q1t, Q2t,
        obs_t, action_t, reward_t, obs_tp1, terminated_tp1,
        agent, discount=.99, UT_trick=False, with_entropy=True, 
        for_graph=True,
    )
    print(l1, l2, sep='\n', end='\n\n')
