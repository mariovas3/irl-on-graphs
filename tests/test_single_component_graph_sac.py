import sys
from pathlib import Path

p = Path(__file__).absolute().parent.parent
sys.path.append(str(p))

from graph_irl.sac import *
from graph_irl.graph_rl_utils import GraphEnv
from graph_irl.buffer_v2 import GraphBuffer
from graph_irl.vis_utils import vis_graph_building

from torch_geometric.data import Data

import random
import numpy as np
import torch

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


class SingleComponentGraphReward:
    def __init__(self, n_nodes):
        # set input attributes;
        self.n_nodes = n_nodes
        
        # initialise attributes;
        self.edge_bonus = 5.
        self.component_reduction_bonus = 2.
        self.degrees = np.zeros(n_nodes, dtype=np.int64)
        self.adj_list = [[] for _ in range(n_nodes)]
        self.edge_set = set()
        self.n_components_last = n_nodes
        self.degree_gt3 = False
        self.should_terminate = False
        self.sum_degrees = 0
        self._verbose = False
    
    def verbose(self):
        self._verbose = True
    
    def reset(self):
        self.component_reduction_bonus = 2.
        self.degrees = np.zeros(self.n_nodes, dtype=np.int64)
        self.adj_list = [[] for _ in range(self.n_nodes)]
        self.edge_set = set()
        self.n_components_last = self.n_nodes
        self.degree_gt3 = False
        self.should_terminate = False
        self.sum_degrees = 0
        self._verbose = False

    def _dfs(self, i, visited):
        if not visited[i]:
            visited[i] = 1
            for nei in self.adj_list[i]:
                self._dfs(nei, visited)
    
    def _count_graph_components(self):
        visited = np.zeros(self.n_nodes, dtype=np.int8)
        n_components = 0
        for i in range(self.n_nodes):
            if not visited[i]:
                n_components += 1
                self._dfs(i, visited)
        return n_components
    
    def __call__(self, data: Data, first: int, second: int):
        assert not self.should_terminate
        # if self-loop reward is -100;
        if first == second:
            self.should_terminate = True
            if self._verbose:
                print("self-loop")
            return - 100
        
        # keep edges in ascending order in indexes;
        if first > second:
            first, second = second, first
        
        # if the edge is a repeat, reward is -100;
        if (first, second) in self.edge_set:
            self.should_terminate = True
            if self._verbose:
                print("repeated edge")
            return - 100
        
        # increment degrees of relevant nodes from new edge;
        self.degrees[first] += 1
        self.degrees[second] += 1
        self.sum_degrees += 2

        # add new edge to edge set;
        self.edge_set.add((first, second))

        # update adjacency list;
        self.adj_list[first].append(second)
        self.adj_list[second].append(first)

        n_comp_old = self.n_components_last
        self.n_components_last = self._count_graph_components()
        
        component_bonus = (n_comp_old - self.n_components_last) * self.component_reduction_bonus
        
        # if number of components decreased,
        # increase component reduction bonus for next time;
        if component_bonus > 0:
            self.component_reduction_bonus += 2.
        
        reward = component_bonus
        if self._verbose:
            print("add edge")
        return reward


if __name__ == "__main__":
    n_nodes = 10
    encoder_hiddens = [16, 16, 2]
    config = dict(
        buffer_kwargs=dict(
            max_size=10_000,
            nodes='identity',
        ),
        general_kwargs=dict(
            buffer_len=10_000,
            n_nodes=n_nodes,
            nodes='identity',
            num_steps_to_sample=5 * n_nodes,
            min_steps_to_presample=5 * n_nodes,
            batch_size=min(10 * n_nodes, 100),
            num_iters=100,
            num_epochs=10,
            num_grad_steps=1,
            seed=0,
            discount=0.99,
            tau=0.005,
            UT_trick=False,
            with_entropy=False,
            for_graph=True,
        ),
        env_kwargs=dict(
            x='identity',
            reward_fn='circle_reward',
            max_episode_steps=n_nodes,
            num_expert_steps=n_nodes,
            max_repeats=1000,
            max_self_loops=1000,
            drop_repeats_or_self_loops=False,
            reward_fn_termination=True,
        ),
        encoder_kwargs=dict(
            encoder_hiddens=encoder_hiddens,
            with_layer_norm=False,
            final_tanh=True,
        ),
        gauss_policy_kwargs= dict(
            obs_dim=encoder_hiddens[-1],
            action_dim=encoder_hiddens[-1],
            hiddens=[256, 256],
            with_layer_norm=True,
            encoder="GCN",
            two_action_vectors=True,
        ),
        tsg_policy_kwargs=dict(
            obs_dim=encoder_hiddens[-1],
            action_dim=encoder_hiddens[-1],
            hiddens1=[256, 256],
            hiddens2=[256,],
            encoder="GCN",
            with_layer_norm=True,
        ),
        agent_kwargs=dict(
            name="SACAgentGraph",
            policy='TwoStageGaussPolicy',
            policy_lr=3e-4,
            entropy_lb=encoder_hiddens[-1],
            temperature_lr=3e-4,
        ),
        qfunc_kwargs=dict(
            qfunc_hiddens=[256, 256],
            qfunc_layer_norm=True,
            qfunc_lr=3e-4,
            qfunc1_encoder='GCN',
            qfunc2_encoder='GCN',
            qfunc1t_encoder='GCN',
            qfunc2t_encoder='GCN',
        ),
    )

    # see if a path to a config file was supplied;
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'rb') as f:
            config = pickle.load(f)
            if 'max_self_loops' not in config['env_kwargs']:
                config['env_kwargs']['max_self_loops'] = int(1e8)
    
    # create the node features;
    nodes = torch.eye(config['general_kwargs']['n_nodes'])
    nodes = torch.randn((n_nodes, n_nodes))

    # instantiate buffer;
    buffer = GraphBuffer(
        max_size=config['buffer_kwargs']['max_size'],
        nodes=nodes,
        seed=config['general_kwargs']['seed'],
    )

    # instantiate reward;
    reward_fn = SingleComponentGraphReward(config['general_kwargs']['n_nodes'])

    # env setup;
    env = GraphEnv(
        x=nodes,
        reward_fn=reward_fn,
        max_episode_steps=config['general_kwargs']['n_nodes'],
        num_expert_steps=config['general_kwargs']['n_nodes'],
        max_repeats=config['env_kwargs']['max_repeats'],
        max_self_loops=config['env_kwargs']['max_self_loops'],
        drop_repeats_or_self_loops=config['env_kwargs']['drop_repeats_or_self_loops'],
        reward_fn_termination=config['env_kwargs']['reward_fn_termination'],
    )
    print(env.spec.id)

    # encoder setup;
    encoder = GCN(nodes.shape[-1],        
                config['encoder_kwargs']['encoder_hiddens'],  
                with_layer_norm=config['encoder_kwargs']['with_layer_norm'],
                final_tanh=config['encoder_kwargs']['final_tanh'],
            )
    qfunc1_encoder = GCN(nodes.shape[-1], 
                config['encoder_kwargs']['encoder_hiddens'],  
                with_layer_norm=config['encoder_kwargs']['with_layer_norm'],
                final_tanh=config['encoder_kwargs']['final_tanh'],
            )
    qfunc2_encoder = GCN(nodes.shape[-1], 
                config['encoder_kwargs']['encoder_hiddens'],  
                with_layer_norm=config['encoder_kwargs']['with_layer_norm'],
                final_tanh=config['encoder_kwargs']['final_tanh'],
            )
    qfunc1t_encoder = GCN(nodes.shape[-1],
                 config['encoder_kwargs']['encoder_hiddens'], 
                 with_layer_norm=config['encoder_kwargs']['with_layer_norm'],
                 final_tanh=config['encoder_kwargs']['final_tanh'],
            )
    qfunc2t_encoder = GCN(nodes.shape[-1],
                 config['encoder_kwargs']['encoder_hiddens'], 
                 with_layer_norm=config['encoder_kwargs']['with_layer_norm'],
                 final_tanh=config['encoder_kwargs']['final_tanh'],
            )

    # policy setup for GaussPolicy like GraphOpt;
    gauss_policy_kwargs = config['gauss_policy_kwargs'].copy()
    gauss_policy_kwargs['encoder'] = encoder

    # policy setup for TwoStageGaussPolicy;
    tsg_policy_kwargs = config['tsg_policy_kwargs'].copy()
    tsg_policy_kwargs['encoder'] = encoder

    # setup agent;
    agent_kwargs = config['agent_kwargs'].copy()
    agent_kwargs['policy'] = TwoStageGaussPolicy
    agent_policy_kwargs = dict(
        agent_kwargs=agent_kwargs,
        policy_kwargs=tsg_policy_kwargs,
    )
    
    Q1, Q2, agent = train_sac(
        env=env,
        agent=SACAgentGraph,
        num_iters=config['general_kwargs']['num_iters'],
        qfunc_hiddens=config['qfunc_kwargs']['qfunc_hiddens'],
        qfunc_layer_norm=config['qfunc_kwargs']['qfunc_layer_norm'],
        qfunc_lr=config['qfunc_kwargs']['qfunc_lr'],
        buffer_len=config['buffer_kwargs']['max_size'],
        batch_size=config['general_kwargs']['batch_size'],
        discount=config['general_kwargs']['discount'],
        tau=config['general_kwargs']['tau'],
        seed=config['general_kwargs']['seed'],
        save_returns_to=TEST_OUTPUTS_PATH,
        num_steps_to_sample=config['general_kwargs']['num_steps_to_sample'],
        num_eval_steps_to_sample=config['general_kwargs']['num_steps_to_sample'],
        num_grad_steps=config['general_kwargs']['num_grad_steps'],
        num_epochs=config['general_kwargs']['num_epochs'],
        min_steps_to_presample=config['general_kwargs']['min_steps_to_presample'],
        UT_trick=config['general_kwargs']['UT_trick'],
        with_entropy=config['general_kwargs']['with_entropy'],
        for_graph=config['general_kwargs']['for_graph'], 
        qfunc1_encoder=qfunc1_encoder,
        qfunc2_encoder=qfunc2_encoder,
        qfunc1t_encoder=qfunc1_encoder,
        qfunc2t_encoder=qfunc2t_encoder,
        buffer_instance=buffer,
        config=config,
        **agent_policy_kwargs,
    )

    obs, _, rewards, code = sample_eval_path_graph(n_nodes, env, agent, seed=0, verbose=True)
    print(code, obs[-1].edge_index.tolist(), rewards, sep='\n')
    vis_graph_building(obs[-1].edge_index.tolist())
