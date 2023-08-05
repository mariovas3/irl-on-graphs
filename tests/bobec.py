import sys
from pathlib import Path

p = Path(__file__).absolute().parent.parent
sys.path.append(str(p))

from graph_irl.sac import *
from graph_irl.graph_rl_utils import GraphEnv
from graph_irl.buffer_v2 import GraphBuffer

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
        self.edge_bonus = 5.0
        self.component_reduction_bonus = 2.0
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
        self.component_reduction_bonus = 2.0
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

    def __call__(self, obs_action, action_is_index=False):
        """The action_is_index is here only for compatibility reasons."""
        assert not self.should_terminate

        # idxs should be (1, 2) shape torch.tensor;
        data, idxs = obs_action
        first, second = idxs.view(-1).tolist()

        # if self-loop reward is -100;
        if first == second:
            self.should_terminate = True
            if self._verbose:
                print("self-loop")
            return -100

        # keep edges in ascending order in indexes;
        if first > second:
            first, second = second, first

        # if the edge is a repeat, reward is -100;
        if (first, second) in self.edge_set:
            self.should_terminate = True
            if self._verbose:
                print("repeated edge")
            return -100

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

        component_bonus = (
            n_comp_old - self.n_components_last
        ) * self.component_reduction_bonus

        # if number of components decreased,
        # increase component reduction bonus for next time;
        if component_bonus > 0:
            self.component_reduction_bonus += 2.0

        reward = component_bonus
        if self._verbose:
            print("add edge")
        return reward


n_nodes = 10
encoder_hiddens = [16, 16, 2]
config = dict(
    buffer_kwargs=dict(
        max_size=10_000,
        nodes="gaussian",
    ),
    training_kwargs=dict(
        seed=0,
        num_iters=100,
        num_steps_to_sample=5 * n_nodes,
        num_grad_steps=1,
        batch_size=min(10 * n_nodes, 100),
        num_eval_steps_to_sample=5 * n_nodes,
        min_steps_to_presample=5 * n_nodes,
    ),
    extra_info_kwargs=dict(
        n_nodes=n_nodes,
        nodes="gaussian",
        num_epochs=10,
        UT_trick=False,
        with_entropy=False,
        for_graph=True,
        vis_graph=True,
    ),
    env_kwargs=dict(
        nodes="gaussian",
        reward_fn="SingleComponentGraphReward",
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
    gauss_policy_kwargs=dict(
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
        hiddens2=[256],
        encoder="GCN",
        with_layer_norm=True,
    ),
    agent_kwargs=dict(
        name="SACAgentGraph",
        policy_constructor="TwoStageGaussPolicy",
        qfunc_constructor="Qfunc",
        env_constructor="GraphEnv",
        buffer_constructor="GraphBuffer",
        entropy_lb=encoder_hiddens[-1],
        policy_lr=3e-4,
        temperature_lr=3e-4,
        qfunc_lr=3e-4,
        tau=0.005,
        discount=0.99,
        save_to=str(TEST_OUTPUTS_PATH),
    ),
    qfunc_kwargs=dict(
        obs_action_dim=3 * encoder_hiddens[-1],
        hiddens=[256, 256],
        with_layer_norm=True,
        encoder="GCN",
    ),
)

constructors = {
    "TwoStageGaussPolicy": TwoStageGaussPolicy,
    "GaussPolicy": GaussPolicy,
    "Qfunc": Qfunc,
    "GraphEnv": GraphEnv,
    "GraphBuffer": GraphBuffer,
}


if __name__ == "__main__":
    # see if a path to a config file was supplied;
    if len(sys.argv) > 1:
        with open(sys.argv[1], "rb") as f:
            config = pickle.load(f)
            if "max_self_loops" not in config["env_kwargs"]:
                config["env_kwargs"]["max_self_loops"] = int(1e8)

    # create the node features;
    nodes = torch.randn((n_nodes, n_nodes))

    # instantiate reward;
    reward_fn = SingleComponentGraphReward(
        config["extra_info_kwargs"]["n_nodes"]
    )

    # encoder setup;
    encoder = GCN(
        nodes.shape[-1],
        config["encoder_kwargs"]["encoder_hiddens"],
        with_layer_norm=config["encoder_kwargs"]["with_layer_norm"],
        final_tanh=config["encoder_kwargs"]["final_tanh"],
    )
    qfunc1_encoder = GCN(
        nodes.shape[-1],
        config["encoder_kwargs"]["encoder_hiddens"],
        with_layer_norm=config["encoder_kwargs"]["with_layer_norm"],
        final_tanh=config["encoder_kwargs"]["final_tanh"],
    )
    qfunc2_encoder = GCN(
        nodes.shape[-1],
        config["encoder_kwargs"]["encoder_hiddens"],
        with_layer_norm=config["encoder_kwargs"]["with_layer_norm"],
        final_tanh=config["encoder_kwargs"]["final_tanh"],
    )
    qfunc1t_encoder = GCN(
        nodes.shape[-1],
        config["encoder_kwargs"]["encoder_hiddens"],
        with_layer_norm=config["encoder_kwargs"]["with_layer_norm"],
        final_tanh=config["encoder_kwargs"]["final_tanh"],
    )
    qfunc2t_encoder = GCN(
        nodes.shape[-1],
        config["encoder_kwargs"]["encoder_hiddens"],
        with_layer_norm=config["encoder_kwargs"]["with_layer_norm"],
        final_tanh=config["encoder_kwargs"]["final_tanh"],
    )

    # policy setup for GaussPolicy like GraphOpt;
    gauss_policy_kwargs = config["gauss_policy_kwargs"].copy()
    gauss_policy_kwargs["encoder"] = encoder

    # policy setup for TwoStageGaussPolicy;
    tsg_policy_kwargs = config["tsg_policy_kwargs"].copy()
    tsg_policy_kwargs["encoder"] = encoder

    # select policies from config['agent_kwargs']['policy_constructor'];
    policies = {
        "GaussPolicy": gauss_policy_kwargs,
        "TwoStageGaussPolicy": tsg_policy_kwargs,
    }

    # qfunc kwargs;
    Q1_kwargs = config["qfunc_kwargs"].copy()
    Q1_kwargs["encoder"] = qfunc1_encoder

    Q2_kwargs = config["qfunc_kwargs"].copy()
    Q2_kwargs["encoder"] = qfunc2_encoder

    Q1t_kwargs = config["qfunc_kwargs"].copy()
    Q1t_kwargs["encoder"] = qfunc1t_encoder

    Q2t_kwargs = config["qfunc_kwargs"].copy()
    Q2t_kwargs["encoder"] = qfunc2t_encoder

    # goes in the kwargs of the sac agent constructor;
    sac_agent_kwargs = dict(
        training_kwargs=config["training_kwargs"],
        policy_kwargs=policies[config["agent_kwargs"]["policy_constructor"]],
        Q1_kwargs=Q1_kwargs,
        Q2_kwargs=Q2_kwargs,
        Q1t_kwargs=Q1t_kwargs,
        Q2t_kwargs=Q2t_kwargs,
        buffer_kwargs=dict(
            max_size=config["buffer_kwargs"]["max_size"],
            nodes=nodes,
            seed=config["training_kwargs"]["seed"],
        ),
        env_kwargs=dict(
            x=nodes,
            reward_fn=reward_fn,
            max_episode_steps=config["env_kwargs"]["max_episode_steps"],
            num_expert_steps=config["env_kwargs"]["num_expert_steps"],
            max_repeats=config["env_kwargs"]["max_repeats"],
            max_self_loops=config["env_kwargs"]["max_self_loops"],
            drop_repeats_or_self_loops=config["env_kwargs"][
                "drop_repeats_or_self_loops"
            ],
            reward_fn_termination=config["env_kwargs"][
                "reward_fn_termination"
            ],
        ),
    )

    # for the positional args in the sac agent constructor;
    agent_kwargs = config["agent_kwargs"].copy()
    agent_kwargs["policy_constructor"] = constructors[
        config["agent_kwargs"]["policy_constructor"]
    ]
    agent_kwargs["qfunc_constructor"] = constructors[
        config["agent_kwargs"]["qfunc_constructor"]
    ]
    agent_kwargs["env_constructor"] = constructors[
        config["agent_kwargs"]["env_constructor"]
    ]
    agent_kwargs["buffer_constructor"] = constructors[
        config["agent_kwargs"]["buffer_constructor"]
    ]
    agent_kwargs["save_to"] = Path(config["agent_kwargs"]["save_to"])

    agent = SACAgentGraph(
        **agent_kwargs,
        # cache_best_policy=True,
        UT_trick=config["extra_info_kwargs"]["UT_trick"],
        with_entropy=config["extra_info_kwargs"]["with_entropy"],
        **sac_agent_kwargs,
    )

    agent.train_k_epochs(
        k=config["extra_info_kwargs"]["num_epochs"],
        config=config,
        vis_graph=config["extra_info_kwargs"]["vis_graph"],
    )
