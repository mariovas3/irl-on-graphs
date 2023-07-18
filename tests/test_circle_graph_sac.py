import sys
from pathlib import Path

p = Path(__file__).absolute().parent.parent
sys.path.append(str(p))

from graph_irl.sac import *
from graph_irl.graph_rl_utils import GraphEnv
from graph_irl.buffer_v2 import GraphBuffer

from torch_geometric.data import Data


class CircleGraphReward:
    def __init__(self, n_nodes):
        self.degrees = np.zeros(n_nodes, dtype=np.int64)
        self.adj_list = [[] for _ in range(n_nodes)]
        self.edge_set = set()
        self.component_penalty = - 5
        self.n_nodes = n_nodes
        self.n_components_last = n_nodes
    
    def reset(self):
        self.degrees = np.zeros(self.n_nodes, dtype=np.int64)
        self.adj_list = [[] for _ in range(self.n_nodes)]
        self.edge_set = set()
        self.n_components_last = self.n_nodes

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
        # if self-loop reward is -100;
        if first == second:
            return - 5 + (self.n_components_last - 1) * self.component_penalty
        
        # keep edges in ascending order in indexes;
        if first > second:
            first, second = second, first
        
        # if the edge is a repeat, reward is -100;
        if (first, second) in self.edge_set:
            return - 5 + (self.n_components_last - 1) * self.component_penalty
        
        # increment degrees of relevant nodes from new edge;
        self.degrees[first] += 1
        self.degrees[second] += 1

        # add new edge to edge set;
        self.edge_set.add((first, second))

        # update adjacency list;
        self.adj_list[first].append(second)
        self.adj_list[second].append(first)

        self.n_components_last = self._count_graph_components()

        if self.degrees[first] > 2 or self.degrees[second] > 2:
            return (self.n_nodes + self.n_components_last - 1) * self.component_penalty
        
        # 0 reward if new edge keeps degrees in bound [0, 2]
        # and only penalty for number of connected components;
        return (self.n_components_last - 1) * self.component_penalty


if __name__ == "__main__":
    
    # buffer setup;
    buffer_len = 10_000
    n_nodes = 10
    nodes = torch.eye(n_nodes)

    # instantiate buffer;
    buffer = GraphBuffer(
        max_size=buffer_len,
        nodes=nodes,
        seed=0,
    )

    # instantiate reward;
    reward_fn = CircleGraphReward(n_nodes)

    # env setup;
    env = GraphEnv(
        x=nodes,
        reward_fn=reward_fn,
        max_episode_steps=n_nodes,
        num_expert_steps=n_nodes,
        max_repeats=min(n_nodes // 3, 3),
        drop_repeats_or_self_loops=False,
    )
    print(env.spec.id)

    # encoder setup;
    encoder_hiddens = [16, 16]
    encoder = GCN(nodes.shape[-1], encoder_hiddens, with_layer_norm=True)
    qfunc1_encoder = GCN(nodes.shape[-1], encoder_hiddens, with_layer_norm=True)
    qfunc2_encoder = GCN(nodes.shape[-1], encoder_hiddens, with_layer_norm=True)
    qfunc1t_encoder = GCN(nodes.shape[-1], encoder_hiddens, with_layer_norm=True)
    qfunc2t_encoder = GCN(nodes.shape[-1], encoder_hiddens, with_layer_norm=True)

    # policy setup for GaussPolicy like GraphOpt;
    gauss_policy_kwargs = {
        'obs_dim': encoder_hiddens[-1],
        'action_dim': encoder_hiddens[-1],
        'hiddens': [30, 30],
        'with_layer_norm': True,
        'encoder': encoder,
        'two_action_vectors': True,
    }

    # policy setup for TwoStageGaussPolicy;
    policy_kwargs = {
        "obs_dim": encoder_hiddens[-1],
        "action_dim": encoder_hiddens[-1],
        "hiddens1": [30, 30],
        "hiddens2": [40, 40],
        "encoder": encoder,
        "with_layer_norm": True,
    }

    # setup agent;
    agent_policy_kwargs = dict(
        agent_kwargs=dict(
            name="SACAgentGraph",
            policy=TwoStageGaussPolicy,
            policy_lr=3e-4,
            entropy_lb=encoder_hiddens[-1],
            temperature_lr=3e-4,
        ),
        policy_kwargs=policy_kwargs,
    )

    # agent = SACAgentGraph(**agent_policy_kwargs['agent_kwargs'], 
                        #   **agent_policy_kwargs['policy_kwargs'])
    
    # runtime setup;
    num_steps_to_collect = 5 * n_nodes
    batch_size = 25
    num_iters = 200
    num_epochs = 1
    num_grad_steps = 30

    # buffer.collect_path(
        # env=env,
        # agent=agent,
        # num_steps_to_collect=num_steps_to_collect,
    # )

    """END OF TEST 1 HERE;"""

    # # test graph sac training;
    # optimQ1 = torch.optim.Adam(Q1.parameters(), lr=3e-4)
    # optimQ2 = torch.optim.Adam(Q2.parameters(), lr=3e-4)

    # train_sac_one_epoch(
    #     env,
    #     agent,
    #     Q1,
    #     Q2,
    #     Q1t,
    #     Q2t,
    #     optimQ1,
    #     optimQ2,
    #     tau=.1,
    #     discount=.99,
    #     num_iters=100,
    #     num_grad_steps=200,
    #     num_steps_to_sample=200,
    #     num_eval_steps_to_sample=200,
    #     buffer=buffer,
    #     batch_size=batch_size,
    #     qfunc1_losses=[],
    #     qfunc2_losses=[],
    #     UT_trick=False,
    #     with_entropy=False,
    #     for_graph=True,
    #     eval_path_returns=[],
    #     eval_path_lens=[],
    # )

    # train sac setup;
    # agent_policy_kwargs = dict(
    #     agent_kwargs=dict(
    #         name="SACAgentGraph",
    #         policy=GaussPolicy,
    #         policy_lr=3e-4,
    #         entropy_lb=encoder_hiddens[-1],
    #         temperature_lr=3e-4,
    #     ),
    #     policy_kwargs=policy_kwargs,
    # )
    # num_iters = 100
    # num_epochs = 1

    train_sac(
        env=env,
        agent=SACAgentGraph,
        num_iters=num_iters,
        qfunc_hiddens=[16, 16],
        qfunc_layer_norm=True,
        qfunc_lr=3e-4,
        buffer_len=1_000,
        batch_size=batch_size,
        discount=0.99,
        tau=0.1,
        seed=0,
        save_returns_to=TEST_OUTPUTS_PATH,
        num_steps_to_sample=num_steps_to_collect,
        num_eval_steps_to_sample=num_steps_to_collect,
        num_grad_steps=num_grad_steps,
        num_epochs=num_epochs,
        min_steps_to_presample=50,
        UT_trick=False,
        with_entropy=False,
        for_graph=True, 
        qfunc1_encoder=qfunc1_encoder,
        qfunc2_encoder=qfunc2_encoder,
        qfunc1t_encoder=qfunc1_encoder,
        qfunc2t_encoder=qfunc2t_encoder,
        buffer_instance=buffer,
        **agent_policy_kwargs,
    )
