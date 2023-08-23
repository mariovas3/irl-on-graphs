from scipy.spatial import KDTree
import numpy as np
import torch
import torch.nn.functional as F
from typing import Callable
from torch_geometric.data import Data
from collections import namedtuple


class WeightsProcessor:
    def __init__(self, weight_type, max_ep_len, scaling_type='abs_max'):
        self.weight_type = weight_type
        self.scaling_type = scaling_type
        self.max_ep_len = max_ep_len
        self.log_weights = []
        if self.weight_type == 'vanilla':
            if self.scaling_type == 'abs_max':
                self.scale = 0.
        elif self.weight_type == 'per_dec':
            self.longest = 0
            self.scale = torch.zeros((1, max_ep_len), dtype=torch.float32)
    
    def __call__(self, log_w):
        if self.weight_type == 'per_dec':
            pad_val = float('nan')
            if self.scaling_type == 'abs_max':
                pad_val = - float('inf')
                msk = self.scale[:, :len(log_w)].abs() < log_w.view(1, -1).abs()
                self.scale[:, :len(log_w)][msk] = log_w.view(1, -1)[msk]
            elif self.scaling_type == 'median':
                pad_val = float('nan')
            self.log_weights.append(
                F.pad(
                    log_w,
                    (0, self.max_ep_len - len(log_w)),
                    value=pad_val
                )
            )
            self.longest = max(len(log_w), self.longest)
        elif self.weight_type == 'vanilla':
            if self.scaling_type == 'abs_max':
                self.scale = log_w if abs(log_w) > abs(self.scale) else self.scale
            self.log_weights.append(log_w)
    
    def get_weights(self):
        self.log_weights = torch.stack(self.log_weights)
        if self.weight_type == 'per_dec':
            if self.scaling_type == 'median':
                self.scale[:, :self.longest] = torch.nanmedian(
                    self.log_weights[:, :self.longest],
                    0, keepdim=True
                ).values
            # subtract some log weight based on self.scale_type
            # from all log weights, so each weight is log(wi / w_chosen)
            # and then take exp to get wi / w_chosen;
            self.log_weights[:, :self.longest] = (
                self.log_weights[:, :self.longest] 
                - self.scale[:, :self.longest]
            ).exp()
            # normalise over 0 dim (batch dim);
            # gives pmf for paths of length k for all k;
            self.log_weights[:, :self.longest] = (
                self.log_weights[:, :self.longest] / self.log_weights[:, :self.longest].nansum(
                    dim=0, keepdim=True
                )
            )
            assert torch.allclose(self.log_weights[:, :self.longest].sum(0), torch.ones((1, )))
        elif self.weight_type == 'vanilla':
            if self.scaling_type == 'median':
                self.scale = torch.median(self.log_weights)
            self.log_weights = (self.log_weights - self.scale).exp()
            self.log_weights = self.log_weights / self.log_weights.sum()
            assert torch.allclose(self.log_weights.sum(), torch.ones((1,)))
        if self.weight_type == 'per_dec':
            return self.log_weights, self.longest
        return self.log_weights


def OI_init(model):
    for n, m in model.named_parameters():
        if 'bias' in n:
            m.data.fill_(0.)
        if 'weight' in n and m.ndim == 2:
            torch.nn.init.orthogonal_(m.data)


def get_dfs_edge_order(adj_list, source):
    state = [0 for _ in range(len(adj_list))]
    state[source] = 1
    edge_index = [[], []]
    def f(i):
        if state[i] != 2:
            state[i] = 1
            for j in adj_list[i]:
                if state[j] != 1:
                    edge_index[0].extend([i, j])
                    edge_index[1].extend([j, i])
                    f(j)
            state[i] = 2
    f(source)
    for source in range(len(adj_list)):
        if state[source] != 2:
            assert state[source] == 0
            f(source)
    return torch.tensor(edge_index, dtype=torch.long)


def edge_index_to_adj_list(edge_index, n_nodes):
    """
    edge_index should be edge_index[:, ::2] i.e., no 
    duplicate edges;
    """
    adj_list = [[] for _ in range(n_nodes)]
    for i in range(edge_index.shape[-1]):
        first, second = edge_index[0, i].item(), edge_index[1, i].item()
        adj_list[first].append(second)
        adj_list[second].append(first)
    return adj_list


def inc_lcr_reg(r1, r2, curr_rewards):
    inc = 0
    if r1 is not None and r2 is not None:
        temp = torch.cat((
            torch.stack((r1, r2)), curr_rewards
        ))
        inc = ((temp[2:] + temp[:-2] - 2 * temp[1:-1]) ** 2).sum()
    elif len(curr_rewards) > 2:
        inc = ((curr_rewards[2:] + curr_rewards[:-2] - 2 * curr_rewards[1:-1]) ** 2).sum()
    if len(curr_rewards) > 1:
        r1, r2 = curr_rewards[-2], curr_rewards[-1]
    return r1, r2, inc


def get_rand_edge_index(edge_index, num_edges):
    if num_edges == 0:
        return torch.tensor([[], []], dtype=torch.long)
    T = edge_index.shape[-1]
    num_edges = min(T // 2, num_edges)
    idxs = np.random.choice(
        range(0, T, 2), 
        size=num_edges, 
        replace=False
    )
    idxs = sum([(i, i + 1) for i in idxs], ())
    return edge_index[:, idxs]


def get_unique_edges(edge_index):
    return set(
        [
            (i, j) if i < j else (j, i)
            for (i, j) in zip(*edge_index.tolist())
        ]
    )


def kdtree_similarity(
    node_embeds: np.ndarray, 
    a1: np.ndarray, 
    a2: np.ndarray,
    forbid_self_loops_repeats=False,
    edge_set: set=None,
):
    tree = KDTree(node_embeds)
    first = tree.query(a1, k=1)[-1]
    if forbid_self_loops_repeats:
        second = tree.query(a2, k=10)[-1]
        for s in second:
            temp1, temp2 = min(first, s), max(first, s)
            if first == s or (temp1, temp2) in edge_set:
                continue
            return first, s
        # signal that we're out of recommends by giving a self loop;
        return first, first
    second = tree.query(a2, k=1)[-1]
    return first, second


def sigmoid_similarity(
    node_embeds: np.ndarray, 
    a1: np.ndarray, 
    a2: np.ndarray, 
    forbid_self_loops_repeats=False, 
    edge_set: set=None,
):
    node_embeds = torch.from_numpy(node_embeds)
    a1, a2 = torch.from_numpy(a1), torch.from_numpy(a2)
    actions = torch.cat((a1.view(-1, 1), a2.view(-1, 1)), -1)
    temp = node_embeds @ actions
    temp = torch.sigmoid(temp)
    first = temp[:, 0].argmax().item()
    if forbid_self_loops_repeats:
        for s in torch.topk(temp[:, 1], k=10)[-1]:
            temp1, temp2 = min(s.item(), first), max(s.item(), first)
            if temp1 == temp2 or (temp1, temp2) in edge_set:
                continue
            return first, s.item()
        # signal that we're out of recommends by giving a self loop;
        return first, first
    second = temp[:, 1].argmax().item()
    return first, second


def euc_dist_similariry(
    node_embeds: np.ndarray,
    a1: np.ndarray,
    a2: np.ndarray,
    forbid_self_loops_repeats=False,
    edge_set: set=None,
):
    node_embeds = torch.from_numpy(node_embeds)
    a1, a2 = torch.from_numpy(a1), torch.from_numpy(a2)
    actions = torch.cat((a1.view(-1, 1), a2.view(-1, 1)), -1)
    criterion = (
        (node_embeds * node_embeds).sum(-1, keepdims=True) 
        - 2 * node_embeds @ actions
    )
    first = criterion[:, 0].argmax().item()
    if forbid_self_loops_repeats:
        for s in torch.topk(criterion[:, 1], k=10)[-1]:
            temp1, temp2 = min(s.item(), first), max(s.item(), first)
            if temp1 == temp2 or (temp1, temp2) in edge_set:
                continue
            return first, s.item()
        # return self loops if second not among first 10 options;
        return first, first
    second = criterion[:, 1].argmax().item()
    return first, second


class GraphEnv:
    def __init__(
        self,
        x: torch.Tensor,
        expert_edge_index: torch.Tensor,
        num_edges_start_from: int,
        reward_fn: Callable,
        max_episode_steps: int,
        num_expert_steps: int,
        max_repeats: int,
        max_self_loops: int,
        drop_repeats_or_self_loops: bool = False,
        id: str = None,
        reward_fn_termination: bool = False,
        calculate_reward: bool=True,
        min_steps_to_do: int=3,
        similarity_func: Callable=None,
        forbid_self_loops_repeats: bool=False,
    ):
        """
        Args:
            x (torch.Tensor): Initial node features.
            reward_fn (Callable): Accept graph Data instance
                and output scalar reward.
            max_episode_steps (int): Truncation length.
            num_expert_steps (int): Termination length.
            max_repeats (int): Terminate if algo suggests
                existing edges at least max_repeats times.
            drop_repeats_or_self_loops (bool): If True,
                self.steps_done does not increase if we get
                a repeated edge or a self loop suggestion
                from the policy. Info about the envent will
                be given in the info['self_loop']
                or info['is_repeated'].
        """
        self.spec = namedtuple("spec", "id max_episode_steps")
        self.spec.id = id or "GraphEnv"
        self.x = x
        self.expert_edge_index = expert_edge_index
        self.num_edges_start_from = num_edges_start_from
        self.similarity_func = similarity_func or kdtree_similarity
        self.forbid_self_loops_repeats = forbid_self_loops_repeats
        self.spec.max_episode_steps = max_episode_steps
        self.num_expert_steps = num_expert_steps
        self.drop_repeats_or_self_loops = drop_repeats_or_self_loops
        self.max_repeats = max_repeats
        self.max_self_loops = max_self_loops
        if self.forbid_self_loops_repeats:
            assert self.max_repeats == 1 and self.max_self_loops == 1
        self.reward_fn_termination = reward_fn_termination
        self.min_steps_to_do = min_steps_to_do
        # reward fn should have its own GNN encoder;
        # due to deterministic transitions, make reward
        # state dependent only, since S x A -> S deterministically.
        self.reward_fn = reward_fn
        self.calculate_reward = calculate_reward

        # attributes to be reset when reset called;
        if self.expert_edge_index is None:
            self.edge_index = torch.tensor([[], []], dtype=torch.long)
        else: 
            self.edge_index = get_rand_edge_index(
                expert_edge_index, 
                self.num_edges_start_from
            )
        self.unique_edges = get_unique_edges(self.edge_index)
        self.steps_done = len(self.unique_edges)
        self.repeats_done = 0
        self.self_loops_done = 0
        self.terminated, self.truncated = False, False
        self.num_self_loops = 0

    def reset(self, seed=0):
        """
        Returns (observation, None) to be compatible with openai gym.
        Also the seed parameter is given for compatibility but is not
        used.
        """
        self.reward_fn.reset()
        if self.expert_edge_index is None:
            self.edge_index = torch.tensor([[], []], dtype=torch.long)
        else:
            self.edge_index = get_rand_edge_index(
                self.expert_edge_index, 
                self.num_edges_start_from
            )

        data = Data(
            x=self.x, edge_index=self.edge_index.clone()
        )
        # self.edge_index = torch.tensor([[], []], dtype=torch.long)
        self.unique_edges = get_unique_edges(self.edge_index)
        self.steps_done = len(self.unique_edges)
        self.repeats_done = 0
        self.self_loops_done = 0
        self.terminated, self.truncated = False, False
        self.num_self_loops = 0
        return data, None

    def _get_edge_hash(self, first, second):
        if first > second:
            return second, first
        return first, second

    def _update_info_terminals(self, info):
        self.terminated = (
            # enforce at least min steps to do;
            (self.steps_done >= self.min_steps_to_do + self.num_edges_start_from) 
            and (
                len(self.unique_edges) >= self.num_expert_steps
                or (
                    self.reward_fn_termination and self.reward_fn.should_terminate
                )
                # experimental, terminate when max repeats or max_ep_steps reached;
                # or (
                #     self.steps_done >= self.spec.max_episode_steps
                #     or self.repeats_done >= self.max_repeats
                #     or self.self_loops_done >= self.max_self_loops
                # )
            )
        )
        self.truncated = (
            # enforce at least min steps to do;
            (self.steps_done >= self.min_steps_to_do + self.num_edges_start_from) 
            and (
                self.steps_done >= self.spec.max_episode_steps
                or self.repeats_done >= self.max_repeats
                or self.self_loops_done >= self.max_self_loops
            )
        )

        info["terminated"] = self.terminated
        info["expert_episode_length_reached"] = (
            self.steps_done >= self.num_expert_steps
        )
        info["max_repeats_reached"] = self.repeats_done >= self.max_repeats
        info["max_self_loops_reached"] = (
            self.self_loops_done >= self.max_self_loops
        )
        info["truncated"] = self.truncated
        info["episode_truncation_length_reached"] = (
            self.steps_done >= self.spec.max_episode_steps
        )
        info["steps_done"] = self.steps_done
        info["repeats_done"] = self.repeats_done
        info["self_loops_done"] = self.self_loops_done

    def step(self, action):
        """
        Returns (observation, terminated, truncated, info)
        
        Note: You may wish to set calculate_reward=False if you wish 
                ot e.g., sample only states and actions and then calculate
                reward on batches of states and actions. This is generally 
                useful when reward_fn is Deep net.
        """
        assert not (self.terminated or self.truncated)
        self.steps_done += 1
        info = {
            "terminated": False,
            "expert_episode_length_reached": False,
            "max_repeats_reached": False,
            "max_self_loops_reached": False,
            "truncated": False,
            "episode_truncation_length_reached": False,
            "self_loop": False,
            "is_repeat": False,
            "steps_done": self.steps_done,
            "repeats_done": self.repeats_done,
            "self_loops_done": self.self_loops_done,
            "first": None,
            "second": None,
        }

        # unpack action; a1 and a2 are numpy arrays;
        (a1, a2), node_embeds = action

        # a1 and a2 should be flattened and have same len as node embeds;
        assert len(a1.shape) == 1 == len(a2.shape)
        action_dim = len(a1)
        obs_dim = node_embeds.shape[-1]
        assert action_dim == obs_dim

        # get idx of proposed nodes to connect;
        first, second = self.similarity_func(
            node_embeds, a1, a2,
            forbid_self_loops_repeats=self.forbid_self_loops_repeats,
            edge_set=self.unique_edges,
        )

        # make data object;
        data = Data(x=self.x, edge_index=self.edge_index)

        # see which nodes will be connected;
        # order matters since the first node is the commiting node
        # and the next is the suitable node;
        info["first"] = first
        info["second"] = second

        # calculate reward;
        idxs = torch.tensor([[first, second]], dtype=torch.long)
        reward = None
        if self.calculate_reward:
            raise NotImplementedError(
                'currently only state-action experience is stored in '
                'buffer and reward is calculated at batch-sampling '
                'time, given current config of reward. This is '
                'so that buffer does not be cleared after reward grad '
                'step and is efficient since deep learning is good with '
                'batch computation rather than per-instance computation.'
            )
            reward = self.reward_fn((data, idxs), action_is_index=True)

        # if self loop;
        if first == second:
            self.self_loops_done += 1
            info["self_loop"] = True
            if self.drop_repeats_or_self_loops:
                self.steps_done -= 1
            self._update_info_terminals(info)
            return (
                data,
                reward,
                self.terminated,
                self.truncated,
                info,
            )

        # if repeated edge;
        if self._get_edge_hash(first, second) in self.unique_edges:
            self.repeats_done += 1
            info["is_repeat"] = True
            if self.drop_repeats_or_self_loops:
                self.steps_done -= 1
            self._update_info_terminals(info)
            return (
                data,
                reward,
                self.terminated,
                self.truncated,
                info,
            )
        else:
            # add undirected edge to the graph Data container;
            data.edge_index = torch.cat(
                (
                    data.edge_index,
                    torch.tensor(
                        [(first, second), (second, first)], dtype=torch.long
                    ),
                ),
                -1,
            )
            self.edge_index = data.edge_index

            # add edge to set of edges;
            self.unique_edges.add(self._get_edge_hash(first, second))

        # update info;
        self._update_info_terminals(info)

        # reward fn should have its own encoder GNN;
        return (
            data,
            reward,
            self.terminated,
            self.truncated,
            info,
        )


def get_action_vector_from_idx(node_embeds, action_idxs, num_graphs):
    """
    For each graph according to batch_idxs, return the node
    embeddings according to the indexes in action_idxs.
    Assumes all graphs in the batch have the same number of
    nodes.

    Args:
        node_embeds (torch.Tensor): Tensor of node embeddings.
        action_idxs (torch.Tensor): Indexes of shape (B, 2) where
            B is the size of the batch and for each graph we
            select 2 node embeddings.
        num_graphs (int): number of graphs in batch.
    """
    n_nodes = node_embeds.shape[0] // num_graphs
    action_idxs = (
        action_idxs + torch.arange(len(action_idxs)).view(-1, 1) * n_nodes
    )
    # return shape is (B, 2 * node_embed_dim)
    return node_embeds[action_idxs.view(-1)].view(
        -1, 2 * node_embeds.shape[-1]
    )
