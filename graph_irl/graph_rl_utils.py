import numpy as np
import torch
import torch.nn.functional as F
from typing import Callable
from torch_geometric.data import Data
from collections import namedtuple


class WeightsProcessor:
    def __init__(self, weight_type, max_ep_len):
        self.weight_type = weight_type
        self.max_ep_len = max_ep_len
        self.log_weights = []
        assert weight_type in ('per_dec', 'vanilla')
        if weight_type == 'per_dec':
            self.longest = 0

    def __call__(self, log_w):
        if self.weight_type == 'per_dec':
            pad_val = - float('inf')
            # append and pad to the right until len is max_ep_len;
            self.log_weights.append(
                F.pad(
                    log_w,
                    (0, self.max_ep_len - len(log_w)),
                    value=pad_val
                )
            )
            self.longest = max(len(log_w), self.longest)
        elif self.weight_type == 'vanilla':
            self.log_weights.append(log_w)
    
    def get_weights(self):
        self.log_weights = torch.stack(self.log_weights)
        if self.weight_type == 'per_dec':
            weights = torch.softmax(self.log_weights, 0)
            assert torch.allclose(weights[:, :self.longest].sum(0), torch.ones((1,)))
            return weights, self.longest
        elif self.weight_type == 'vanilla':
            weights = torch.softmax(self.log_weights, -1)
            assert torch.allclose(weights.sum(), torch.ones((1,)))
            return weights


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
    Returns adjacency list built from edge index.

    Note: If there is an edge (i, j) in an undirected graph,
        we have (i, j) and (j, i) in the typical edge_index.
        In this case, before passing edge_index to this function,
        make sure you clean it by passing edge_index[:, ::2] instead.
        This will leave only one entry per edge.
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


def get_batch_knn_index(batch, knn_edge_index):
    num_graphs = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
    assert len(batch.x) % num_graphs == 0
    n_nodes = len(batch.x) // num_graphs
    return torch.cat(
        [
            knn_edge_index + i * n_nodes
            for i in range(num_graphs)
        ], -1
    )


def get_rand_edge_index(edge_index, num_edges):
    """
    Return a subset of edge_index with num_edges randomly sampled edges.
    
    Note: Assumes undirected graph edge index format. That is,
        each edge has two entries in edge_index as
        (from, to), (to, from) and these are contiguous in edge_index.
    """
    if num_edges == 0:
        return torch.tensor([[], []], dtype=torch.long)
    T = edge_index.shape[-1]
    if num_edges == T // 2:
        return edge_index
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


def batch_argmax(scores):
    firsts = scores[:, :, 0].argmax(-1)
    seconds = scores[:, :, 1].argmax(-1)
    return firsts.tolist(), seconds.tolist()


def select_actions_from_scores(scores, positives_dict, edge_set):
    which_action = None
    recip_rank = None
    first, second = 0, 0
    if scores.ndim == 3:
        recip_rank = 0
        flag = True
        firsts, seconds = batch_argmax(scores)
        for i, (f, s) in enumerate(zip(firsts, seconds)):
            e1, e2 = min(f, s), max(f, s)
            if positives_dict is not None and recip_rank == 0:
                if (e1, e2) in positives_dict:
                    recip_rank = 1 / (i + 1)
            # ignore repeats and self loops;
            if (e1, e2) in edge_set or e1 == e2:
                continue
            if flag:
                first, second = f, s
                which_action = i
                flag = False
            if not flag and recip_rank > 0:
                return first, second, which_action, recip_rank
    else:
        assert scores.ndim == 2    
        first = scores[:, 0].argmax(-1).item()
        second = scores[:, 1].argmax(-1).item()
    return first, second, which_action, recip_rank


def sigmoid_similarity(
    node_embeds: np.ndarray, 
    a1: np.ndarray, 
    a2: np.ndarray, 
    edge_set: set=None,
    positives_dict=None
):
    node_embeds = torch.from_numpy(node_embeds)
    a1, a2 = torch.from_numpy(a1), torch.from_numpy(a2)
    actions = torch.cat((a1.unsqueeze(-1), a2.unsqueeze(-1)), -1)
    temp = node_embeds @ actions
    temp = torch.sigmoid(temp)
    return select_actions_from_scores(temp, positives_dict, edge_set)


def euc_dist_similarity(
    node_embeds: np.ndarray,
    a1: np.ndarray,
    a2: np.ndarray,
    edge_set: set=None,
    positives_dict=None
):
    node_embeds = torch.from_numpy(node_embeds)
    a1, a2 = torch.from_numpy(a1), torch.from_numpy(a2)
    actions = torch.cat((a1.unsqueeze(-1), a2.unsqueeze(-1)), -1)
    # neg euc dist;
    temp = - (
        (node_embeds * node_embeds).sum(-1, keepdims=True) 
        - 2 * node_embeds @ actions 
    )
    return select_actions_from_scores(temp, positives_dict, edge_set)


def get_valid_action_vectors(firsts, seconds, batch, proposals):
    """
    Returns a valid selection of proposal actions from proposals.
    Validity is based on actions not resulting in self loops or 
    repeated edges.

    Args:
        firsts (torch.Tensor): is of shape (batch, num_proposals);
        seconds (torch.Tensor): is of shape (batch, num_proposals);
        batch (tg Batch or tg Data object): batch of graphs or single graph;
        proposals (torch.Tensor): is of shape (batch, num_proposals, embed_dim, 2);
    """
    # idxs to choose the correct proposal action for each batch;
    # at end len(idxs) must be equal to batch_size;
    idxs = []
    n_graphs = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
    assert len(batch.x) % n_graphs == 0
    
    def small_sort(x, y):
        return min(x, y), max(x, y)
    
    # this is nucessary, since the batch is just a giant graph
    # with multiple components, each of which is an individual graph
    # (state) from the replay buffer; Some graphs can have more edges 
    # than other so it's not trivial to directly work with the 
    # edge index of the batch itself;
    given_eis = [obs.edge_index for obs in batch.to_data_list()]
    
    for b, (f, s, aeis) in enumerate(zip(firsts, seconds, given_eis)):
        # get the edge set;
        eset = set([small_sort(x[0], x[1]) 
                    for x in zip(*aeis.tolist())])
        idx = 0
        for i, (x, y) in enumerate(zip(f, s)):
            # skip if is self loop or is an existing edge;
            x, y = x.item(), y.item()
            if x == y or small_sort(x, y) in eset:
                continue
            # choose the action if the edge is not 
            # in the edge set and not self loop;
            idx = i
            break
        idxs.append((b, idx))
    idxs = torch.tensor(idxs, dtype=torch.long)
    actions = proposals[idxs[:, 0], idxs[:, 1]]
    assert actions.shape == (len(proposals), proposals.shape[-2], 2)
    return actions[:, :, 0], actions[:, :, 1]


def get_valid_proposal(node_embeds_detached, batch, a1s, a2s):
    """
    Return action vectors that don't lead to self loop 
    or repeated edge.

    Args:
        node_embeds_detached (torch.Tensor): shape is (B * N, embed_dim)
        batch (tg Batch or Data instance): batch of graphs (actual states)
        a1s (torch.Tensor): shape is (k_proposals, batch_size, node_embed_dim)
        a2s (torch.Tensor): shape is (k_proposals, batch_size, node_embed_dim)
    Return:
        Returns (valid1, valid2) where the each element is of shape 
            (batch_size, node_embed_dim).
    Note:
        If no valid action is found among the samples, returns the 
            first action among the proposals.
    """
    # a1s and a2s are of shape 
    # (k_proposals, batch_size, node_embed_dim)
    
    # proposals is of shape (batch_size, k_proposals, node_embed_dim, 2)
    proposals = torch.cat((a1s.unsqueeze(-1), a2s.unsqueeze(-1)), -1).permute(1, 0, 2, 3)
    # get number of graphs and number of nodes per graph;
    n_graphs = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
    assert len(node_embeds_detached) % n_graphs == 0
    n_nodes = len(node_embeds_detached) // n_graphs
    # scores is of shape (batch_size, k_proposals, n_nodes, 2)
    scores = node_embeds_detached.view(n_graphs, n_nodes, -1).unsqueeze(1) @ proposals
    # argmax scores on n_nodes dim;
    firsts = scores[:, :, :, 0].argmax(-1)
    seconds = scores[:, :, :, 1].argmax(-1)
    return get_valid_action_vectors(firsts, seconds, batch, proposals)


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
        self.similarity_func = similarity_func or sigmoid_similarity
        self.spec.max_episode_steps = max_episode_steps
        self.num_expert_steps = num_expert_steps
        self.drop_repeats_or_self_loops = drop_repeats_or_self_loops
        self.max_repeats = max_repeats
        self.max_self_loops = max_self_loops
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

    def step(self, action, positives_dict=None):
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
            "which_action": None,
            "recip_rank": None, 
        }

        # unpack action; a1 and a2 are numpy arrays;
        (a1, a2), node_embeds = action

        # a1 and a2 should either be 1d vectors
        # unless k_proposals > 1, in which case they 
        # should be 2d tensors;
        # assert len(a1.shape) == 1 == len(a2.shape)
        assert len(a1.shape) == len(a2.shape)
        action_dim = a1.shape[-1]
        obs_dim = node_embeds.shape[-1]
        assert action_dim == obs_dim

        # get idx of proposed nodes to connect;
        first, second, which_action, recip_rank = self.similarity_func(
            node_embeds, a1, a2,
            edge_set=self.unique_edges,
            positives_dict=positives_dict,
        )

        # make data object;
        data = Data(x=self.x, edge_index=self.edge_index)

        # see which nodes will be connected;
        # order matters since the first node is the commiting node
        # and the next is the suitable node;
        info["first"] = first
        info["second"] = second
        info['which_action'] = which_action
        info['recip_rank'] = recip_rank

        # calculate reward;
        idxs = torch.tensor([[first, second]], dtype=torch.long)
        reward = None
        if self.calculate_reward:
            raise NotImplementedError(
                'currently only state-action experience is stored in '
                'buffer and reward is calculated at batch-sampling '
                'time, given current config of reward. This is '
                'so that buffer does not get cleared after reward grad '
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
