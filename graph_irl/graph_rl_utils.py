from scipy.spatial import KDTree
import torch
from typing import Callable
from torch_geometric.data import Data
from collections import namedtuple


class GraphEnv:
    def __init__(
        self,
        x: torch.Tensor,
        reward_fn: Callable,
        max_episode_steps: int,
        num_expert_steps: int,
        max_repeats: int,
        max_self_loops: int,
        drop_repeats_or_self_loops: bool = False,
        id: str = None,
        reward_fn_termination: bool = False,
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
        self.spec.id = id if id is not None else "GraphEnv"
        self.x = x
        self.spec.max_episode_steps = max_episode_steps
        self.num_expert_steps = num_expert_steps
        self.drop_repeats_or_self_loops = drop_repeats_or_self_loops
        self.max_repeats = max_repeats
        self.max_self_loops = max_self_loops
        self.reward_fn_termination = reward_fn_termination
        # reward fn should have its own GNN encoder;
        # due to deterministic transitions, make reward
        # state dependent only, since S x A -> S deterministically.
        self.reward_fn = reward_fn

        # attributes to be reset when reset called;
        self.edge_index = torch.tensor([[], []], dtype=torch.long)
        self.unique_edges = set()
        self.steps_done = 0
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
        data = Data(
            x=self.x, edge_index=torch.tensor([[], []], dtype=torch.long)
        )
        self.edge_index = torch.tensor([[], []], dtype=torch.long)
        self.unique_edges = set()
        self.steps_done = 0
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
            len(self.unique_edges) >= self.num_expert_steps
            or self.repeats_done >= self.max_repeats
            or self.self_loops_done >= self.max_self_loops
            or (
                self.reward_fn_termination and self.reward_fn.should_terminate
            )
        )
        self.truncated = self.steps_done >= self.spec.max_episode_steps

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
        """Returns (observation, terminated, truncated, None)."""
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

        # a1 and a2 should be flattened;
        assert len(a1.shape) == 1 == len(a2.shape)
        tree = KDTree(node_embeds)
        obs_dim = node_embeds.shape[-1]
        action_dim = len(a1)

        # action vectors should have the same
        # length as node embeddings;
        assert action_dim == obs_dim
        data = Data(x=self.x, edge_index=self.edge_index)

        # find node embeddings closest to the action vector(s);
        first = tree.query(a1, k=1)[-1]
        second = tree.query(a2, k=1)[-1]

        # see which nodes will be connected;
        # order matters since the first node is the commiting node
        # and the next is the suitable node;
        info["first"] = first
        info["second"] = second

        # calculate reward;
        idxs = torch.tensor([[first, second]], dtype=torch.long)
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
