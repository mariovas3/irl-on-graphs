from scipy.spatial import KDTree
import torch
from typing import Callable
from torch_geometric.data import Data


class GraphEnv:
    def __init__(
        self,
        x: torch.Tensor,
        reward_fn: Callable,
        max_episode_steps: int,
        num_expert_steps: int,
        max_repeats: int,
        drop_repeats_or_self_loops: bool = False,
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
        self.x = x
        self.max_episode_steps = max_episode_steps
        self.num_expert_steps = num_expert_steps
        self.drop_repeats_or_self_loops = drop_repeats_or_self_loops
        self.max_repeats = max_repeats
        # reward fn should have its own GNN encoder;
        # due to deterministic transitions, make reward
        # state dependent only, since S x A -> S deterministically.
        self.reward_fn = reward_fn

        # attributes to be reset when reset called;
        self.edge_index = torch.tensor([[], []], dtype=torch.long)
        self.unique_edges = set()
        self.steps_done = 0
        self.repeats_done = 0
        self.terminated, self.truncated = False, False
        self.num_self_loops = 0

    def reset(self, seed=0):
        """
        Returns (observation, None) to be compatible with openai gym.
        Also the seed parameter is given for compatibility but is not
        used.
        """
        data = Data(
            x=self.x, edge_index=torch.tensor([[], []], dtype=torch.long)
        )
        self.edge_index = torch.tensor([[], []], dtype=torch.long)
        self.unique_edges = set()
        self.steps_done = 0
        self.repeats_done = 0
        self.terminated, self.truncated = False, False
        self.num_self_loops = 0
        return data, None

    def _update_info_terminals(self, info):
        self.terminated = (
            self.steps_done >= self.num_expert_steps
            or self.repeats_done >= self.max_repeats
        )
        self.truncated = self.steps_done >= self.max_episode_steps

        info["terminated"] = self.terminated
        info["expert_episode_length_reached"] = (
            self.steps_done >= self.num_expert_steps
        )
        info["max_repeats_reached"] = self.repeats_done >= self.max_repeats
        info["truncated"] = self.truncated
        info["episode_truncation_length_reached"] = (
            self.steps_done >= self.max_episode_steps
        )
        info["steps_done"] = self.steps_done
        info["repeats_done"] = self.repeats_done
        info["num_self_loops"] = self.num_self_loops

    def step(self, action):
        """Returns (observation, terminated, truncated, None)."""
        assert not (self.terminated or self.truncated)
        self.steps_done += 1
        info = {
            "terminated": False,
            "expert_episode_length_reached": False,
            "max_repeats_reached": False,
            "truncated": False,
            "episode_truncation_length_reached": False,
            "self_loop": False,
            "is_repeat": False,
            "steps_done": self.steps_done,
            "repeats_done": self.repeats_done,
            "num_self_loops": self.num_self_loops,
            "first": None,
            "second": None,
        }

        # unpack action;
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

        if first == second:
            self.num_self_loops += 1
            info["self_loop"] = True
            if self.drop_repeats_or_self_loops:
                self.steps_done -= 1
            self._update_info_terminals(info)
            return (
                data,
                self.reward_fn(data),
                self.terminated,
                self.truncated,
                info,
            )

        if (first, second) in self.unique_edges or (
            second,
            first,
        ) in self.unique_edges:
            self.repeats_done += 1
            info["is_repeat"] = True
            if self.drop_repeats_or_self_loops:
                self.steps_done -= 1
            self._update_info_terminals(info)
            return (
                data,
                self.reward_fn(data),
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
            self.unique_edges.add((first, second))
            self.unique_edges.add((second, first))

        # update info;
        self._update_info_terminals(info)

        # reward fn should have its own encoder GNN;
        return (
            data,
            self.reward_fn(data),
            self.terminated,
            self.truncated,
            info,
        )
