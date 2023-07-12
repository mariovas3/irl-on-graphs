import numpy as np
from scipy.spatial import KDTree
import torch
from torch_geometric.data import Data


class GraphEnv:
    def __init__(self, x: torch.Tensor, 
                 max_episode_steps, 
                 num_expert_steps,
                 max_repeats):
        """
        Args:
            x (torch.Tensor): Initial node features.
            max_episode_steps (int): Truncation length.
            num_expert_steps (int): Termination length.
            max_repeats (int): Truncate if add >= max_repeats existing edges.
        """
        self.x = x 
        self.data = Data(x=x, 
                         edge_index=torch.tensor([[], []], dtype=torch.long))
        self.unique_edges = set()
        self.max_episode_steps = max_episode_steps
        self.num_expert_steps = num_expert_steps
        self.steps_done = 0
        self.max_repeats = max_repeats
        self.repeats_done = 0
        self.terminated, self.truncated = False, False
        self.num_self_loops = 0
    
    def reset(self):
        """Returns (observation, None) to be compatible with openai gym."""
        self.data.edge_index = torch.tensor([[], []], dtype=torch.long)
        self.unique_edges = set()
        self.steps_done = 0
        self.repeats_done = 0
        self.terminated, self.truncated = False, False
        self.num_self_loops = 0
        return self.data, None
    
    def step(self, a: np.ndarray, x_embeds: np.ndarray):
        """Returns (observation, terminated, truncated, None)."""
        assert not (self.terminated or self.truncated)
        info = {
            'terminated': False,
            'expert_episode_length_reached': False,
            'truncated': False,
            'max_repeats_reached': False,
            'episode_truncation_length_reached': False,
            'self_loop': False,
            'steps_done': self.steps_done,
            'repeats_done': self.repeats_done,
            'num_self_loops': self.num_self_loops
        }

        tree = KDTree(x_embeds)
        obs_dim = x_embeds.shape[-1]
        action_dim = len(a)
        
        # graph opt like edge addition;
        # connects point closes to a1 to point 
        # closest to a2, these could be very far apart.
        # also not clear to me how NN will figure out
        # which ones to connect since they share a common embedding;
        # and a1 e.g., picks a point to connect, and a2 should depend
        # on the selection for a1.
        # I think Victor's 2 stage mechanism makes more sense
        # i.e., first select x1 = closest(x, a1)
        # and then incorporate that info somehow (mby distance weight)
        # like softmax to select x2 = func(x, dist_weights);
        if action_dim == 2 * obs_dim:
            a1, a2 = a[:obs_dim], a[obs_dim:]
            first = tree.query(a1, k=1)[-1]
            second = tree.query(a2, k=1)[-1]
        else:
            idxs = tree.query(a, k=2)[-1]
            first, second = idxs[0], idxs[1]
        
        if first == second:
            self.num_self_loops += 1
            info['self_loop'] = True
            info['repeats_done'] = self.repeats_done
            info['steps_done'] = self.steps_done
            info['num_self_loops'] = self.num_self_loops
            return self.data, False, False, info
        
        if (first, second) in self.unique_edges or (second, first) in self.unique_edges:
            self.repeats_done += 1
            if self.repeats_done >= self.max_repeats:
                info['max_repeats_reached'] = True
                info['truncated'] = True
                info['repeats_done'] = self.repeats_done
                info['steps_done'] = self.steps_done
                info['num_self_loops'] = self.num_self_loops
                return self.data, False, True, info
        else:
            # add undirected edge to the graph Data container;
            self.data.edge_index = torch.cat(
                    (
                        self.data.edge_index, 
                        torch.tensor([(first, second), (second, first)], 
                                    dtype=torch.long)
                    ), -1
                )
            
            # increment steps done;
            self.steps_done += 1

            # add edge to set of edges;
            self.unique_edges.add((first, second))
            self.unique_edges.add((second, first))
        
        # calculate outcomes;
        terminated = self.steps_done >= self.num_expert_steps
        truncated = self.steps_done >= self.max_episode_steps or self.repeats_done >= self.max_repeats
        info['terminated'] = terminated
        info['truncated'] = truncated
        info['episode_truncation_length_reached'] = self.steps_done >= self.max_episode_steps
        info['max_repeats_reached'] = self.repeats_done >= self.max_repeats
        info['expert_episode_length_reached'] = terminated
        info['steps_done'] = self.steps_done
        info['repeats_done'] = self.repeats_done
        info['num_self_loops'] = self.num_self_loops
        self.terminated, self.truncated = terminated, truncated
        return self.data, terminated, truncated, info
