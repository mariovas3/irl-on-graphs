import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data, Batch

from typing import Tuple


class IRLGraphTrainer:
    def __init__(
            self, 
            reward_fn, 
            reward_optim, 
            agent,
            nodes, 
            expert_edge_index, 
            num_expert_traj, 
            num_generated_traj,
            num_graphs_in_batch, 
            reward_optim_lr_scheduler=None,
            reward_grad_clip=False,
    ):
        self.reward_fn = reward_fn
        self.reward_optim = reward_optim
        self.reward_optim_lr_scheduler = reward_optim_lr_scheduler
        self.reward_grad_clip = reward_grad_clip
        self.agent = agent
        self.nodes = nodes
        self.expert_edge_index = expert_edge_index

        self.num_expert_traj = num_expert_traj
        self.num_generated_traj = num_generated_traj
        self.num_graphs_in_batch = num_graphs_in_batch
        
    
    def train_policy_k_epochs(self, k, **kwargs):
        self.agent.train_k_epochs(k, **kwargs)
    
    def do_reward_grad_step(
            self, 
    ):
        self.reward_optim.zero_grad()
        
        # get avg(expert_returns)
        expert_avg_returns, expert_rewards = self.get_avg_expert_returns()
        
        # get imp_sampled generated returns with stop grad on the weights;
        imp_samples_gen_rewards = self.get_avg_generated_returns()
        
        # get loss;
        loss = - (expert_avg_returns - imp_samples_gen_rewards)
        loss.backward()
        if self.reward_grad_clip:
            nn.utils.clip_grad_norm_(self.reward_fn.parameters(), max_norm=1.)

        # optimise;
        self.reward_optim.step()
        if self.reward_optim_lr_scheduler is not None:
            self.reward_optim_lr_scheduler.step()
    
    def get_avg_expert_returns(self) -> Tuple[float, torch.Tensor]:
        """
        Returns avg returns over self.num_expert_traj trajectories
        together with all rewards along the trajectories in 
        (num_expert_trajectories, episode_length) shaped torch.Tensor.

        Note: This is if I want to add the expert trajectories in the 
                imp sampled term similar to the GCL paper.
        """
        avg = 0.
        N = 0
        cached_expert_rewards = []
        for _ in range(self.num_expert_traj):
            R, rewards = self._get_single_ep_expert_return()
            N += 1
            avg = avg + (R - avg) / N
            cached_expert_rewards.append(rewards)
        return avg, torch.stack(cached_expert_rewards)
    
    def _get_single_ep_expert_return(self) -> Tuple[float, torch.Tensor]:
        """
        Returns sum of rewards along single expert trajectory as well as all 
        rewards along the trajectory in 1-D torch.Tensor.
        """
        # permute even indexes of edge index;
        # edge index is assumed to correspond to an undirected graph;
        perm = np.random.permutation(
            range(0, self.expert_edge_index.shape[-1], 2)
        )
        # expert_edge_index[:, (even_index, even_index + 1)] should 
        # correspond to the same edge in the undirected graph in torch_geom.
        # this is because in torch_geometric, undirected graph duplicates 
        # each edge e.g., (from, to), (to, from) are both in edge_index;
        idxs = sum([(i, i + 1) for i in perm], ())

        # put episode in batches of no more than self.num_graphs_in_batch
        # graphs. Deep Learning is usually faster when applied to batches
        # rather than single input at a time.
        pointer = 0
        return_val = 0.
        cached_rewards = []
        # the * 2 multiplier is because undirected graph is assumed
        # and effectively each edge is duplicated in edge_index;
        while pointer * self.num_graphs_in_batch * 2 < self.expert_edge_index.shape[-1]:
            batch_list = []
            action_idxs = []
            for i in range(pointer * self.num_graphs_in_batch * 2,
                           min(
                            (pointer + 1) * self.num_graphs_in_batch * 2, 
                            self.expert_edge_index.shape[-1]
                           ), 2):
                batch_list.append(
                    Data(x=self.nodes, 
                         edge_index=self.expert_edge_index[:, 
                                                           idxs[pointer * self.num_graphs_in_batch * 2:i]])
                )
                first, second = self.expert_edge_index[0, idxs[i]], self.expert_edge_index[1, idxs[i]]
                action_idxs.append([first, second])
            
            # create batch of graphs;
            batch = Batch.from_data_list(batch_list)
            pointer += 1
            curr_rewards = self.reward_fn((batch, torch.tensor(action_idxs)), action_is_index=True).view(-1)
            return_val += curr_rewards.sum()
            cached_rewards.append(curr_rewards)
        return return_val, torch.cat(cached_rewards)



    def train_irl(self, num_iters, policy_epochs, **kwargs):
        for _ in range(num_iters):
            self.do_reward_grad_step()
            self.train_policy_k_epochs(policy_epochs, **kwargs)
