import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
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
            per_decision_imp_sample=False,
            match_expert_step_count=False,
            add_expert_to_generated=False,
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
        self.per_decision_imp_sample = per_decision_imp_sample

        # see if should match expert step count rather than 
        # just generating num_generated_traj;
        # this is because in the beginning the policy may be bad
        # and the generated episodes may be short 
        # this can happen if e.g., there are too many edge repeats
        self.match_expert_step_count = match_expert_step_count

        # this is what the GCL paper does;
        # still haven't implemented it though; so it's a TODO;
        # it involves more complicated imp samp weights because
        # we now effectively sample from 2 dists;
        self.add_expert_to_generated = add_expert_to_generated

    def train_policy_k_epochs(self, k, **kwargs):
        self.agent.train_k_epochs(k, **kwargs)
    
    def do_reward_grad_step(self):
        self.reward_optim.zero_grad()
        
        # get avg(expert_returns)
        expert_avg_returns, expert_rewards = self.get_avg_expert_returns()
        
        # get imp_sampled generated returns with stop grad on the weights;
        imp_sampled_gen_rewards = self.get_avg_generated_returns()
        
        # get loss;
        loss = - (expert_avg_returns - imp_sampled_gen_rewards)
        loss.backward()
        if self.reward_grad_clip:
            nn.utils.clip_grad_norm_(self.reward_fn.parameters(), max_norm=1.)

        # optimise;
        self.reward_optim.step()
        if self.reward_optim_lr_scheduler is not None:
            self.reward_optim_lr_scheduler.step()
        
        # clear buffer since reward changed making
        # previousely sampled trajectories invalid;
        self.agent.clear_buffer()
    
    def get_avg_generated_returns(self):
        if self.per_decision_imp_sample:
            return self._get_per_dec_imp_samp_returns()
        return self._get_vanilla_imp_sampled_returns()
    
    def _get_per_dec_imp_samp_returns(self):
        T = self.expert_edge_index.shape[-1]
        n_steps_to_sample = self.num_expert_traj * T
        n_steps_done = 0
        weights = []
        rewards = []
        longest = 0
        while n_steps_done < n_steps_to_sample:
            r, w, code, steps = self.agent.buffer.get_single_ep_rewards_and_weights(
                self.agent.env,
                self.agent
            )
            assert len(w) == len(r)
            assert r.requires_grad and not w.requires_grad
            
            # increment steps;
            n_steps_done += steps

            # pad all weights to the right with zeros;
            weights.append(F.pad(w, (0, T - len(w)), value=0))
            
            # pad all rewards to len = T;
            rewards.append(F.pad(r, (0, T - len(r)), value=0))
            
            longest = max(len(w), longest)

        weights = torch.stack(weights)
        weights[:, :longest] = weights[:, :longest] / weights[:, :longest].sum(0, keepdim=True)
        return (weights * torch.stack(rewards)).sum()
    
    def _get_vanilla_imp_sampled_returns(self):
        T = self.expert_edge_index.shape[-1]
        n_steps_to_sample = self.num_expert_traj * T
        n_steps_done = 0
        sum_w = 0.
        returns = []
        ws = []
        while n_steps_done < n_steps_to_sample:
            r, w, code, steps = self.agent.buffer.get_single_ep_rewards_and_weights(
                self.agent.env,
                self.agent
            )
            assert r.requires_grad and not w.requires_grad
            n_steps_done += steps
            sum_w += w
            ws.append(w)
            returns.append(r)
        return (torch.stack(returns) * (torch.stack(ws) / sum_w)).sum()
    
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
