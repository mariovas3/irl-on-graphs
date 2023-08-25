import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

import matplotlib.pyplot as plt
from networkx import Graph, draw_networkx

from graph_irl.graph_rl_utils import *

from typing import Tuple
import warnings
from tqdm import tqdm

DO_PLOT = True


class IRLGraphTrainer:
    def __init__(
        self,
        reward_fn,
        reward_optim,
        agent,
        nodes,
        expert_edge_index,
        num_expert_traj,
        graphs_per_batch,
        num_extra_paths_gen=0,
        num_edges_start_from=0,
        reward_optim_lr_scheduler=None,
        reward_grad_clip=False,
        reward_scale=1.,
        per_decision_imp_sample=False,
        weight_scaling_type='abs_max',
        unnorm_policy=False,
        add_expert_to_generated=False,
        lcr_regularisation_coef=None,
        mono_regularisation_on_demo_coef=None,
        verbose=False,
        do_dfs_expert_paths=True,
        num_reward_grad_steps=1,
        ortho_init=True,
    ):
        self.verbose = verbose
        # reward-related params;
        self.reward_fn = reward_fn
        self.reward_optim = reward_optim
        self.reward_optim_lr_scheduler = reward_optim_lr_scheduler
        self.reward_grad_clip = reward_grad_clip
        self.reward_scale = reward_scale
        self.num_reward_grad_steps = num_reward_grad_steps

        # set agent;
        self.agent = agent

        if self.agent.buffer.lcr_reg:
            assert lcr_regularisation_coef is not None

        # set expert example;
        self.nodes = nodes
        self.expert_edge_index = expert_edge_index
        self.adj_list = edge_index_to_adj_list(expert_edge_index[:, ::2], 
                                               len(nodes))
        self.do_dfs_expert_paths = do_dfs_expert_paths

        # sampling params;
        # generated trajectories will match
        # number of expert steps i.e.,
        # gen_steps = num_expert_traj * expert_edge_index.shape[-1] // 2
        # since edges of undirected graphs are duplicated;
        self.num_expert_traj = num_expert_traj
        self.graphs_per_batch = graphs_per_batch
        self.num_extra_paths_gen = num_extra_paths_gen
        # if True, then starting state is some 
        # expert path with num_edges_start_from number of edges;
        # this is so that gnn can actually pass some messages;
        self.num_edges_start_from = num_edges_start_from

        # reward-loss-related params;
        self.per_decision_imp_sample = per_decision_imp_sample
        self.weight_type = 'per_dec' if self.per_decision_imp_sample else 'vanilla'
        self.weight_scaling_type = weight_scaling_type
        self.unnorm_policy = unnorm_policy
        self.lcr_regularisation_coef = lcr_regularisation_coef
        self.mono_regularisation_on_demo_coef = mono_regularisation_on_demo_coef

        # See if should add expert trajectories to sampled ones
        # in the reward loss;
        # still haven't implemented it though; so it's a TODO;
        # it involves more complicated imp samp weights because
        # we now effectively sample from 2 behaviour dists;
        self.add_expert_to_generated = add_expert_to_generated

        # see if should init with (pseudo)orthogonal mats
        self.ortho_init = ortho_init
        if ortho_init:
            self.OI_init_nets()

    def train_policy_k_epochs(self, k, **kwargs):
        self.agent.train_k_epochs(k, **kwargs)
    
    def _get_lcr_loss(self, rewards):
        if rewards.shape[-1] > 2:
            lcr_loss = (
                (
                    (
                        rewards[:, :-2]
                        + rewards[:, 2:]
                        - 2 * rewards[:, 1:-1]
                    ) ** 2 * ((rewards[:, 2:] != 0).detach().int())
                ).sum(-1).mean()
            ) * self.lcr_regularisation_coef
            return lcr_loss
        warnings.warn("expert trajectories less than 3 steps long")
        return 0.

    def do_reward_grad_step(self):
        self.reward_optim.zero_grad()
        mono_loss, lcr_loss1 = 0.0, 0.0

        # get avg(expert_returns)
        expert_avg_returns, expert_rewards = self.get_avg_expert_returns()
        assert expert_rewards.requires_grad
        
        # get rewards for half trajectories;
        if self.num_edges_start_from > 0:
            expert_rewards = expert_rewards[:, self.num_edges_start_from:]
            expert_avg_returns = expert_rewards.sum(-1).mean()
        
        if self.verbose:
            print("expert_rewards shape: ", expert_rewards.shape)

        # see if should penalise to encourage later steps in the expert traj
        # to receive more reward;
        if self.mono_regularisation_on_demo_coef is not None:
            mono_loss = (
                (
                    torch.relu(
                        expert_rewards[:, :-1] - expert_rewards[:, 1:] + 1.
                    ) ** 2
                ).sum(-1).mean()
            ) * self.mono_regularisation_on_demo_coef

        # lcr loss for the expert examples;
        if self.lcr_regularisation_coef is not None:
            lcr_loss1 = self._get_lcr_loss(expert_rewards)

        # get imp_sampled generated returns with stop grad on the weights;
        imp_sampled_gen_rewards, lcr_loss2 = self.get_avg_generated_returns()

        # get loss;
        loss = (
            -(expert_avg_returns - imp_sampled_gen_rewards)
            + mono_loss
            + lcr_loss1
            + lcr_loss2
        )

        # get grads;
        loss.backward()

        print(f"expert avg rewards: {expert_avg_returns.item()}")
        print(f"imp sampled rewards: {imp_sampled_gen_rewards.item()}")
        print(f"mono loss: {mono_loss.item()}")
        print(f"lcr_expert_loss: {lcr_loss1.item()}")
        print(f"lcr_sampled_loss: {lcr_loss2.item()}")
        print(f"overall reward loss: {loss.item()}")
        if self.verbose:
            for p in self.reward_fn.parameters():
                print(
                    f"len module param: {p.shape}",
                    f"l2 norm of grad of params: "
                    f"{torch.norm(p.grad.detach().view(-1), 2).item()}")
            print('\n')
        
        # see if reward grads need clipping;
        if self.reward_grad_clip:
            nn.utils.clip_grad_norm_(
                self.reward_fn.parameters(), 
                max_norm=1.0, 
                error_if_nonfinite=True
            )

        # do grad step;
        self.reward_optim.step()

        # see if lr scheduler present;
        if self.reward_optim_lr_scheduler is not None:
            self.reward_optim_lr_scheduler.step()
        
    def get_avg_generated_returns(self):
        self.weight_processor = WeightsProcessor(
            self.weight_type, 
            self.agent.env.spec.max_episode_steps,
            scaling_type=self.weight_scaling_type,
        )
        if self.per_decision_imp_sample:
            return self._get_per_dec_imp_samp_returns()
        return self._get_vanilla_imp_sampled_returns()

    def _get_per_dec_imp_samp_returns(self):
        assert self.per_decision_imp_sample
        
        # in undirected graph edges are duplicated;
        T = self.expert_edge_index.shape[-1] // 2
        
        # for the step matching;
        n_steps_to_sample = (self.num_expert_traj + self.num_extra_paths_gen) * T
        n_steps_done = 0

        # together with rewards from that episode;
        rewards = []
        
        # avg lcr reularisation over episodes;
        avg_lcr_reg_term, n_episodes = 0.0, 0

        while n_steps_done < n_steps_to_sample:
            (
                r,
                log_w,
                code,
                steps,
                lcr_reg_term,
                obs,
                _,
                _
            ) = self.agent.buffer.get_single_ep_rewards_and_weights(
                self.agent.env,
                self.agent,
            )
            assert len(log_w) == len(r)
            assert len(log_w) >= self.agent.env.min_steps_to_do
            assert r.requires_grad and not log_w.requires_grad
            if self.verbose:
                print(f"per dec sampled return: {r.detach().sum()}")
                print(f"cumsum log weights range: {log_w.min().item()}, "
                      f"{log_w.max().item()}")

            # update avg lcr_reg_term;
            n_episodes += 1

            # update state of weight processor;
            self.weight_processor(log_w)

            # pad all rewards to len = T with 0;
            rewards.append(F.pad(r, (0, T - len(r)), value=0))

            # increment steps;
            n_steps_done += steps

        # stack rewards and get weights;
        rewards = torch.stack(rewards)
        weights, longest = self.weight_processor.get_weights()

        if self.lcr_regularisation_coef is not None:
            avg_lcr_reg_term = self._get_lcr_loss(rewards[:, :longest])
        
        if self.verbose:
            print("actual weights per dec", weights[:, :longest], sep='\n')
            print("sampled rewards shape ", rewards.shape)
            # print(f"effective steps: {longest * len(log_weights)}")
        
        return (weights[:, :longest] * rewards[:, :longest]).sum(), avg_lcr_reg_term

    def _get_vanilla_imp_sampled_returns(self):
        assert not self.per_decision_imp_sample
        T = self.expert_edge_index.shape[-1] // 2
        n_steps_to_sample = (self.num_expert_traj + self.num_extra_paths_gen) * T
        n_steps_done = 0
        returns = []
        avg_lcr_reg_term, n_episodes = 0.0, 0

        # sample steps;
        while n_steps_done < n_steps_to_sample:
            (
                r,
                log_w,
                code,
                steps,
                lcr_reg_term,
                obs,
                _,
                _
            ) = self.agent.buffer.get_single_ep_rewards_and_weights(
                self.agent.env,
                self.agent,
            )
            assert r.requires_grad and not log_w.requires_grad
            if self.verbose:
                print(f"sampled undiscounted return: {r.item()}")
            
            # update avg_lcr_reg_term;
            n_episodes += 1
            avg_lcr_reg_term = (
                avg_lcr_reg_term
                + (lcr_reg_term - avg_lcr_reg_term) / n_episodes
            )

            # update trajectory params;
            self.weight_processor(log_w)
            returns.append(r)

            # house keeping;
            n_steps_done += steps
        
        # stack returns and get weights;
        returns = torch.stack(returns)
        weights = self.weight_processor.get_weights()
        if self.verbose:
            print("actual weights vanilla", weights, sep='\n')
        return returns @ weights, avg_lcr_reg_term

    def get_avg_expert_returns(self) -> Tuple[float, torch.Tensor]:
        """
        Returns avg returns over self.num_expert_traj trajectories
        together with all rewards along the trajectories in
        (num_expert_trajectories, episode_length) shaped torch.Tensor.

        Note: This is if I want to add the expert trajectories in the
                imp sampled term similar to the GCL paper.
        """
        avg = 0.0
        N = 0
        cached_expert_rewards = []
        for _ in range(self.num_expert_traj):
            if self.do_dfs_expert_paths:
                source = np.random.randint(0, len(self.nodes))
                R, rewards = self._get_single_ep_dfs_return(source)
            else:
                R, rewards = self._get_single_ep_expert_return()
            if self.verbose:
                print(f"expert return: {R}")
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
        return self._get_return_and_rewards_on_path(
            self.expert_edge_index, idxs
        )
    
    def _get_single_ep_dfs_return(self, source):
        edge_index = get_dfs_edge_order(self.adj_list, source)
        assert edge_index.shape == self.expert_edge_index.shape
        idxs = list(range(edge_index.shape[-1]))
        return self._get_return_and_rewards_on_path(edge_index, idxs)
    
    def _get_return_and_rewards_on_path(self, edge_index, idxs):
        pointer = 0
        return_val = 0.0
        cached_rewards = []
        # the * 2 multiplier is because undirected graph is assumed
        # and effectively each edge is duplicated in edge_index;
        while (
            pointer * self.graphs_per_batch * 2
            < edge_index.shape[-1]
        ):
            batch_list = []
            action_idxs = []
            for i in range(
                pointer * self.graphs_per_batch * 2,
                min(
                    (pointer + 1) * self.graphs_per_batch * 2,
                    edge_index.shape[-1],
                ),
                2,
            ):
                batch_list.append(
                    Data(
                        x=self.nodes,
                        edge_index=edge_index[
                            :, idxs[pointer * self.graphs_per_batch * 2 : i]
                        ],
                    )
                )
                first, second = (
                    edge_index[0, idxs[i]],
                    edge_index[1, idxs[i]],
                )
                action_idxs.append([first, second])
                if self.agent.buffer.state_reward:
                    batch_list[-1].edge_index = torch.cat(
                        (
                            batch_list[-1].edge_index, 
                            torch.tensor([[first, second], 
                                            [second, first]], dtype=torch.long)
                        ), -1
                    )
    
            # create batch of graphs;
            batch = Batch.from_data_list(batch_list)
            extra_graph_level_feats = None
            if self.agent.buffer.transform_ is not None:
                self.agent.buffer.transform_(batch)
                if self.agent.buffer.transform_.get_graph_level_feats_fn is not None:
                    extra_graph_level_feats = self.agent.buffer.transform_.get_graph_level_feats_fn(batch)

            pointer += 1
            if self.agent.buffer.state_reward:
                curr_rewards = self.reward_fn(
                    batch, extra_graph_level_feats,
                ).view(-1) * self.reward_scale
            else:
                curr_rewards = self.reward_fn(
                    (batch, torch.tensor(action_idxs)), 
                    extra_graph_level_feats,
                    action_is_index=True
                ).view(-1) * self.reward_scale
            return_val += curr_rewards.sum()
            cached_rewards.append(curr_rewards)
        
        global DO_PLOT
        if DO_PLOT:
            G = Graph()
            G.add_edges_from(list(zip(*batch_list[-1].edge_index.tolist())))
            fig = plt.figure()
            draw_networkx(G)
            plt.savefig('expert_example.png')
            plt.close()
            DO_PLOT = False
        return return_val, torch.cat(cached_rewards)

    def train_irl(self, num_iters, policy_epochs, **kwargs):
        buffer_verbose = self.agent.buffer.verbose
        for it in tqdm(range(num_iters)):
            print(f"IRL TRAINER ITER {it+1}:\n------------------------")
            # put reward fn in train mode and policy in eval mode
            # for the reward update step;
            self.agent.buffer.verbose = buffer_verbose
            self.reward_fn.requires_grad_(True)
            self.reward_fn.train()
            self.agent.policy.requires_grad_(False)
            for _ in range(self.num_reward_grad_steps):
                self.do_reward_grad_step()

            # when training policy, set policy to train mode
            # and reward to eval mode;
            self.agent.buffer.verbose = False
            self.reward_fn.requires_grad_(False)
            self.reward_fn.eval()
            self.agent.policy.requires_grad_(True)
            self.agent.policy.eval()
            self.train_policy_k_epochs(policy_epochs, **kwargs)
    
    def OI_init_nets(self):
        OI_init(self.reward_fn)
        self.agent.OI_init_nets()
