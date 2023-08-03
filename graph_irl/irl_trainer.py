import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

from typing import Tuple
import warnings
from tqdm import tqdm


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
        reward_optim_lr_scheduler=None,
        reward_grad_clip=False,
        reward_scale=1.,
        per_decision_imp_sample=False,
        add_expert_to_generated=False,
        lcr_regularisation_coef=None,
        mono_regularisation_on_demo_coef=None,
        verbose=False,
    ):
        self.verbose = verbose

        # reward-related params;
        self.reward_fn = reward_fn
        self.reward_optim = reward_optim
        self.reward_optim_lr_scheduler = reward_optim_lr_scheduler
        self.reward_grad_clip = reward_grad_clip
        self.reward_scale = reward_scale

        # set agent;
        self.agent = agent

        # set expert example;
        self.nodes = nodes
        self.expert_edge_index = expert_edge_index

        # sampling params;
        # generated trajectories will match
        # number of expert steps i.e.,
        # gen_steps = num_expert_traj * expert_edge_index.shape[-1] // 2
        # since edges of undirected graphs are duplicated;
        self.num_expert_traj = num_expert_traj
        self.graphs_per_batch = graphs_per_batch

        # reward-loss-related params;
        self.per_decision_imp_sample = per_decision_imp_sample
        self.lcr_regularisation_coef = lcr_regularisation_coef
        self.mono_regularisation_on_demo_coef = mono_regularisation_on_demo_coef

        # See if should add expert trajectories to sampled ones
        # in the reward loss;
        # still haven't implemented it though; so it's a TODO;
        # it involves more complicated imp samp weights because
        # we now effectively sample from 2 behaviour dists;
        self.add_expert_to_generated = add_expert_to_generated

    def train_policy_k_epochs(self, k, **kwargs):
        self.agent.train_k_epochs(k, **kwargs)

    def do_reward_grad_step(self):
        self.reward_optim.zero_grad()
        mono_loss, lcr_loss1 = 0.0, 0.0

        # get avg(expert_returns)
        expert_avg_returns, expert_rewards = self.get_avg_expert_returns()
        assert expert_rewards.requires_grad

        # see if should penalise to encourage later steps in the expert traj
        # to receive more reward;
        if self.mono_regularisation_on_demo_coef is not None:
            mono_loss = (
                (
                    torch.relu(
                        expert_rewards[:, :-1] - expert_rewards[:, 1:] - 1.0
                    )
                    ** 2
                )
                .sum(-1)
                .mean()
            ) * self.mono_regularisation_on_demo_coef

        # lcr loss for the expert examples;
        if self.lcr_regularisation_coef is not None:
            if expert_rewards.shape[-1] > 2:
                lcr_loss1 = (
                    (
                        (
                            expert_rewards[:, :-2]
                            + expert_rewards[:, 2:]
                            - 2 * expert_rewards[:, 1:-1]
                        )
                        ** 2
                    )
                    .sum(-1)
                    .mean()
                ) * self.lcr_regularisation_coef
            else:
                warnings.warn("expert trajectories less than 3 steps long")

        # get imp_sampled generated returns with stop grad on the weights;
        imp_sampled_gen_rewards, lcr_loss2 = self.get_avg_generated_returns()
        if self.lcr_regularisation_coef is not None:
            lcr_loss2 *= self.lcr_regularisation_coef

        # get loss;
        loss = (
            -(expert_avg_returns - imp_sampled_gen_rewards)
            + mono_loss
            + lcr_loss1
            + lcr_loss2
        )

        # get grads;
        loss.backward()

        if self.verbose:
            print(f"expert avg rewards: {expert_avg_returns.item()}")
            print(f"imp sampled rewards: {imp_sampled_gen_rewards.item()}")
            print(f"mono loss: {mono_loss}")
            print(f"lcr_expert_loss: {lcr_loss1}")
            print(f"lcr_sampled_loss: {lcr_loss2}")
            print(f"overall reward loss: {loss.item()}")
            for p in self.reward_fn.parameters():
                print(
                    f"len module param: {len(p)}",
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

        # clear buffer since reward changed, making
        # previousely sampled trajectories invalid;
        # also empty lists with losses and other tracked metrics;
        self.agent.clear_buffer()

    def get_avg_generated_returns(self):
        if self.per_decision_imp_sample:
            return self._get_per_dec_imp_samp_returns()
        return self._get_vanilla_imp_sampled_returns()

    def _get_per_dec_imp_samp_returns(self):
        assert self.per_decision_imp_sample
        
        # in undirected graph edges are duplicated;
        T = self.expert_edge_index.shape[-1] // 2
        
        # for the step matching;
        n_steps_to_sample = self.num_expert_traj * T
        n_steps_done, longest = 0, 0

        weights, rewards = [], []
        
        # avg lcr reularisation over episodes;
        avg_lcr_reg_term, n_episodes = 0.0, 0

        while n_steps_done < n_steps_to_sample:
            (
                r,
                w,
                code,
                steps,
                lcr_reg_term,
            ) = self.agent.buffer.get_single_ep_rewards_and_weights(
                self.agent.env,
                self.agent,
                self.lcr_regularisation_coef is not None,
            )
            assert len(w) == len(r)
            assert r.requires_grad and not w.requires_grad
            # print(steps, lcr_reg_term)

            # update avg lcr_reg_term;
            n_episodes += 1
            avg_lcr_reg_term = (
                avg_lcr_reg_term
                + (lcr_reg_term - avg_lcr_reg_term) / n_episodes
            )

            # increment steps;
            n_steps_done += steps

            # pad all weights to the right with zeros;
            weights.append(F.pad(w, (0, T - len(w)), value=0))
            # if self.verbose:
                # print(w)

            # pad all rewards to len = T;
            rewards.append(F.pad(r, (0, T - len(r)), value=0))

            # keep track of len of longest episode;
            longest = max(len(w), longest)
            # print(len(w), longest)

        weights = torch.stack(weights)
        weights[:, :longest] = weights[:, :longest] / weights[:, :longest].sum(
            0, keepdim=True
        )
        print(weights[:, :longest].numpy().round(3))

        # if self.verbose:
            # print("weights per dec", weights, sep='\n')
        
        assert torch.allclose(weights[:, :longest].sum(0), torch.ones((1, )))
        return (weights * torch.stack(rewards)).sum(), avg_lcr_reg_term

    def _get_vanilla_imp_sampled_returns(self):
        assert not self.per_decision_imp_sample
        T = self.expert_edge_index.shape[-1] // 2
        n_steps_to_sample = self.num_expert_traj * T
        n_steps_done = 0
        sum_w = 0.0
        returns, ws = [], []
        avg_lcr_reg_term, n_episodes = 0.0, 0

        # sample steps;
        while n_steps_done < n_steps_to_sample:
            (
                r,
                w,
                code,
                steps,
                lcr_reg_term,
            ) = self.agent.buffer.get_single_ep_rewards_and_weights(
                self.agent.env,
                self.agent,
                self.lcr_regularisation_coef is not None,
            )
            # print(steps, lcr_reg_term)
            assert r.requires_grad and not w.requires_grad
            
            # update avg_lcr_reg_term;
            n_episodes += 1
            avg_lcr_reg_term = (
                avg_lcr_reg_term
                + (lcr_reg_term - avg_lcr_reg_term) / n_episodes
            )

            # update trajectory params;
            sum_w += w
            ws.append(w)
            returns.append(r)

            # house keeping;
            n_steps_done += steps
        
        if self.verbose:
            print("weights vanilla", ws, sep='\n')
            
        return (
            torch.stack(returns) * (torch.stack(ws) / sum_w)
        ).sum(), avg_lcr_reg_term

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

        # put episode in batches of no more than self.graphs_per_batch
        # graphs. Deep Learning is usually faster when applied to batches
        # rather than single input at a time.
        pointer = 0
        return_val = 0.0
        cached_rewards = []
        # the * 2 multiplier is because undirected graph is assumed
        # and effectively each edge is duplicated in edge_index;
        while (
            pointer * self.graphs_per_batch * 2
            < self.expert_edge_index.shape[-1]
        ):
            batch_list = []
            action_idxs = []
            for i in range(
                pointer * self.graphs_per_batch * 2,
                min(
                    (pointer + 1) * self.graphs_per_batch * 2,
                    self.expert_edge_index.shape[-1],
                ),
                2,
            ):
                batch_list.append(
                    Data(
                        x=self.nodes,
                        edge_index=self.expert_edge_index[
                            :, idxs[pointer * self.graphs_per_batch * 2 : i]
                        ],
                    )
                )
                first, second = (
                    self.expert_edge_index[0, idxs[i]],
                    self.expert_edge_index[1, idxs[i]],
                )
                action_idxs.append([first, second])

            # create batch of graphs;
            batch = Batch.from_data_list(batch_list)
            pointer += 1
            curr_rewards = self.reward_fn(
                (batch, torch.tensor(action_idxs)), action_is_index=True
            ).view(-1) * self.reward_scale
            return_val += curr_rewards.sum()
            cached_rewards.append(curr_rewards)
        return return_val, torch.cat(cached_rewards)

    def train_irl(self, num_iters, policy_epochs, **kwargs):
        for _ in tqdm(range(num_iters)):
            self.do_reward_grad_step()
            self.train_policy_k_epochs(policy_epochs, **kwargs)
