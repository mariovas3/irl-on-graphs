import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import torch_geometric.nn as tgnn

import matplotlib.pyplot as plt
from networkx import Graph, draw_networkx

from graph_irl.graph_rl_utils import *
from graph_irl.sac import save_metric_plots

from typing import Tuple
import warnings
import pickle
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
        reward_scale=1.0,
        per_decision_imp_sample=False,
        weight_scaling_type="abs_max",
        unnorm_policy=False,
        add_expert_to_generated=False,
        lcr_regularisation_coef=None,
        mono_regularisation_on_demo_coef=None,
        verbose=False,
        do_dfs_expert_paths=True,
        num_reward_grad_steps=1,
        ortho_init=True,
        do_graphopt=False,
        zero_interm_rew=False,
        quad_reward_penalty=None,
        reward_l2_coef=None,
    ):
        self.reward_l2_coef = reward_l2_coef
        self.quad_reward_penalty = quad_reward_penalty
        self.zero_interm_rew = zero_interm_rew
        self.do_graphopt = do_graphopt
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
        self.seed = self.agent.buffer.seed

        if self.agent.buffer.lcr_reg:
            assert lcr_regularisation_coef is not None

        # set expert example;
        self.nodes = nodes
        self.expert_edge_index = expert_edge_index
        self.adj_list = edge_index_to_adj_list(
            expert_edge_index[:, ::2], len(nodes)
        )
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
        self.weight_type = (
            "per_dec" if self.per_decision_imp_sample else "vanilla"
        )
        self.weight_scaling_type = weight_scaling_type
        self.unnorm_policy = unnorm_policy
        self.lcr_regularisation_coef = lcr_regularisation_coef
        self.mono_regularisation_on_demo_coef = (
            mono_regularisation_on_demo_coef
        )

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

        # saving stuff;
        self.expert_avg_returns = []
        self.imp_sampled_returns = []
        self.mono_losses = []
        self.lcr_expert_losses = []
        self.lcr_sampled_losses = []
        self.gradients = []
        self.mtt_losses = []
        self.quad_expert_losses = []
        self.quad_sampled_losses = []
        self.reward_l2_losses = []

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
                    )
                    ** 2
                    * ((rewards[:, 2:] != 0).detach().int())
                )
                .sum(-1)
                .mean()
            ) * self.lcr_regularisation_coef
            return lcr_loss
        warnings.warn("expert trajectories less than 3 steps long")
        return 0.0

    def get_source_graph_reward(self):
        source_graph = Data(
            x=self.nodes, edge_index=self.expert_edge_index
        )
        extra_graph_level_feats = None
        if self.agent.buffer.transform_ is not None:
            self.agent.buffer.transform_(source_graph)
            if (
                self.agent.buffer.transform_.get_graph_level_feats_fn
                is not None
            ):
                extra_graph_level_feats = (
                    self.agent.buffer.transform_.get_graph_level_feats_fn(
                        source_graph
                    )
                )

        if self.do_graphopt:
            out = self.agent.old_encoder(
                source_graph, extra_graph_level_feats
            )[0]
            curr_rewards = self.reward_fn(out)
        else:
            curr_rewards = self.reward_fn(
                source_graph,
                extra_graph_level_feats,
                get_graph_embeds=False,
            )
        curr_rewards = curr_rewards.view(-1) * self.reward_scale
        return curr_rewards, None, None, None

    def do_reward_grad_step(self):
        self.reward_optim.zero_grad()
        mono_loss, lcr_loss1 = 0.0, 0.0
        quad_expert = 0.0

        if self.zero_interm_rew:
            (
                expert_avg_returns,
                expert_rewards,
                tot_mse1,
                tot_len1,
            ) = self.get_source_graph_reward()
        else:
            # get avg(expert_returns)
            (
                expert_avg_returns,
                expert_rewards,
                tot_mse1,
                tot_len1,
            ) = self.get_avg_expert_returns()
            assert expert_rewards.requires_grad

        # get rewards for truncated trajectories;
        if not self.zero_interm_rew and self.num_edges_start_from > 0:
            expert_rewards = expert_rewards[:, self.num_edges_start_from :]
            expert_avg_returns = expert_rewards.sum(-1).mean()

        if not self.zero_interm_rew and self.verbose:
            print("expert_rewards shape: ", expert_rewards.shape)

        # see if should penalise to encourage later steps in the expert traj
        # to receive more reward;
        if (
            not self.zero_interm_rew
            and self.mono_regularisation_on_demo_coef is not None
        ):
            mono_loss = (
                (
                    torch.relu(
                        expert_rewards[:, :-1] - expert_rewards[:, 1:]
                    )
                    ** 2
                )
                .sum(-1)
                .mean()
            ) * self.mono_regularisation_on_demo_coef

        # lcr loss for the expert examples;
        if (
            not self.zero_interm_rew
            and self.lcr_regularisation_coef is not None
        ):
            lcr_loss1 = self._get_lcr_loss(expert_rewards)

        # get imp_sampled generated returns with stop grad on the weights;
        (
            imp_sampled_gen_rewards,
            lcr_loss2,
            tot_mse2,
            tot_len2,
            quad_sampled,
        ) = self.get_avg_generated_returns()

        if (
            not self.zero_interm_rew
            and self.quad_reward_penalty is not None
        ):
            quad_expert = (expert_rewards**2).sum(
                -1
            ).mean() * self.quad_reward_penalty
            quad_sampled = quad_sampled * self.quad_reward_penalty

        l2_loss = 0.0
        if self.reward_l2_coef is not None:
            l2_loss = self.reward_l2_coef * (
                torch.cat(
                    [
                        p.view(-1)
                        for p in self.reward_fn.net[-1].parameters()
                    ]
                )
                ** 2
            ).sum(-1)
        # get loss;
        loss = (
            -(expert_avg_returns - imp_sampled_gen_rewards)
            + mono_loss
            + lcr_loss1
            + lcr_loss2
            + quad_expert
            + quad_sampled
            + l2_loss
        )

        # get grads;
        loss.backward(retain_graph=self.agent.multitask_net is not None)

        # multitask loss;
        if (
            self.agent.multitask_net is not None
            and not self.zero_interm_rew
            and not self.do_graphopt
        ):
            self.agent.optim_multitask_net.zero_grad()
            mtt_loss = tot_mse1 * (
                tot_len1 / (tot_len1 + tot_len2)
            ) + tot_mse2 * (tot_len2 / (tot_len1 + tot_len2))
            mtt_loss = self.agent.multitask_coef * mtt_loss
            self.mtt_losses.append(mtt_loss.item())
            print(f"multitask gnn loss: {self.mtt_losses[-1]}")
            mtt_loss.backward()

        self.expert_avg_returns.append(expert_avg_returns.item())
        self.imp_sampled_returns.append(imp_sampled_gen_rewards.item())
        if self.mono_regularisation_on_demo_coef is not None:
            self.mono_losses.append(mono_loss.item())
        if self.lcr_regularisation_coef is not None:
            self.lcr_expert_losses.append(lcr_loss1.item())
            self.lcr_sampled_losses.append(lcr_loss2.item())
        if self.quad_reward_penalty is not None:
            self.quad_expert_losses.append(quad_expert.item())
            self.quad_sampled_losses.append(quad_sampled.item())
        if self.reward_l2_coef is not None:
            self.reward_l2_losses.append(l2_loss.item())

        print(f"expert avg rewards: {expert_avg_returns.item()}")
        print(f"imp sampled rewards: {imp_sampled_gen_rewards.item()}")
        if self.mono_regularisation_on_demo_coef is not None:
            print(f"mono loss: {mono_loss.item()}")
        if self.lcr_regularisation_coef is not None:
            print(f"lcr_expert_loss: {lcr_loss1.item()}")
            print(f"lcr_sampled_loss: {lcr_loss2.item()}")
        if self.quad_reward_penalty is not None:
            print(f"quad_expert reward loss: {quad_expert.item()}")
            print(f"quad_sampled reward loss: {quad_sampled.item()}")
        if self.reward_l2_coef is not None:
            print(f"reward last layer square l2 loss: {l2_loss.item()}")
        print(f"overall reward loss: {loss.item()}")

        # see if reward grads need clipping;
        if self.reward_grad_clip:
            nn.utils.clip_grad_norm_(
                self.reward_fn.parameters(),
                max_norm=1.0,
                error_if_nonfinite=True,
            )

        if self.verbose:
            curr_grads = []
            for p in self.reward_fn.parameters():
                this_grad = torch.norm(p.grad.detach().view(-1), 2).item()
                curr_grads.append(this_grad)
                print(
                    f"len module param: {p.shape}",
                    f"l2 norm of grad of params: " f"{this_grad}",
                )
            self.gradients.append(curr_grads)
            print("\n")

        # do grad step;
        self.reward_optim.step()

        # see if lr scheduler present;
        if self.reward_optim_lr_scheduler is not None:
            self.reward_optim_lr_scheduler.step()

    def get_avg_generated_returns(self):
        self.weight_processor = WeightsProcessor(
            self.weight_type,
            self.agent.env.spec.max_episode_steps,
        )
        if self.per_decision_imp_sample:
            return self._get_per_dec_imp_samp_returns()
        return self._get_vanilla_imp_sampled_returns()

    def _get_per_dec_imp_samp_returns(self):
        assert self.per_decision_imp_sample

        # in undirected graph edges are duplicated;
        T = self.expert_edge_index.shape[-1] // 2

        # for the step matching;
        n_steps_to_sample = (
            self.num_expert_traj + self.num_extra_paths_gen
        ) * T
        n_steps_done = 0

        # together with rewards from that episode;
        rewards = []

        # this is for getting avg sum of sq rewards over num episodes;
        quad_rewards = 0.0

        # avg lcr reularisation over episodes;
        avg_lcr_reg_term, n_episodes = 0.0, 0
        flag = True
        curr_buffer_verbose = self.agent.buffer.verbose
        curr_irl_verbose = self.verbose

        # this is for gnn multitask;
        tot_mse, tot_len = 0.0, 0

        while n_steps_done < n_steps_to_sample:
            (
                r,
                log_w,
                code,
                steps,
                lcr_reg_term,
                obs,
                _,
                _,
                temp_mse,
            ) = self.agent.buffer.get_single_ep_rewards_and_weights(
                self.agent.env,
                self.agent,
                reward_encoder=self.agent.old_encoder
                if self.do_graphopt
                else None,
            )
            assert steps == len(r)
            tot_mse = tot_mse * (
                tot_len / (tot_len + steps)
            ) + temp_mse * (steps / (tot_len + steps))
            tot_len += steps

            # update avg sum of squares;
            if self.quad_reward_penalty is not None:
                quad_rewards = quad_rewards + (
                    (r**2).sum(-1) - quad_rewards
                ) / (n_episodes + 1)

            assert len(log_w) == len(r)
            assert len(log_w) >= self.agent.env.min_steps_to_do
            assert r.requires_grad and not log_w.requires_grad
            if self.verbose:
                print(
                    f"single ep per dec sampled return: {r.detach().sum()}"
                )
                print(
                    f"single ep cumsum log weights range: {log_w.min().item()}, "
                    f"{log_w.max().item()}\n"
                )

            # update avg lcr_reg_term;
            n_episodes += 1

            # update state of weight processor;
            self.weight_processor(log_w)

            # pad all rewards to len = T with 0;
            rewards.append(F.pad(r, (0, T - len(r)), value=0))

            # increment steps;
            n_steps_done += steps

            if flag:
                # stop printing after first sampled path;
                self.agent.buffer.verbose = False
                self.verbose = False
                flag = False
        # stack rewards and get weights;
        rewards = torch.stack(rewards)
        weights, longest = self.weight_processor.get_weights()

        if self.lcr_regularisation_coef is not None:
            avg_lcr_reg_term = self._get_lcr_loss(rewards[:, :longest])

        self.verbose = curr_irl_verbose
        if self.verbose:
            # print("actual weights per dec", weights[:, :longest], sep='\n')
            print("sampled rewards shape ", rewards.shape, end="\n\n")

        # set back to old value of verbose
        self.agent.buffer.verbose = curr_buffer_verbose
        return (
            (weights[:, :longest] * rewards[:, :longest]).sum(),
            avg_lcr_reg_term,
            tot_mse,
            tot_len,
            quad_rewards,
        )

    def _get_vanilla_imp_sampled_returns(self):
        assert not self.per_decision_imp_sample
        T = self.expert_edge_index.shape[-1] // 2
        n_steps_to_sample = (
            self.num_expert_traj + self.num_extra_paths_gen
        ) * T
        n_steps_done = 0
        returns = []
        avg_lcr_reg_term, n_episodes = 0.0, 0
        flag = True
        curr_buffer_verbose = self.agent.buffer.verbose
        curr_irl_verbose = self.verbose
        quad_rewards = 0.0

        tot_mse, tot_len = 0.0, 0

        # sample steps;
        while n_steps_done < n_steps_to_sample:
            # r is a tensor of rewards over the episode now;
            # before it was return;
            (
                r,
                log_w,
                code,
                steps,
                lcr_reg_term,
                obs,
                _,
                _,
                temp_mse,
            ) = self.agent.buffer.get_single_ep_rewards_and_weights(
                self.agent.env,
                self.agent,
                reward_encoder=self.agent.old_encoder
                if self.do_graphopt
                else None,
            )

            # this is for multitask training;
            tot_mse = tot_mse * (
                tot_len / (tot_len + steps)
            ) + temp_mse * (steps / (tot_len + steps))
            tot_len += steps

            assert r.requires_grad and not log_w.requires_grad
            if self.quad_reward_penalty is not None:
                quad_rewards = quad_rewards + (
                    (r**2).sum(-1).mean() - quad_rewards
                ) / (n_episodes + 1)

            # make r undiscounted return again;
            r = r.sum()
            if self.verbose:
                print(
                    f"single ep sampled undiscounted return: {r.item()}\n"
                )

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

            if flag:
                self.agent.buffer.verbose = False
                self.verbose = False
                flag = False
        # stack returns and get weights;
        returns = torch.stack(returns)
        weights = self.weight_processor.get_weights()
        self.agent.buffer.verbose = curr_buffer_verbose
        self.verbose = curr_irl_verbose
        # if self.verbose:
        # print("actual weights vanilla", weights, sep='\n')
        return (
            returns @ weights,
            avg_lcr_reg_term,
            tot_mse,
            tot_len,
            quad_rewards,
        )

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
        flag = True
        curr_irl_verbose = self.verbose
        tot_mse, tot_len = 0.0, 0
        for _ in range(self.num_expert_traj):
            if self.do_dfs_expert_paths:
                source = np.random.randint(0, len(self.nodes))
                (
                    R,
                    rewards,
                    temp_mse,
                    temp_len,
                ) = self._get_single_ep_dfs_return(source)
            else:
                (
                    R,
                    rewards,
                    temp_mse,
                    temp_len,
                ) = self._get_single_ep_expert_return()
            if self.verbose:
                print(f"single ep expert return: {R}")
            if flag:
                self.verbose = False
                flag = False
            N += 1

            # update stuff for multitask;
            if (
                self.agent.multitask_net is not None
                and not self.do_graphopt
            ):
                tot_mse = tot_mse * (
                    tot_len / (tot_len + temp_len)
                ) + temp_mse * (temp_len / (tot_len + temp_len))
                tot_len += temp_len

            # update returns;
            avg = avg + (R - avg) / N
            cached_expert_rewards.append(rewards)
        self.verbose = curr_irl_verbose
        return avg, torch.stack(cached_expert_rewards), tot_mse, tot_len

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

        # see if should do multitask gnn training;
        mtt = self.agent.multitask_net is not None
        temp_mse, temp_len = 0.0, 0
        # the * 2 multiplier is because undirected graph is assumed
        # and effectively each edge is duplicated in edge_index;
        while pointer * self.graphs_per_batch * 2 < edge_index.shape[-1]:
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
                        edge_index=edge_index[:, idxs[:i]],
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
                            torch.tensor(
                                [[first, second], [second, first]],
                                dtype=torch.long,
                            ),
                        ),
                        -1,
                    )

            # create batch of graphs;
            batch = Batch.from_data_list(batch_list)
            extra_graph_level_feats = None
            if self.agent.buffer.transform_ is not None:
                self.agent.buffer.transform_(batch)
                if (
                    self.agent.buffer.transform_.get_graph_level_feats_fn
                    is not None
                ):
                    extra_graph_level_feats = self.agent.buffer.transform_.get_graph_level_feats_fn(
                        batch
                    )

            pointer += 1
            if self.agent.buffer.state_reward:
                if self.do_graphopt:
                    out = self.agent.old_encoder(
                        batch, extra_graph_level_feats
                    )[0]
                    curr_rewards = self.reward_fn(out)
                else:
                    curr_rewards = self.reward_fn(
                        batch,
                        extra_graph_level_feats,
                        get_graph_embeds=mtt,
                    )
            else:
                if self.do_graphopt:
                    raise NotImplementedError(
                        "graphopt only has state reward func!"
                    )
                curr_rewards = self.reward_fn(
                    (batch, torch.tensor(action_idxs)),
                    extra_graph_level_feats,
                    action_is_index=True,
                )
            # see if should do multitask;
            # graphopt doesn't have an encoder that belongs only
            # to reward_fn, so can't compute multitask loss;
            # if do graphopt is True and mtt is True, the mtt
            # loss will be only computed on the encoder of the policy;
            if mtt and not self.do_graphopt:
                targets = tgnn.global_add_pool(batch.x[:, -1], batch.batch)
                curr_rewards, graph_embeds = curr_rewards
                outs = self.agent.multitask_net(graph_embeds)
                curr_length = len(curr_rewards.view(-1))
                temp_mse = temp_mse * (
                    temp_len / (temp_len + curr_length)
                ) + nn.MSELoss()(outs.view(-1), targets.view(-1)) * (
                    curr_length / (temp_len + curr_length)
                )
                temp_len += curr_length
            curr_rewards = curr_rewards.view(-1) * self.reward_scale
            return_val += curr_rewards.sum()
            cached_rewards.append(curr_rewards)

        global DO_PLOT
        if DO_PLOT:
            G = Graph()
            G.add_edges_from(
                list(zip(*batch_list[-1].edge_index.tolist()))
            )
            fig = plt.figure()
            draw_networkx(G)
            plt.savefig("expert_example.png")
            plt.close()
            DO_PLOT = False
        return return_val, torch.cat(cached_rewards), temp_mse, temp_len

    def train_irl(self, num_iters, policy_epochs, **kwargs):
        buffer_verbose = self.agent.buffer.verbose
        for it in tqdm(range(num_iters)):
            print(f"IRL TRAINER ITER {it+1}:\n------------------------")
            # put reward fn in train mode and policy in eval mode
            # for the reward update step;
            self.agent.buffer.verbose = buffer_verbose
            self.reward_fn.requires_grad_(True)
            self.reward_fn.eval()
            self.agent.policy.requires_grad_(False)
            self.agent.policy.eval()
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
        irl_dir = self.agent.save_to / "irl_training_metrics"
        if not irl_dir.exists():
            irl_dir.mkdir(parents=True)

        metric_names = [
            "expert_avg_returns",
            "imp_sampled_returns",
            "mono_losses",
            "lcr_expert_losses",
            "lcr_sampled_losses",
            "quad_expert_losses",
            "quad_sampled_losses",
            "reward_l2_losses",
            "l2_norm_gradients",
            "mtt_losses",
        ]
        metrics = [
            self.expert_avg_returns,
            self.imp_sampled_returns,
            self.mono_losses,
            self.lcr_expert_losses,
            self.lcr_sampled_losses,
            self.quad_expert_losses,
            self.quad_sampled_losses,
            self.reward_l2_losses,
            self.gradients,
            self.mtt_losses,
        ]
        save_named_metrics(
            irl_dir,
            metric_names=metric_names,
            metrics=metrics,
        )
        metric_names[-2] = "avg_l2norm_gradients"
        grads = np.array(metrics[-2])
        metrics[-2] = grads.mean(-1)
        metric_names.append("max_l2norm_grads")
        metrics.append(grads.max(-1))
        save_metric_plots(
            metric_names,
            metrics,
            irl_dir,
            seed=self.seed,
            suptitle=f"irl training for {num_iters} iters",
        )

    def OI_init_nets(self):
        OI_init(self.reward_fn)
        self.agent.OI_init_nets()


def save_named_metrics(path, metric_names, metrics):
    assert path.exists()
    assert len(metric_names) == len(metrics)
    for n, m in zip(metric_names, metrics):
        with open(path / (n + ".pkl"), "wb") as f:
            pickle.dump(m, f)
