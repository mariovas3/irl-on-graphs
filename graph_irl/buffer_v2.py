"""
    TODO:
        (1): Deprecate sample_eval_paths_graph since haven't implemented it 
                for case when drop self loop or repeated edge. It is recommended
                to use get_single_ep_rewards_and_weights.
"""

import numpy as np
import torch
from torch_geometric.data import Data, Batch
import torch_geometric.nn as tgnn

from graph_irl.graph_rl_utils import (
    get_action_vector_from_idx,
    inc_lcr_reg,
)
from graph_irl.distributions import GaussDist, TanhGauss

import warnings
from typing import Callable

np.random.seed(0)


class BufferBase:
    def __init__(
        self,
        max_size,
        state_reward=False,
        seed=None,
        transform_: Callable = None,
    ):
        self.seed = 0 if seed is None else seed
        self.idx, self.max_size = 0, max_size
        self.reward_idx = 0
        self.looped = False
        self.state_reward = state_reward
        self.transform_ = transform_

        # log variables;
        self.undiscounted_returns = []
        self.path_lens = []
        self.avg_rewards_per_episode = []

    def add_sample_without_reward(self, *args, **kwargs):
        pass

    def add_rewards(self, rewards, **kwargs):
        pass

    def add_sample(self, *args, **kwargs):
        pass

    def sample(self, batch_size):
        pass

    def __len__(self):
        return self.max_size if self.looped else self.idx

    def collect_path(self, env, agent, num_steps_to_collect):
        pass

    def clear_buffer(self):
        """Reset tracked logs and idx in buffer."""
        self.idx = 0
        self.reward_idx = 0
        self.looped = False

        self.undiscounted_returns = []
        self.path_lens = []
        self.avg_rewards_per_episode = []


class GraphBuffer(BufferBase):
    def __init__(
        self,
        max_size,
        nodes: torch.Tensor,
        state_reward=False,
        seed=None,
        transform_=None,
        drop_repeats_or_self_loops=False,
        graphs_per_batch=None,
        action_is_index=True,
        action_dim=None,
        per_decision_imp_sample=False,
        reward_scale=1.0,
        log_offset=0.0,
        lcr_reg=False,
        verbose=False,
        unnorm_policy=False,
        be_deterministic=False,
        zero_interm_rew=False,
    ):
        super(GraphBuffer, self).__init__(
            max_size,
            state_reward,
            seed,
            transform_,
        )
        # I will only keep state action trajectories and eval the reward
        # as I sample a batch in the sac training loop - for more efficiency;

        self.zero_interm_rew = zero_interm_rew
        self.lcr_reg = lcr_reg
        self.verbose = verbose
        self.unnorm_policy = unnorm_policy
        self.be_deterministic = be_deterministic
        self.graphs_per_batch = graphs_per_batch

        # see what the action will be;
        self.action_is_index = action_is_index
        self.action_dim = action_dim
        if self.action_dim is None:
            assert self.action_is_index

        # place holder edge index;
        edge_index = torch.tensor([[], []], dtype=torch.long)

        # containers to store mdp info;
        self.obs_t = [
            Data(x=nodes, edge_index=edge_index)
            for _ in range(self.max_size)
        ]
        if self.action_is_index:
            self.action_t = np.empty((max_size, 2), dtype=np.int64)
        else:
            self.action_t = np.empty(
                (max_size, action_dim), dtype=np.float32
            )
        self.obs_tp1 = [
            Data(x=nodes, edge_index=edge_index)
            for _ in range(self.max_size)
        ]
        self.terminal_tp1 = np.empty((max_size,), dtype=np.int8)
        self.reward_t = None

        # keep nodes reference;
        self.nodes = nodes

        # log_offset is subtracted in
        # reward - log_prob_policy - log_offset
        # the reward is always in (-inf, 0)
        # as the reward and policy both become better, the
        # reward will give high reward for sampled traj
        # so reward -> 0, however log_prob_policy is generally
        # negative and depends on dim of action vector;
        # at that point our importance weights (reward - log_prob).exp()
        # might become too large so an offset might help;
        self.log_offset = log_offset

        # similar purpose as log offset, although has slightly different
        # interpretation as amplifying reward signal;
        self.reward_scale = reward_scale

        # extra config for reward computation;
        self.drop_repeats_or_self_loops = drop_repeats_or_self_loops
        self.per_decision_imp_sample = per_decision_imp_sample

    def clear_buffer(self):
        super().clear_buffer()
        edge_index = torch.tensor([[], []], dtype=torch.long)

        # reset buffer;
        self.obs_t = [
            Data(x=self.nodes, edge_index=edge_index)
            for _ in range(self.max_size)
        ]
        if self.action_is_index:
            self.action_t = np.empty((self.max_size, 2), dtype=np.int64)
        else:
            self.action_t = np.empty(
                (self.max_size, self.action_dim), dtype=np.float32
            )
        self.obs_tp1 = [
            Data(x=self.nodes, edge_index=edge_index)
            for _ in range(self.max_size)
        ]
        self.terminal_tp1 = np.empty((self.max_size,), dtype=np.int8)

    def add_sample(self, obs_t, action_t, reward_t, obs_tp1, terminal_tp1):
        idx = self.idx % self.max_size
        if not self.looped and self.idx and not idx:
            self.looped = True
        self.idx = idx
        self.obs_t[idx] = obs_t
        self.action_t[idx] = action_t
        self.reward_t[idx] = reward_t
        self.obs_tp1[idx] = obs_tp1
        self.terminal_tp1[idx] = terminal_tp1
        self.idx += 1

    def add_sample_without_reward(
        self, obs_t, action_t, obs_tp1, terminal_tp1
    ):
        idx = self.idx % self.max_size
        if not self.looped and self.idx and not idx:
            self.looped = True
        self.idx = idx
        self.obs_t[idx] = obs_t
        self.action_t[idx] = action_t
        self.obs_tp1[idx] = obs_tp1
        self.terminal_tp1[idx] = terminal_tp1
        self.idx += 1

    def add_rewards(self, rewards):
        """
        Add 1-D numpy array of rewards to the buffer.
        """
        reward_idx = self.reward_idx % self.max_size
        idxs = (reward_idx + np.arange(len(rewards))) % self.max_size
        self.reward_t[idxs] = rewards
        self.reward_idx = idxs[-1] + 1

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.graphs_per_batch
        assert batch_size <= self.__len__()

        # always sample without replacement;
        idxs = np.random.choice(
            self.__len__(), size=batch_size, replace=False
        )

        # get batch of graphs;
        batch = Batch.from_data_list([self.obs_t[idx] for idx in idxs])
        batch_tp1 = Batch.from_data_list(
            [self.obs_tp1[idx] for idx in idxs]
        )

        # see if should process inplace;
        # it's either this compute overhead
        # or it will be a memory overhead that
        # scales as O(buffer_max_len * num_nodes_of_graph * new_columns)
        if self.transform_ is not None:
            self.transform_(batch)
            self.transform_(batch_tp1)

        # return batch, actions, no_reward, future_batch, terminated
        return (
            batch,
            torch.from_numpy(self.action_t[idxs]),
            None,
            batch_tp1,
            torch.tensor(self.terminal_tp1[idxs], dtype=torch.float32),
        )

    def _process_rewards_log_weights(self, rewards, log_weights, code):
        rewards = torch.cat(rewards, -1)
        log_weights = torch.cat(log_weights, -1)
        return (rewards, torch.cumsum(log_weights, 0), code, len(rewards))

    def get_reward(
        self,
        batch_list,
        actions,
        reward_fn,
        extra_graph_level_feats_list=None,
        get_graph_embeds=False,
        reward_encoder=None,
    ):
        if self.state_reward:
            assert len(batch_list) == len(actions) + 1
            batch = Batch.from_data_list(batch_list[1:])
            if extra_graph_level_feats_list:
                extra_graph_level_feats = torch.cat(
                    extra_graph_level_feats_list[1:], 0
                )
            else:
                extra_graph_level_feats = None
            # return (rewards, [graph_embeds]), targets
            if reward_encoder is not None:
                out = reward_encoder(batch, extra_graph_level_feats)[0]
                return reward_fn(out)
            temp = reward_fn(
                batch,
                extra_graph_level_feats=extra_graph_level_feats,
                get_graph_embeds=get_graph_embeds,
            )
            if get_graph_embeds:
                return temp, tgnn.global_add_pool(
                    batch.x[:, -1], batch.batch
                )
            return temp
        batch = Batch.from_data_list(batch_list)
        if extra_graph_level_feats_list:
            extra_graph_level_feats = torch.cat(
                extra_graph_level_feats_list, 0
            )
        else:
            extra_graph_level_feats = None
        assert len(batch_list) == len(actions)
        # return (rewards, [graph_embeds]), targets
        temp = reward_fn(
            (batch, actions),
            extra_graph_level_feats,
            action_is_index=self.action_is_index,
            get_graph_embeds=get_graph_embeds,
        )
        if get_graph_embeds:
            return temp, tgnn.global_add_pool(batch.x[:, -1], batch.batch)
        return temp

    def get_log_probs_and_dists(
        self,
        batch_list,
        actions,
        agent,
        extra_graph_level_feats_list=None,
    ):
        if self.state_reward:
            assert len(batch_list) == len(actions) + 1
            batch = Batch.from_data_list(batch_list[:-1])
            if extra_graph_level_feats_list:
                assert len(extra_graph_level_feats_list) == len(batch_list)
                extra_graph_level_feats = torch.cat(
                    extra_graph_level_feats_list[:-1], 0
                )
            else:
                extra_graph_level_feats = None
        else:
            assert len(batch_list) == len(actions)
            batch = Batch.from_data_list(batch_list)
            if extra_graph_level_feats_list:
                extra_graph_level_feats = torch.cat(
                    extra_graph_level_feats_list, 0
                )
            else:
                extra_graph_level_feats = None

        policy_dists, node_embeds = agent.policy(
            batch, extra_graph_level_feats
        )

        # get action vectors from indexes by taking the node embeds
        # for the relevant indexes;
        if self.action_is_index:
            actions = get_action_vector_from_idx(
                node_embeds, actions, batch.num_graphs
            )

        # get node embedding;
        _, D = node_embeds.shape
        assert actions.shape[-1] == 2 * D

        # get log_probs;
        if self.unnorm_policy:
            log_probs = policy_dists.get_unnorm_log_prob(
                actions[:, :D], actions[:, D:]
            ).sum(-1)
        else:
            log_probs = policy_dists.log_prob(
                actions[:, :D], actions[:, D:]
            ).sum(-1)
        assert len(log_probs) == batch.num_graphs
        return log_probs, policy_dists

    def get_single_ep_rewards_and_weights(
        self,
        env,
        agent,
        with_mrr=False,
        with_auc=False,
        positives_dict=None,
        k_proposals: int = 1,
        reward_encoder=None,
    ):
        temp_mse, temp_len = 0.0, 0
        flag = agent.multitask_net is not None
        rrs_this_episode = []  # reciprocal ranks;
        reward_on_path = []
        if with_mrr or with_auc:
            assert positives_dict is not None

        # init lcr regularisation term and last two rewards;
        lcr_reg_term = 0.0
        r1, r2 = None, None
        rewards = []

        if self.per_decision_imp_sample:
            log_weights = []
        else:
            log_imp_weight, return_val = 0, 0

        # start the episode;
        obs, info = env.reset(seed=self.seed)
        extra_graph_level_feats = None
        if self.transform_ is not None:
            self.transform_(obs)
            if self.transform_.get_graph_level_feats_fn is not None:
                extra_graph_level_feats = (
                    self.transform_.get_graph_level_feats_fn(obs)
                )

        # eval policy and reward fn in batch mode;
        action_idxs = []
        if self.state_reward:
            # save first obs to get log prob later;
            # this will be excluded from reward_fn call;
            batch_list = [obs]
            if extra_graph_level_feats is not None:
                extra_graph_level_feats_list = [extra_graph_level_feats]
            else:
                extra_graph_level_feats_list = []
        else:
            batch_list = []
            extra_graph_level_feats_list = []

        if self.verbose:
            env.reward_fn.verbose()

        terminated, truncated = False, False

        # this steps variable is for debugging
        steps = 0

        # process the episode;
        while not (terminated or truncated):
            # sample actions;
            if self.be_deterministic:
                (a1, a2), node_embeds = agent.sample_deterministic(
                    obs, extra_graph_level_feats
                )
            else:
                (a1, a2), node_embeds = agent.sample_action(
                    obs, k_proposals, extra_graph_level_feats
                )

            # make env step;
            new_obs, reward, terminated, truncated, info = env.step(
                ((a1.numpy(), a2.numpy()), node_embeds.detach().numpy()),
                positives_dict,
            )

            # resample action if repeated edge or self loop;
            if self.drop_repeats_or_self_loops and not (
                terminated or truncated
            ):
                if info["self_loop"] or info["is_repeat"]:
                    continue

            skip_sample = (
                self.drop_repeats_or_self_loops
                and (
                    info["max_repeats_reached"]
                    or info["max_self_loops_reached"]
                )
                # also check if current is self loop or repeated edge;
                # since if max repeats or max self loops are true
                # but min_steps_to_do is not reached in env
                # we may discard good edges;
                and (info["self_loop"] or info["is_repeat"])
            )

            steps -= int(skip_sample)

            # get idxs of nodes to connect;
            first, second = info["first"], info["second"]
            which_action = info["which_action"]
            recip_rank = info["recip_rank"]

            # add action to action list and observation in batch_list;
            if not skip_sample:
                if which_action is not None:
                    assert len(a1.shape) == 2
                    a1 = a1[which_action, :].view(-1)
                    a2 = a2[which_action, :].view(-1)

                    # this is where you track recip rank
                    if with_mrr:
                        rrs_this_episode.append(recip_rank)

                    # and the entry of positives_dict is turned on
                    if with_auc:
                        e1, e2 = env._get_edge_hash(first, second)
                        # see if a positive was discovered;
                        if (e1, e2) in positives_dict:
                            positives_dict[(e1, e2)] = 1
                if self.transform_ is not None:
                    self.transform_(new_obs)
                    if (
                        self.transform_.get_graph_level_feats_fn
                        is not None
                    ):
                        extra_graph_level_feats_tp1 = (
                            self.transform_.get_graph_level_feats_fn(
                                new_obs
                            )
                        )

                if self.action_is_index:
                    action_idxs.append(
                        torch.tensor([first, second], dtype=torch.long)
                    )
                else:
                    assert a1.ndim == 1
                    action_idxs.append(torch.cat((a1, a2), -1))

                if not self.state_reward:
                    batch_list.append(obs)
                    if extra_graph_level_feats is not None:
                        extra_graph_level_feats_list.append(
                            extra_graph_level_feats
                        )
                else:
                    batch_list.append(new_obs)
                    if extra_graph_level_feats is not None:
                        extra_graph_level_feats_list.append(
                            extra_graph_level_feats_tp1
                        )

            # if max batch length reached, compute rewards;
            sr = int(self.state_reward)
            if len(batch_list) - sr and (
                len(batch_list) == self.graphs_per_batch + sr
                or terminated
                or truncated
            ):
                # make batch of graphs;
                action_idxs = torch.stack(action_idxs)

                # calculate rewards on batch;
                curr_rewards = self.get_reward(
                    batch_list,
                    action_idxs,
                    env.reward_fn,
                    extra_graph_level_feats_list,
                    get_graph_embeds=flag,
                    reward_encoder=reward_encoder,
                )

                # this is for multitask gnn training;
                if flag and reward_encoder is None:
                    (curr_rewards, graph_embeds), targets = curr_rewards
                    outs = agent.multitask_net(graph_embeds)
                    curr_length = len(curr_rewards.view(-1))
                    temp_mse = temp_mse * (
                        temp_len / (temp_len + curr_length)
                    ) + torch.nn.MSELoss()(
                        outs.view(-1), targets.view(-1)
                    ) * (
                        curr_length / (temp_len + curr_length)
                    )
                    temp_len += curr_length
                curr_rewards = curr_rewards.view(-1) * self.reward_scale
                if self.zero_interm_rew:
                    last_rew = curr_rewards[-1]
                    curr_rewards = torch.zeros_like(curr_rewards)
                    if terminated or truncated:
                        curr_rewards[-1] = last_rew

                # track rewards on path;
                reward_on_path.extend(curr_rewards.detach().tolist())

                # see if lcr_reg should be calculated;
                # a single lcr reg term requires rewards from 3 time steps;
                if self.lcr_reg:
                    r1, r2, inc = inc_lcr_reg(r1, r2, curr_rewards)
                    lcr_reg_term += inc

                # get log probs and policy dists;
                log_probs, policy_dists = self.get_log_probs_and_dists(
                    batch_list,
                    action_idxs,
                    agent,
                    extra_graph_level_feats_list,
                )

                # see if should print stuff;
                if self.verbose:
                    print(
                        "\nreward range within buffer sampling: ",
                        curr_rewards.min().item(),
                        curr_rewards.max().item(),
                        sep=" ",
                    )
                    i = log_probs.argmin().item()
                    j = log_probs.argmax().item()
                    print(
                        f"log_prob range: ",
                        log_probs[i].item(),
                        log_probs[j].item(),
                        sep=" ",
                    )
                    if not isinstance(policy_dists, TanhGauss):
                        print(
                            f"corresponding entropies: ",
                            policy_dists.entropy()
                            .sum(-1)[[i, j]]
                            .detach()
                            .numpy(),
                        )

                # rewards and log_weights
                rewards.append(curr_rewards)
                if self.per_decision_imp_sample:
                    log_weights.append(
                        (
                            curr_rewards - log_probs - self.log_offset
                        ).detach()
                    )
                else:
                    log_imp_weight += (
                        (curr_rewards - log_probs - self.log_offset)
                        .sum()
                        .detach()
                    )
                    # return_val += curr_rewards.sum()
                    # if self.verbose:
                    # print(f"step: {steps} of sampling, vanilla sum of log weights: {log_imp_weight}",
                    # f"vanilla return: {return_val}\n", end='\n\n')

                # reset batch list and action idxs list;
                if self.state_reward:
                    batch_list = [batch_list[-1]]
                    if extra_graph_level_feats_list:
                        extra_graph_level_feats_list = [
                            extra_graph_level_feats_list[-1]
                        ]
                else:
                    batch_list = []
                    extra_graph_level_feats_list = []
                action_idxs = []

            # update current state;
            obs = new_obs
            steps += 1
            code = -1
            if terminated and not truncated:
                code = 0
            elif terminated and truncated:
                code = 1
            elif truncated:
                code = 2
            if code != -1:
                assert steps == env.steps_done - env.num_edges_start_from
                if self.verbose:
                    print(f"steps done: {steps}, code: {code}")
                if self.per_decision_imp_sample:
                    return self._process_rewards_log_weights(
                        rewards, log_weights, code
                    ) + (
                        lcr_reg_term,
                        obs,
                        rrs_this_episode,
                        reward_on_path,
                        temp_mse,
                    )
                return (
                    torch.cat(rewards, -1),
                    log_imp_weight,
                    code,
                    env.steps_done,
                    lcr_reg_term,
                    obs,
                    rrs_this_episode,
                    reward_on_path,
                    temp_mse,
                )

        assert steps == env.steps_done - env.num_edges_start_from
        if self.verbose:
            print(f"steps done: {steps}, code: {code}")
        if self.per_decision_imp_sample:
            # in the per dicision case, need to cumsum the log_weights
            # until the current time point;
            return self._process_rewards_log_weights(
                rewards, log_weights, 2
            ) + (
                lcr_reg_term,
                obs,
                rrs_this_episode,
                reward_on_path,
                temp_mse,
            )
        return (
            torch.cat(rewards, -1),
            log_imp_weight,
            2,
            env.steps_done,
            lcr_reg_term,
            obs,
            rrs_this_episode,
            reward_on_path,
            temp_mse,
        )

    def collect_path(
        self,
        env,
        agent,
        num_steps_to_collect,
    ):
        """
        Collect observations and actions from MDP.

        Args:
            env: Supports similar api to gymnasium.Env..
            agent: Supports sample_action(obs) api.
            num_steps_to_collect: Number of (obs, action, reward, next_obs, terminated)
                tuples to be added to the buffer.

        Note:
            This will collect only (obs, action, obs_tp1, terminal)
            tuples as steps. The rewards will be computed within
            the SAC training loop, so that we always eval the
            current reward and don't have to discard old experience
            with old rewards.
        """
        num_steps_to_collect = min(num_steps_to_collect, self.max_size)
        t = 0
        obs_t, info = env.reset(seed=self.seed)
        extra_graph_level_feats = None
        self.seed += 1
        k_proposals = 1  # this is for compatibility

        # get at least num_steps_to_collect steps
        # and exit when terminated or truncated;
        while t < num_steps_to_collect or not (terminated or truncated):
            # see if should preprocess;
            if self.transform_ is not None:
                self.transform_(obs_t)
                if self.transform_.get_graph_level_feats_fn is not None:
                    extra_graph_level_feats = (
                        self.transform_.get_graph_level_feats_fn(obs_t)
                    )

            # sample action;
            (a1, a2), node_embeds = agent.sample_action(
                obs_t, k_proposals, extra_graph_level_feats
            )
            a1, a2 = a1.numpy(), a2.numpy()

            # sample dynamics;
            obs_tp1, reward, terminated, truncated, info = env.step(
                ((a1, a2), node_embeds.detach().numpy())
            )

            # resample action if repeated edge or self loop;
            if self.drop_repeats_or_self_loops and not (
                terminated or truncated
            ):
                if info["self_loop"] or info["is_repeat"]:
                    continue

            skip_sample = (
                self.drop_repeats_or_self_loops
                and (
                    info["max_repeats_reached"]
                    or info["max_self_loops_reached"]
                )
                and (info["self_loop"] or info["is_repeat"])
            )

            # this is if obs_t was set to point to memory of obs_tp1
            # and was then modified inplace but not added to buffer
            # and consequently not with original node features;
            obs_t.x = self.nodes

            # see if this sample should be skipped;
            t -= int(skip_sample)

            if not skip_sample:
                # indexes of nodes to be connected;
                first, second = info["first"], info["second"]

                if self.action_is_index:
                    action_t = np.array([first, second])
                else:
                    assert a1.ndim == 1
                    action_t = np.concatenate((a1, a2), -1)

                # reward will be computed during training of SAC;
                self.add_sample_without_reward(
                    obs_t, action_t, obs_tp1, terminated
                )

            # restart env if episode ended;
            if terminated or truncated:
                # print("repeats or self loops: "
                #       f"{info['max_repeats_reached']}, "
                #       f"{info['max_self_loops_reached']}")
                obs_t, info = env.reset(seed=self.seed)
                self.seed += 1
            else:
                obs_t = obs_tp1
            t += 1


class Buffer(BufferBase):
    def __init__(self, max_size, obs_dim, action_dim, seed=None):
        super(Buffer, self).__init__(max_size, seed)
        self.obs_t = np.empty((max_size, obs_dim))
        self.action_t = np.empty((max_size, action_dim))
        self.obs_tp1 = np.empty((max_size, obs_dim))
        self.terminal_tp1 = np.empty((max_size,))
        self.reward_t = np.empty((max_size,))
        self.obs_dim, self.action_dim = obs_dim, action_dim

    def clear_buffer(self):
        super().clear_buffer()
        self.obs_t = np.empty((self.max_size, self.obs_dim))
        self.action_t = np.empty((self.max_size, self.action_dim))
        self.obs_tp1 = np.empty((self.max_size, self.obs_dim))
        self.terminal_tp1 = np.empty((self.max_size,))
        self.reward_t = np.empty((self.max_size,))

    def add_sample(self, obs_t, action_t, reward_t, obs_tp1, terminal_tp1):
        idx = self.idx % self.max_size
        if not self.looped and self.idx and not idx:
            self.looped = True
        self.idx = idx
        self.obs_t[idx] = obs_t
        self.action_t[idx] = action_t
        self.reward_t[idx] = reward_t
        self.obs_tp1[idx] = obs_tp1
        self.terminal_tp1[idx] = terminal_tp1
        self.idx += 1

    def sample(self, batch_size):
        assert batch_size <= self.__len__()
        idxs = np.random.choice(
            self.__len__(), size=batch_size, replace=False
        )
        return (
            torch.tensor(self.obs_t[idxs], dtype=torch.float32),
            torch.tensor(self.action_t[idxs], dtype=torch.float32),
            torch.tensor(self.reward_t[idxs], dtype=torch.float32),
            torch.tensor(self.obs_tp1[idxs], dtype=torch.float32),
            torch.tensor(self.terminal_tp1[idxs], dtype=torch.float32),
        )

    def collect_path(
        self,
        env,
        agent,
        num_steps_to_collect,
    ):
        """
        Collect steps from MDP induced by env and agent.

        Args:
            env: Supports similar api to gymnasium.Env..
            agent: Supports sample_action(obs) api.
            num_steps_to_collect: Number of (obs, action, reward, next_obs, terminated)
                tuples to be added to the buffer.
        """
        num_steps_to_collect = min(num_steps_to_collect, self.max_size)
        t = 0
        obs_t, info = env.reset(seed=self.seed)
        self.seed += 1
        avg_reward, num_rewards = 0.0, 0.0
        undiscounted_return = 0.0
        while t < num_steps_to_collect:
            obs_t = torch.tensor(obs_t, dtype=torch.float32)

            # sample action;
            action_t = agent.sample_action(obs_t).numpy()

            # sample dynamics;
            obs_tp1, reward, terminal, truncated, info = env.step(action_t)

            # add sampled tuple to buffer;
            self.add_sample(obs_t, action_t, reward, obs_tp1, terminal)

            # house keeping for observed rewards.
            num_rewards += 1

            avg_reward = avg_reward + (reward - avg_reward) / num_rewards
            undiscounted_return += reward

            # restart env if episode ended;
            if terminal or truncated:
                obs_t, info = env.reset(seed=self.seed)
                self.seed += 1
                self.undiscounted_returns.append(undiscounted_return)
                self.avg_rewards_per_episode.append(avg_reward)
                self.path_lens.append(num_rewards)
                avg_reward = 0.0
                num_rewards = 0.0
                undiscounted_return = 0.0
            else:
                obs_t = obs_tp1
            t += 1


def sample_eval_path(T, env, agent, seed):
    agent.policy.eval()
    observations, actions, rewards = [], [], []
    obs, info = env.reset(seed=seed)
    observations.append(obs)
    for _ in range(T):
        obs = torch.tensor(obs, dtype=torch.float32)
        action = agent.sample_deterministic(obs).numpy()
        new_obs, reward, terminated, truncated, info = env.step(action)
        actions.append(action)
        observations.append(new_obs)
        rewards.append(reward)
        obs = new_obs
        if terminated and not truncated:
            return observations, actions, rewards, 0
        if terminated and truncated:
            return observations, actions, rewards, 1
        if truncated:
            return observations, actions, rewards, 2
    return observations, actions, rewards, 2


if __name__ == "__main__":
    import gymnasium as gym

    class DummyAgent:
        def __init__(self, env):
            self.env = env

        def sample_action(self, obs_t):
            return torch.tensor(
                self.env.action_space.sample(), dtype=torch.float32
            )

    env = gym.make("Hopper-v2", max_episode_steps=300)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_size = 1_000
    batch_size = 100
    buffer = Buffer(max_size, obs_dim, action_dim)
    returns = []
    agent = DummyAgent(env)
    buffer.collect_path(env, agent, 1200)
    print(len(buffer))
    for _ in range(5):
        batch = buffer.sample(batch_size)
