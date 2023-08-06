import math
import random
import numpy as np

import torch
from torch import nn

from torch_geometric.nn import global_mean_pool

from graph_irl.policy import *
from graph_irl.distributions import batch_UT_trick_from_samples
from graph_irl.buffer_v2 import *
from graph_irl.vis_utils import *
from graph_irl.sac import track_params, save_metrics

from typing import Optional
from copy import deepcopy
from pathlib import Path
import time
import pickle
from tqdm import tqdm



class SACAgentBase:
    def __init__(
        self,
        name,
        policy_constructor,
        vfunc_constructor,
        env_constructor,
        buffer_constructor,
        optimiser_constructors,
        entropy_lb,
        policy_lr,
        temperature_lr,
        vfunc_lr,
        tau,
        discount,
        save_to: Optional[Path] = TEST_OUTPUTS_PATH,
        cache_best_policy=False,
        clip_grads=False,
        **kwargs,
    ):
        self.name = name
        self.save_to = save_to
        self.clip_grads = clip_grads
        self.num_policy_updates = 0

        # experimental:
        # chache best nets;
        self.cache_best_policy = cache_best_policy
        self.best_eval_return = None
        self.best_policy = None
        self.best_V1, self.best_V2 = None, None
        self.best_V1t, self.best_V2t = None, None
        self.best_log_temp = None

        # training params;
        self.seed = kwargs["training_kwargs"]["seed"]
        self.num_iters = kwargs["training_kwargs"]["num_iters"]
        self.num_steps_to_sample = kwargs["training_kwargs"][
            "num_steps_to_sample"
        ]
        self.num_grad_steps = kwargs["training_kwargs"]["num_grad_steps"]
        self.batch_size = kwargs["training_kwargs"]["batch_size"]
        self.num_eval_steps_to_sample = kwargs["training_kwargs"][
            "num_eval_steps_to_sample"
        ]
        self.min_steps_to_presample = kwargs["training_kwargs"][
            "min_steps_to_presample"
        ]

        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)

        # instantiate necessary items;
        self.V1 = vfunc_constructor(**kwargs["V1_kwargs"])
        self.V2 = vfunc_constructor(**kwargs["V2_kwargs"])
        self.V1t = vfunc_constructor(**kwargs["V1t_kwargs"])
        self.V2t = vfunc_constructor(**kwargs["V2t_kwargs"])
        self.policy = policy_constructor(**kwargs["policy_kwargs"])

        # don't track grads for target q funcs
        # and set them in eval mode;
        self.V1t.requires_grad_(False)
        self.V2t.requires_grad_(False)
        self.V1t.eval()
        self.V2t.eval()

        # instantiate buffer and env and check some of their attributes;
        self.buffer = buffer_constructor(**kwargs["buffer_kwargs"])
        self.env = env_constructor(**kwargs["env_kwargs"])
        assert self.buffer.drop_repeats_or_self_loops == self.env.drop_repeats_or_self_loops
        assert (not self.env.calculate_reward) == self.buffer.get_batch_reward
        if self.num_eval_steps_to_sample is None:
            self.num_eval_steps_to_sample = self.env.spec.max_episode_steps
        
        # init temperature and other parameters;
        self.log_temperature = torch.tensor(0.0, requires_grad=True)
        self.entropy_lb = (
            entropy_lb
            if entropy_lb
            else -kwargs["policy_kwargs"]["action_dim"]
        )
        self.tau = tau
        self.discount = discount

        # instantiate the optimisers;
        self.policy_optim = optimiser_constructors['policy_optim'](
            self.policy.parameters(), lr=policy_lr
        )
        self.temperature_optim = optimiser_constructors['temperature_optim'](
            [self.log_temperature], lr=temperature_lr
        )
        self.V1_optim = optimiser_constructors['V1_optim'](self.V1.parameters(), lr=vfunc_lr)
        self.V2_optim = optimiser_constructors['V2_optim'](self.V2.parameters(), lr=vfunc_lr)

        # loss variables;
        self.temperature_loss = None
        self.policy_loss = None
        self.V1_loss, self.V2_loss = None, None

        # bookkeeping metrics;
        self.policy_losses = []
        self.temperature_losses = []
        self.V1_losses, self.V2_losses = [], []
        self.temperatures = [math.exp(self.log_temperature.item())]
        self.eval_path_returns, self.eval_path_lens = [], []

    def sample_action(self, obs):
        pass

    def clear_buffer(self):
        self.buffer.clear_buffer()
        # empty agent bookkeeping;
        self.policy_losses = []
        self.temperature_losses = []
        self.V1_losses, self.V2_losses = [], []
        self.temperatures = []
        self.eval_path_returns, self.eval_path_lens = [], []

    def _cache(self):
        self.best_policy = deepcopy(self.policy)
        self.best_V1 = deepcopy(self.V1)
        self.best_V2 = deepcopy(self.V2)
        self.best_V1t = deepcopy(self.V1t)
        self.best_V2t = deepcopy(self.V2t)
        self.best_log_temp = deepcopy(self.log_temperature)

    def sample_deterministic(self, obs):
        pass

    def get_policy_loss_and_temperature_loss(self, *args, **kwargs):
        pass

    def get_v_losses(self, *args, **kwargs):
        pass

    def track_vfunc_params(self):
        track_params(self.V1t, self.V1, self.tau)
        track_params(self.V2t, self.V2, self.tau)

    def update_parameters(self):
        # update policy params;
        self.policy_optim.zero_grad()
        self.policy_loss.backward()
        if self.clip_grads:
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0, error_if_nonfinite=True)
        self.policy_optim.step()

        # update temperature;
        self.temperature_optim.zero_grad()
        self.temperature_loss.backward()
        self.temperature_optim.step()

        # add new temperature to list;
        self.temperatures.append(math.exp(self.log_temperature.item()))

        # update vfunc1
        self.V1_optim.zero_grad()
        self.V1_loss.backward()
        if self.clip_grads:
            nn.utils.clip_grad_norm_(self.V1.parameters(), max_norm=1.0, error_if_nonfinite=True)
        self.V1_optim.step()

        # update vfunc2
        self.V2_optim.zero_grad()
        self.V2_loss.backward()
        if self.clip_grads:
            nn.utils.clip_grad_norm_(self.V2.parameters(), max_norm=1.0, error_if_nonfinite=True)
        self.V2_optim.step()

    def check_presample(self):
        if self.min_steps_to_presample:
            self.buffer.collect_path(
                self.env, self, self.min_steps_to_presample
            )
            self.min_steps_to_presample = 0

    def train_one_epoch(self, *args, **kwargs):
        pass

    def train_k_epochs(
        self, k, *args, config=None, vis_graph=False, **kwargs
    ):
        for _ in tqdm(range(k)):
            self.train_one_epoch(*args, **kwargs)
        
        self.num_policy_updates += 1
        # assert np.max(self.buffer.path_lens) <= self.env.spec.max_episode_steps
        # assert np.max(self.eval_path_lens) <= self.env.spec.max_episode_steps

        # check if have to save;
        if self.save_to is not None:
            metric_names = [
                "policy-loss",
                "temperature-loss",
                "vfunc1-loss",
                "vfunc2-loss",
                "avg-reward",
                "avg-reward-ma-30",
                "path-lens",
                "path-lens-ma-30",
                "returns",
                "returns-ma-30",
                "eval-path-returns",
                "eval-returns-ma-30",
                "eval-path-lens",
                "eval-path-lens-ma-30",
                "temperatures",
            ]
            metrics = [
                self.policy_losses,
                self.temperature_losses,
                self.V1_losses,
                self.V2_losses,
                self.buffer.avg_rewards_per_episode,
                get_moving_avgs(self.buffer.avg_rewards_per_episode, 30),
                self.buffer.path_lens,
                get_moving_avgs(self.buffer.path_lens, 30),
                self.buffer.undiscounted_returns,
                get_moving_avgs(self.buffer.undiscounted_returns, 30),
                self.eval_path_returns,
                get_moving_avgs(self.eval_path_returns, 30),
                self.eval_path_lens,
                get_moving_avgs(self.eval_path_lens, 30),
                self.temperatures,
            ]

            edge_index, last_eval_rewards = None, None
            if vis_graph:
                if self.best_policy is not None:
                    self.policy = self.best_policy
                    self.V1 = self.best_V1
                    self.V2 = self.best_V2
                    self.V1t = self.best_V1t
                    self.V2t = self.best_V2t
                    self.log_temperature = self.best_log_temp
                    print(f"final eval with best policy")
                
                # do final eval;
                obs, _, rewards, code = sample_eval_path_graph(
                    self.env.spec.max_episode_steps,
                    self.env,
                    self,
                    self.seed,
                    verbose=True,
                )
                print(f"code from sampling eval episode: {code}")
                edge_index = obs[-1].edge_index.tolist()
                last_eval_rewards = rewards

            save_metrics(
                self.save_to,
                metric_names,
                metrics,
                self.name,
                self.env.spec.id,
                self.seed,
                config,
                edge_index=edge_index,
                last_eval_rewards=last_eval_rewards,
                suptitle=(
                    f"figure after {self.num_policy_updates}" 
                    " policy updates"
                )
            )


class SACAgentGraph(SACAgentBase):
    def __init__(
        self,
        name,
        policy_constructor,
        vfunc_constructor,
        env_constructor,
        buffer_constructor,
        optimiser_constructors,
        entropy_lb,
        policy_lr,
        temperature_lr,
        vfunc_lr,
        tau,
        discount,
        save_to: Optional[Path] = TEST_OUTPUTS_PATH,
        cache_best_policy=False,
        clip_grads=False,
        UT_trick=False,
        with_entropy=False,
        **kwargs,
    ):
        super(SACAgentGraph, self).__init__(
            name,
            policy_constructor,
            vfunc_constructor,
            env_constructor,
            buffer_constructor,
            optimiser_constructors,
            entropy_lb,
            policy_lr,
            temperature_lr,
            vfunc_lr,
            tau,
            discount,
            save_to,
            cache_best_policy,
            clip_grads,
            **kwargs,
        )
        self.UT_trick = UT_trick
        self.with_entropy = with_entropy

        # update name;
        self.name = (
            name + f"-{self.policy.name}-UT-{int(self.UT_trick)}"
            f"-entropy-{int(self.with_entropy)}"
            f"-buffer-size-{self.buffer.max_size}"
            f"-iters-{self.num_iters}"
            f"-env-{self.env.spec.id}"
            f"-seed-{self.seed}"
            f"-timestamp-{time.time()}"
        )

        # created save dir;
        self.save_to = save_to
        if self.save_to is not None:
            self.save_to = self.save_to / self.name
            if not self.save_to.exists():
                self.save_to.mkdir(parents=True)

    def sample_action(self, obs):
        policy_dist, node_embeds = self.policy(obs)
        return policy_dist.sample(), node_embeds

    def sample_deterministic(self, obs):
        policy_dist, node_embeds = self.policy(obs)
        mus1, mus2 = policy_dist.mean
        return (mus1.detach(), mus2.detach()), node_embeds

    def get_policy_loss_and_temperature_loss(self, obs_t):
        raise NotImplementedError('implement adding edges to batch of graphs')
        # freeze q-nets and eval at current observation
        # and reparam trick action and choose min for policy loss;
        self.V1.requires_grad_(False)
        self.V2.requires_grad_(False)
        self.V1.eval()
        self.V2.eval()
        self.policy.train()

        # get policy;
        policy_density, node_embeds = self.policy(obs_t)

        # do reparam trick;
        repr_trick1, repr_trick2 = policy_density.rsample()

        # get log prob for policy optimisation;
        if self.with_entropy:
            log_prob = -policy_density.entropy().sum(-1)
        else:
            log_prob = policy_density.log_prob(
                repr_trick1, repr_trick2
            ).sum(-1)

        obs_t = global_mean_pool(
            node_embeds, obs_t.batch
        ).detach()  # pretend this was input with no grad tracking;
        vfunc_in = torch.cat((obs_t), -1)
        v_est = torch.min(
            self.V1.net(vfunc_in), self.V2.net(vfunc_in)
        ).view(-1)

        # get loss for policy;
        self.policy_loss = (
            math.exp(self.log_temperature.item()) * log_prob - v_est
        ).mean()

        # get temperature loss;
        self.temperature_loss = -(
            self.log_temperature.exp()
            * (log_prob + self.entropy_lb).detach()
        ).mean()

        # housekeeping;
        self.policy_losses.append(self.policy_loss.item())
        self.temperature_losses.append(self.temperature_loss.item())

    def get_q_losses(
        self,
        obs_t,
        action_t,
        reward_t,
        obs_tp1,
        terminated_tp1,
    ):
        self.V1.requires_grad_(True)
        self.V2.requires_grad_(True)
        self.V1.train()
        self.V2.train()

        # get predictions from q functions;
        obs_action_t = (obs_t, action_t)
        policy_density, node_embeds = self.policy(obs_tp1)
        node_embeds = node_embeds.detach()  # use embeds only as inputs;

        # the action_is_index boolean will only be considered
        # if self.encoder is not None;
        V1_est = self.V1(obs_action_t, action_is_index=True).view(-1)
        V2_est = self.V2(obs_action_t, action_is_index=True).view(-1)

        if self.UT_trick:
            # get (B, 2 * action_dim + 1, action_dim) samples;
            UT_trick_samples = policy_density.get_UT_trick_input()

            # eval expectation of q-target functions by averaging over the
            # 2 * action_dim + 1 samples and get (B, 1) output;
            qt1_est = batch_UT_trick_from_samples(
                self.V1t.net, self.V1t.encoder(obs_tp1)[0], UT_trick_samples
            )
            qt2_est = batch_UT_trick_from_samples(
                self.V2t.net, self.V2t.encoder(obs_tp1)[0], UT_trick_samples
            )

            # get negative entropy by using the UT trick;
            if self.with_entropy:
                log_probs = -policy_density.entropy().sum(-1)
            else:
                log_probs = policy_density.log_prob_UT_trick().sum(-1)
        else:
            # sample future action;
            action_tp1 = policy_density.sample()

            # get log probs;
            if self.with_entropy:
                log_probs = -policy_density.entropy().sum(-1).view(-1)
            else:
                log_probs = (
                    policy_density.log_prob(*action_tp1).sum(-1).view(-1)
                )

            # input for target nets;
            obs_tp1 = global_mean_pool(node_embeds, obs_tp1.batch)
            # action_tp1 = (a1, a2) -> tuple of action vectors;
            obs_action_tp1 = torch.cat((obs_tp1,) + action_tp1, -1)
            qt1_est, qt2_est = self.V1t.net(obs_action_tp1), self.V2t.net(
                obs_action_tp1
            )

        # use the values from the target net that
        # had lower value predictions;
        q_target = (
            torch.min(qt1_est, qt2_est).view(-1)
            - self.log_temperature.exp() * log_probs
        )

        q_target = (
            reward_t + (1 - terminated_tp1.int()) * self.discount * q_target
        ).detach()

        # loss for first q func;
        self.V1_loss = nn.MSELoss()(V1_est, q_target)

        # loss for second q func;
        self.V2_loss = nn.MSELoss()(V2_est, q_target)

        self.V1_losses.append(self.V1_loss.item())
        self.V2_losses.append(self.V2_loss.item())
    
    def train_one_epoch(self):
        # presample if needed;
        self.check_presample()

        for _ in tqdm(range(self.num_iters)):
            # sample paths;
            self.buffer.collect_path(
                self.env,
                self,
                self.num_steps_to_sample,
            )

            # sample paths with delta func policy;
            observations, actions, rewards, code = sample_eval_path_graph(
                self.num_eval_steps_to_sample,
                self.env,
                self,
                seed=self.buffer.seed - 1,
            )
            self.eval_path_returns.append(np.sum(rewards))
            self.eval_path_lens.append(len(actions))

            if self.cache_best_policy:
                if self.best_eval_return is None or self.eval_path_returns[-1] > self.best_eval_return:
                    self.best_eval_return = self.eval_path_returns[-1]
                    self._cache()

            # do the gradient updates;
            for _ in range(self.num_grad_steps):
                (
                    obs_t,
                    action_t,
                    reward_t,
                    obs_tp1,
                    terminated_tp1,
                ) = self.buffer.sample(self.batch_size)

                # get temperature and policy loss;
                self.get_policy_loss_and_temperature_loss(obs_t)

                # value func updates;
                self.get_q_losses(
                    obs_t,
                    action_t,
                    reward_t,
                    obs_tp1,
                    terminated_tp1,
                )

                # grad step on policy, temperature, V1 and V2;
                self.update_parameters()

                # target q funcs update;
                self.track_vfunc_params()