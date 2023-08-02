"""
TODO:
        (1): Implement the UT trick for the graph case.
                - currently works for GaussPolicy;
"""

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

from typing import Optional
from copy import deepcopy
from pathlib import Path
import time
import pickle
from tqdm import tqdm

# path to save logs;
TEST_OUTPUTS_PATH = Path(__file__).absolute().parent.parent / "test_output"


def track_params(Q1t, Q1, tau):
    """
    Updates parameters of Q1t to be:
    Q1t = Q1 * tau + Q1t * (1 - tau).
    """
    theta = nn.utils.parameters_to_vector(Q1.parameters()) * tau + (
        1 - tau
    ) * nn.utils.parameters_to_vector(Q1t.parameters())

    # load theta as the new params of Q1;
    nn.utils.vector_to_parameters(theta, Q1t.parameters())


class SACAgentBase:
    def __init__(
        self,
        name,
        policy_constructor,
        qfunc_constructor,
        env_constructor,
        buffer_constructor,
        optimiser_constructors,
        entropy_lb,
        policy_lr,
        temperature_lr,
        qfunc_lr,
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
        self.best_Q1, self.best_Q2 = None, None
        self.best_Q1t, self.best_Q2t = None, None
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
        self.Q1 = qfunc_constructor(**kwargs["Q1_kwargs"])
        self.Q2 = qfunc_constructor(**kwargs["Q2_kwargs"])
        self.Q1t = qfunc_constructor(**kwargs["Q1t_kwargs"])
        self.Q2t = qfunc_constructor(**kwargs["Q2t_kwargs"])
        self.policy = policy_constructor(**kwargs["policy_kwargs"])

        # don't track grads for target q funcs
        # and set them in eval mode;
        self.Q1t.requires_grad_(False)
        self.Q2t.requires_grad_(False)
        self.Q1t.eval()
        self.Q2t.eval()

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
        self.Q1_optim = optimiser_constructors['Q1_optim'](self.Q1.parameters(), lr=qfunc_lr)
        self.Q2_optim = optimiser_constructors['Q2_optim'](self.Q2.parameters(), lr=qfunc_lr)

        # loss variables;
        self.temperature_loss = None
        self.policy_loss = None
        self.Q1_loss, self.Q2_loss = None, None

        # bookkeeping metrics;
        self.policy_losses = []
        self.temperature_losses = []
        self.Q1_losses, self.Q2_losses = [], []
        self.temperatures = [math.exp(self.log_temperature.item())]
        self.eval_path_returns, self.eval_path_lens = [], []

    def sample_action(self, obs):
        pass

    def clear_buffer(self):
        self.buffer.clear_buffer()
        # empty agent bookkeeping;
        self.policy_losses = []
        self.temperature_losses = []
        self.Q1_losses, self.Q2_losses = [], []
        self.temperatures = []
        self.eval_path_returns, self.eval_path_lens = [], []

    def _cache(self):
        self.best_policy = deepcopy(self.policy)
        self.best_Q1 = deepcopy(self.Q1)
        self.best_Q2 = deepcopy(self.Q2)
        self.best_Q1t = deepcopy(self.Q1t)
        self.best_Q2t = deepcopy(self.Q2t)
        self.best_log_temp = deepcopy(self.log_temperature)

    def sample_deterministic(self, obs):
        pass

    def get_policy_loss_and_temperature_loss(self, *args, **kwargs):
        pass

    def get_q_losses(self, *args, **kwargs):
        pass

    def track_qfunc_params(self):
        track_params(self.Q1t, self.Q1, self.tau)
        track_params(self.Q2t, self.Q2, self.tau)

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

        # update qfunc1
        self.Q1_optim.zero_grad()
        self.Q1_loss.backward()
        if self.clip_grads:
            nn.utils.clip_grad_norm_(self.Q1.parameters(), max_norm=1.0, error_if_nonfinite=True)
        self.Q1_optim.step()

        # update qfunc2
        self.Q2_optim.zero_grad()
        self.Q2_loss.backward()
        if self.clip_grads:
            nn.utils.clip_grad_norm_(self.Q2.parameters(), max_norm=1.0, error_if_nonfinite=True)
        self.Q2_optim.step()

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
                "qfunc1-loss",
                "qfunc2-loss",
                "avg-reward",
                "avg-reward-ma-30",
                "path-lens",
                "path-lens-ma-30",
                "undiscounted-returns",
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
                self.Q1_losses,
                self.Q2_losses,
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
                    self.Q1 = self.best_Q1
                    self.Q2 = self.best_Q2
                    self.Q1t = self.best_Q1t
                    self.Q2t = self.best_Q2t
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
        qfunc_constructor,
        env_constructor,
        buffer_constructor,
        optimiser_constructors,
        entropy_lb,
        policy_lr,
        temperature_lr,
        qfunc_lr,
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
            qfunc_constructor,
            env_constructor,
            buffer_constructor,
            optimiser_constructors,
            entropy_lb,
            policy_lr,
            temperature_lr,
            qfunc_lr,
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
        # freeze q-nets and eval at current observation
        # and reparam trick action and choose min for policy loss;
        self.Q1.requires_grad_(False)
        self.Q2.requires_grad_(False)
        self.Q1.eval()
        self.Q2.eval()
        self.policy.train()

        # get policy;
        policy_density, node_embeds = self.policy(obs_t)

        if self.UT_trick:
            # currently works for gaussian policy;
            if self.with_entropy:
                log_pi_integral = -policy_density.entropy().sum(-1)
            else:
                log_pi_integral = policy_density.log_prob_UT_trick().sum(-1)
            UT_trick_samples = policy_density.get_UT_trick_input()
            
            # encoder(obs_t)[0] gives one vector per graph in batch;
            q1_integral = batch_UT_trick_from_samples(
                self.Q1.net, self.Q1.encoder(obs_t)[0], UT_trick_samples
            )
            q2_integral = batch_UT_trick_from_samples(
                self.Q2.net, self.Q2.encoder(obs_t)[0], UT_trick_samples
            )
            q_integral = torch.min(q1_integral, q2_integral).view(-1)

            # get policy_loss;
            self.policy_loss = (
                np.exp(self.log_temperature.item()) * log_pi_integral
                - q_integral
            ).mean()

            # get temperature loss;
            self.temperature_loss = -(
                self.log_temperature.exp()
                * (log_pi_integral + self.entropy_lb).detach()
            ).mean()
        else:
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
            qfunc_in = torch.cat((obs_t, repr_trick1, repr_trick2), -1)
            q_est = torch.min(
                self.Q1.net(qfunc_in), self.Q2.net(qfunc_in)
            ).view(-1)

            # get loss for policy;
            self.policy_loss = (
                math.exp(self.log_temperature.item()) * log_prob - q_est
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
        self.Q1.requires_grad_(True)
        self.Q2.requires_grad_(True)
        self.Q1.train()
        self.Q2.train()

        # get predictions from q functions;
        obs_action_t = (obs_t, action_t)
        policy_density, node_embeds = self.policy(obs_tp1)
        node_embeds = node_embeds.detach()  # use embeds only as inputs;

        # the action_is_index boolean will only be considered
        # if self.encoder is not None;
        q1_est = self.Q1(obs_action_t, action_is_index=True).view(-1)
        q2_est = self.Q2(obs_action_t, action_is_index=True).view(-1)

        if self.UT_trick:
            # get (B, 2 * action_dim + 1, action_dim) samples;
            UT_trick_samples = policy_density.get_UT_trick_input()

            # eval expectation of q-target functions by averaging over the
            # 2 * action_dim + 1 samples and get (B, 1) output;
            qt1_est = batch_UT_trick_from_samples(
                self.Q1t.net, self.Q1t.encoder(obs_tp1)[0], UT_trick_samples
            )
            qt2_est = batch_UT_trick_from_samples(
                self.Q2t.net, self.Q2t.encoder(obs_tp1)[0], UT_trick_samples
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
            qt1_est, qt2_est = self.Q1t.net(obs_action_tp1), self.Q2t.net(
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
        self.Q1_loss = nn.MSELoss()(q1_est, q_target)

        # loss for second q func;
        self.Q2_loss = nn.MSELoss()(q2_est, q_target)

        self.Q1_losses.append(self.Q1_loss.item())
        self.Q2_losses.append(self.Q2_loss.item())
    
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

                # grad step on policy, temperature, Q1 and Q2;
                self.update_parameters()

                # target q funcs update;
                self.track_qfunc_params()


class SACAgentMuJoCo(SACAgentBase):
    def __init__(
        self,
        name,
        policy_constructor,
        qfunc_constructor,
        env_constructor,
        buffer_constructor,
        optimiser_constructors,
        entropy_lb,
        policy_lr,
        temperature_lr,
        qfunc_lr,
        tau,
        discount,
        save_to: Optional[Path] = TEST_OUTPUTS_PATH,
        cache_best_policy=False,
        clip_grads=False,
        UT_trick=False,
        with_entropy=False,
        **kwargs,
    ):
        super(SACAgentMuJoCo, self).__init__(
            name,
            policy_constructor,
            qfunc_constructor,
            env_constructor,
            buffer_constructor,
            optimiser_constructors,
            entropy_lb,
            policy_lr,
            temperature_lr,
            qfunc_lr,
            tau,
            discount,
            save_to,
            cache_best_policy,
            clip_grads,
            **kwargs,
        )
        self.UT_trick = UT_trick
        self.with_entropy = with_entropy

    def sample_action(self, obs):
        policy_density = self.policy(obs)
        action = policy_density.sample()
        return action

    def sample_deterministic(self, obs):
        policy_density = self.policy(obs)
        return policy_density.mean.detach()

    def get_policy_loss_and_temperature_loss(self, obs_t):
        # freeze q-nets and eval at current observation
        # and reparam trick action and choose min for policy loss;
        self.Q1.requires_grad_(False)
        self.Q2.requires_grad_(False)
        self.Q1.eval()
        self.Q2.eval()
        self.policy.train()

        # get policy;
        policy_density = self.policy(obs_t)

        if self.UT_trick:
            if self.with_entropy:
                log_pi_integral = -policy_density.entropy().sum(-1)
            else:
                log_pi_integral = policy_density.log_prob_UT_trick().sum(-1)
            UT_trick_samples = policy_density.get_UT_trick_input()
            q1_integral = batch_UT_trick_from_samples(
                self.Q1, obs_t, UT_trick_samples
            )
            q2_integral = batch_UT_trick_from_samples(
                self.Q2, obs_t, UT_trick_samples
            )
            q_integral = torch.min(q1_integral, q2_integral).view(-1)

            # get policy_loss;
            self.policy_loss = (
                math.exp(self.log_temperature.item()) * log_pi_integral
                - q_integral
            ).mean()

            # get temperature loss;
            self.temperature_loss = -(
                self.log_temperature.exp()
                * (log_pi_integral + self.entropy_lb).detach()
            ).mean()
        else:
            # do reparam trick;
            repr_trick = policy_density.rsample()

            # get log prob for policy optimisation;
            if self.with_entropy:
                log_prob = -policy_density.entropy().sum(-1)
            else:
                log_prob = policy_density.log_prob(repr_trick).sum(-1)

            qfunc_in = torch.cat((obs_t, repr_trick), -1)
            q_est = torch.min(self.Q1(qfunc_in), self.Q2(qfunc_in)).view(-1)

            # get loss for policy;
            self.policy_loss = (
                np.exp(self.log_temperature.item()) * log_prob - q_est
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
        self.Q1.requires_grad_(True)
        self.Q2.requires_grad_(True)
        self.Q1.train()
        self.Q2.train()

        # get predictions from q functions;
        obs_action_t = torch.cat((obs_t, action_t), -1)
        policy_density = self.policy(obs_tp1)

        # the action_is_index boolean will only be considered
        # if self.encoder is not None;
        q1_est = self.Q1(obs_action_t, action_is_index=True).view(-1)
        q2_est = self.Q2(obs_action_t, action_is_index=True).view(-1)

        if self.UT_trick:
            # get (B, 2 * action_dim + 1, action_dim) samples;
            UT_trick_samples = policy_density.get_UT_trick_input()

            # eval expectation of q-target functions by averaging over the
            # 2 * action_dim + 1 samples and get (B, 1) output;
            qt1_est = batch_UT_trick_from_samples(
                self.Q1t, obs_tp1, UT_trick_samples
            )
            qt2_est = batch_UT_trick_from_samples(
                self.Q2t, obs_tp1, UT_trick_samples
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
                    policy_density.log_prob(action_tp1).sum(-1).view(-1)
                )

            # input for target nets;
            obs_action_tp1 = torch.cat((obs_tp1, action_tp1), -1)
            # estimate values with target nets;
            qt1_est, qt2_est = self.Q1t(obs_action_tp1), self.Q2t(
                obs_action_tp1
            )

        # use the values from the target net that
        # had lower value predictions;
        q_target = (
            torch.min(qt1_est, qt2_est).view(-1)
            - agent.log_temperature.exp() * log_probs
        )

        q_target = (
            reward_t + (1 - terminated_tp1.int()) * self.discount * q_target
        ).detach()

        # loss for first q func;
        self.Q1_loss = nn.MSELoss()(q1_est, q_target)

        # loss for second q func;
        self.Q2_loss = nn.MSELoss()(q2_est, q_target)

        # bookkeeping;
        self.Q1_losses.append(self.Q1_loss.item())
        self.Q2_losses.append(self.Q2_loss.item())

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
            observations, actions, rewards, code = sample_eval_path(
                self.num_eval_steps_to_sample,
                self.env,
                self,
                seed=self.buffer.seed - 1,
            )
            self.eval_path_returns.append(np.sum(rewards))
            self.eval_path_lens.append(len(actions))
            
            # check if can cache;
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

                # grad step on policy, temperature, Q1 and Q2;
                self.update_parameters()

                # target q funcs update;
                self.track_qfunc_params()


def save_metrics(
    save_returns_to,
    metric_names,
    metrics,
    agent_name,
    env_name,
    seed,
    config=None,
    edge_index=None,
    last_eval_rewards=None,
    suptitle=None,
):
    now = time.time()
    new_dir = agent_name + f"-{env_name}-seed-{seed}-{now}"
    new_dir = save_returns_to / new_dir
    new_dir.mkdir(parents=True)
    save_returns_to = new_dir

    # illustrate the graph building stages if edge_index supplied;
    if edge_index is not None:
        vis_graph_building(edge_index, save_returns_to)
        file_name = save_returns_to / "edge-index.pkl"
        with open(file_name, "wb") as f:
            pickle.dump(edge_index, f)

    # save rewards from last eval episode if given;
    if last_eval_rewards is not None:
        file_name = save_returns_to / "last-eval-episode-rewards.pkl"
        with open(file_name, "wb") as f:
            pickle.dump(last_eval_rewards, f)

    # save pickle files;
    for metric_name, metric in zip(metric_names, metrics):
        file_name = f"{metric_name}-seed-{seed}.pkl"
        file_name = save_returns_to / file_name
        with open(file_name, "wb") as f:
            pickle.dump(metric, f)
    if config is not None:
        file_name = save_returns_to / "config.pkl"
        with open(file_name, "wb") as f:
            pickle.dump(config, f)

    # save plots of the metrics;
    save_metric_plots(
        metric_names,
        metrics,
        save_returns_to,
        seed,
        suptitle=suptitle,
    )
