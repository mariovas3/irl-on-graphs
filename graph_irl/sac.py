"""
TODO:
    (1): Implement the policy and temperature loss in the SAC graph
        agent. Currently needs the Qfunc to be implemented.
    * Implement GNN and train sac on graph task;
    * Maybe wrap the train loops for sac in a policy training
      class, to be easy to train for several iterations in the 
      IRL setting.
"""

import random
import numpy as np
import torch
from torch import nn

from torch_geometric.nn import global_mean_pool

from graph_irl.policy import *
from graph_irl.distributions import batch_UT_trick_from_samples
from graph_irl.buffer_v2 import Buffer, sample_eval_path
from graph_irl.vis_utils import *

from typing import Iterable
from pathlib import Path
import time
import pickle
from tqdm import tqdm

# path to save logs;
TEST_OUTPUTS_PATH = Path(__file__).absolute().parent.parent / "test_output"


class SACAgentBase:
    def __init__(
        self,
        name,
        policy,
        policy_lr,
        entropy_lb,
        temperature_lr,
        **policy_kwargs,
    ):
        self.name = name
        self.policy = policy(**policy_kwargs)
        self.policy_optim = torch.optim.Adam(
            self.policy.parameters(), lr=policy_lr
        )
        self.log_temperature = torch.tensor(0.0, requires_grad=True)
        self.temperature_optim = torch.optim.Adam(
            [self.log_temperature], lr=temperature_lr
        )
        self.entropy_lb = (
            entropy_lb if entropy_lb else -policy_kwargs["action_dim"]
        )
        self.temperature_loss = None
        self.policy_loss = None
        self.policy_losses = []
        self.temperature_losses = []

    def sample_action(self, obs):
        pass

    def sample_deterministic(self, obs):
        pass

    def get_policy_loss_and_temperature_loss(self, *args, **kwargs):
        pass

    def update_policy_and_temperature(self):
        # update policy params;
        self.policy_optim.zero_grad()
        self.policy_loss.backward()
        self.policy_optim.step()

        # update temperature;
        self.temperature_optim.zero_grad()
        self.temperature_loss.backward()
        self.temperature_optim.step()


class SACAgentGraph(SACAgentBase):
    def __init__(
        self,
        name,
        policy,
        policy_lr,
        entropy_lb,
        temperature_lr,
        **policy_kwargs,
    ):
        super(SACAgentGraph, self).__init__(
            name,
            policy,
            policy_lr,
            entropy_lb,
            temperature_lr,
            **policy_kwargs,
        )

    def sample_action(self, obs):
        policy_dist, node_embeds = self.policy(obs)
        return policy_dist.sample(), node_embeds

    def sample_deterministic(self, obs):
        policy_dist, node_embeds = self.policy(obs)
        mus1, mus2 = policy_dist.mean
        return (mus1.detach(), mus2.detach()), node_embeds

    def get_policy_loss_and_temperature_loss(
        self, obs_t, qfunc1, qfunc2, UT_trick=False, with_entropy=False
    ):
        # freeze q-nets and eval at current observation
        # and reparam trick action and choose min for policy loss;
        qfunc1.requires_grad_(False)
        qfunc2.requires_grad_(False)

        # get policy;
        policy_density, node_embeds = self.policy(obs_t)

        if UT_trick:
            # raise NotImplementedError("UT trick not implemented for the "
            #                           "TwoStageGauss Policy. "
            #                           "The cost is quadratic in "
            #                           "action dimension.")
            if with_entropy:
                log_pi_integral = policy_density.entropy().sum(-1)
            else:
                log_pi_integral = policy_density.log_prob_UT_trick().sum(-1)
            UT_trick_samples = policy_density.get_UT_trick_input()
            q1_integral = batch_UT_trick_from_samples(
                qfunc1.net, qfunc1.encoder(obs_t), UT_trick_samples
            )
            q2_integral = batch_UT_trick_from_samples(
                qfunc2.net, qfunc2.encoder(obs_t), UT_trick_samples
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
            if with_entropy:
                log_prob = policy_density.entropy().sum(-1)
            else:
                log_prob = policy_density.log_prob(repr_trick1, repr_trick2).sum(-1)

            obs_t = global_mean_pool(node_embeds, obs_t.batch)
            qfunc_in = torch.cat((obs_t, repr_trick1, repr_trick2), -1)
            q_est = torch.min(qfunc1.net(qfunc_in), qfunc2.net(qfunc_in)).view(-1)

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


class SACAgentMuJoCo(SACAgentBase):
    def __init__(
        self,
        name,
        policy,
        policy_lr,
        entropy_lb,
        temperature_lr,
        **policy_kwargs,
    ):
        super(SACAgentMuJoCo, self).__init__(
            name,
            policy,
            policy_lr,
            entropy_lb,
            temperature_lr,
            **policy_kwargs,
        )

    def sample_action(self, obs):
        policy_density = self.policy(obs)
        action = policy_density.sample()
        return action

    def sample_deterministic(self, obs):
        policy_density = self.policy(obs)
        return policy_density.mean.detach()

    def get_policy_loss_and_temperature_loss(
        self, obs_t, qfunc1, qfunc2, UT_trick=False, with_entropy=False
    ):
        # freeze q-nets and eval at current observation
        # and reparam trick action and choose min for policy loss;
        qfunc1.requires_grad_(False)
        qfunc2.requires_grad_(False)

        # get policy;
        policy_density = self.policy(obs_t)

        if UT_trick:
            if with_entropy:
                log_pi_integral = policy_density.entropy().sum(-1)
            else:
                log_pi_integral = policy_density.log_prob_UT_trick().sum(-1)
            UT_trick_samples = policy_density.get_UT_trick_input()
            q1_integral = batch_UT_trick_from_samples(
                qfunc1, obs_t, UT_trick_samples
            )
            q2_integral = batch_UT_trick_from_samples(
                qfunc2, obs_t, UT_trick_samples
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
            repr_trick = policy_density.rsample()

            # get log prob for policy optimisation;
            if with_entropy:
                log_prob = policy_density.entropy().sum(-1)
            else:
                log_prob = policy_density.log_prob(repr_trick).sum(-1)

            qfunc_in = torch.cat((obs_t, repr_trick), -1)
            q_est = torch.min(qfunc1(qfunc_in), qfunc2(qfunc_in)).view(-1)

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


def load_params_in_net(net: nn.Module, parameters: Iterable):
    """
    Loads parameters in the layers of net.
    """
    nn.utils.vector_to_parameters(
        nn.utils.parameters_to_vector(parameters), net.parameters()
    )


def get_q_losses(
    qfunc1,
    qfunc2,
    qt1,
    qt2,
    obs_t,
    action_t,
    reward_t,
    obs_tp1,
    terminated_tp1,
    agent,
    discount,
    UT_trick=False,
    with_entropy=False,
    for_graph=False,
):
    qfunc1.requires_grad_(True)
    qfunc2.requires_grad_(True)

    # get predictions from q functions;
    if for_graph:
        obs_action_t = (obs_t, action_t)
        policy_density, node_embeds = agent.policy(obs_tp1)
    else:
        obs_action_t = torch.cat((obs_t, action_t), -1)
        policy_density = agent.policy(obs_tp1)
    # the action_is_index boolean will only be considered
    # if self.encoder is not None;
    q1_est = qfunc1(obs_action_t, action_is_index=True).view(-1)
    q2_est = qfunc2(obs_action_t, action_is_index=True).view(-1)

    if UT_trick:
        # get (B, 2 * action_dim + 1, action_dim) samples;
        UT_trick_samples = policy_density.get_UT_trick_input()
        # eval expectation of q-target functions by averaging over the
        # 2 * action_dim + 1 samples and get (B, 1) output;
        if for_graph:
            qt1_est = batch_UT_trick_from_samples(
                qt1.net, qt1.encoder(obs_tp1), UT_trick_samples
            )
            qt2_est = batch_UT_trick_from_samples(
                qt2.net, qt2.encoder(obs_tp1), UT_trick_samples
            )
        else:
            qt1_est = batch_UT_trick_from_samples(
                qt1, obs_tp1, UT_trick_samples
            )
            qt2_est = batch_UT_trick_from_samples(
                qt2, obs_tp1, UT_trick_samples
            )
        # get negative entropy by using the UT trick;
        if with_entropy:
            log_probs = policy_density.entropy().sum(-1)
        else:
            log_probs = policy_density.log_prob_UT_trick().sum(-1)
    else:
        # sample future action;
        action_tp1 = policy_density.sample()

        # get log probs;
        if with_entropy:
            log_probs = policy_density.entropy().sum(-1).view(-1)
        else:
            if for_graph:
                log_probs = policy_density.log_prob(*action_tp1).sum(-1).view(-1)
            else:
                log_probs = policy_density.log_prob(action_tp1).sum(-1).view(-1)

        # input for target nets;
        if for_graph:
            obs_tp1 = global_mean_pool(node_embeds, obs_tp1.batch)
            obs_action_tp1 = torch.cat((obs_tp1,) + action_tp1, -1)
            qt1_est, qt2_est = qt1.net(obs_action_tp1), qt2.net(obs_action_tp1)
        else:
            obs_action_tp1 = torch.cat((obs_tp1, action_tp1), -1)
            # estimate values with target nets;
            qt1_est, qt2_est = qt1(obs_action_tp1), qt2(obs_action_tp1)

    # use the values from the target net that
    # had lower value predictions;
    q_target = (
        torch.min(qt1_est, qt2_est).view(-1)
        - agent.log_temperature.exp() * log_probs
    )
    
    q_target = (
        reward_t + (1 - terminated_tp1.int()) * discount * q_target
    ).detach()

    # loss for first q func;
    loss_q1 = nn.MSELoss()(q1_est, q_target)

    # loss for second q func;
    loss_q2 = nn.MSELoss()(q2_est, q_target)

    return loss_q1, loss_q2


def update_q_funcs(loss_q1, loss_q2, optim_q1, optim_q2):
    # update first q func;
    optim_q1.zero_grad()
    loss_q1.backward()
    optim_q1.step()

    # update second q func;
    optim_q2.zero_grad()
    loss_q2.backward()
    optim_q2.step()


def train_sac_one_epoch(
    env,
    agent,
    Q1,
    Q2,
    Q1t,
    Q2t,
    optimQ1,
    optimQ2,
    tau,
    discount,
    num_iters,
    num_grad_steps,
    num_steps_to_sample,
    num_eval_steps_to_sample,
    buffer,
    batch_size,
    qfunc1_losses: list,
    qfunc2_losses: list,
    UT_trick=False,
    with_entropy=False,
    for_graph=False,
    eval_path_returns: list = None,
    eval_path_lens: list = None,
):
    for _ in tqdm(range(num_iters)):
        # sample paths;
        buffer.collect_path(
            env,
            agent,
            num_steps_to_sample,
        )

        # sample paths with delta func policy;
        if eval_path_returns is not None and eval_path_lens is not None:
            observations, actions, rewards, code = sample_eval_path(
                num_eval_steps_to_sample, env, agent, seed=buffer.seed - 1
            )
            eval_path_returns.append(np.sum(rewards))
            eval_path_lens.append(len(actions))

        # do the gradient updates;
        for _ in range(num_grad_steps):
            (
                obs_t,
                action_t,
                reward_t,
                obs_tp1,
                terminated_tp1,
            ) = buffer.sample(batch_size)
            # get temperature and policy loss;
            agent.get_policy_loss_and_temperature_loss(
                obs_t, Q1, Q2, UT_trick, with_entropy
            )

            # value func updates;
            l1, l2 = get_q_losses(
                Q1,
                Q2,
                Q1t,
                Q2t,
                obs_t,
                action_t,
                reward_t,
                obs_tp1,
                terminated_tp1,
                agent,
                discount,
                UT_trick,
                with_entropy,
                for_graph,
            )

            # qfunc losses housekeeping;
            qfunc1_losses.append(l1.item())
            qfunc2_losses.append(l2.item())

            # grad step on policy and temperature;
            agent.update_policy_and_temperature()

            # grad steps on q funcs;
            update_q_funcs(l1, l2, optimQ1, optimQ2)

            # target q funcs update;
            track_params(Q1t, Q1, tau)
            track_params(Q2t, Q2, tau)


def save_metrics(
    save_returns_to, metric_names, metrics, agent_name, env_name, seed
):
    now = time.time()
    new_dir = agent_name + f"-{env_name}-seed-{seed}-{now}"
    new_dir = save_returns_to / new_dir
    new_dir.mkdir(parents=True)
    save_returns_to = new_dir
    for metric_name, metric in zip(metric_names, metrics):
        file_name = agent_name + f"-{env_name}-{metric_name}-seed-{seed}.pkl"
        file_name = save_returns_to / file_name
        with open(file_name, "wb") as f:
            pickle.dump(metric, f)

    # save plots of the metrics;
    save_metric_plots(
        agent_name,
        env_name,
        metric_names,
        metrics,
        save_returns_to,
        seed,
    )


def train_sac(
    env,
    agent,
    num_iters,
    qfunc_hiddens,
    qfunc_layer_norm,
    qfunc_lr,
    buffer_len,
    batch_size,
    discount,
    tau,
    seed,
    save_returns_to: Path = None,
    num_steps_to_sample=500,
    num_eval_steps_to_sample=500,
    num_grad_steps=500,
    num_epochs=100,
    min_steps_to_presample=1000,
    UT_trick=False,
    with_entropy=False,
    for_graph=False,
    **agent_policy_kwargs,
):
    # instantiate necessary objects;
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    obs_action_dim = obs_dim + action_dim

    # init 2 q nets;
    Q1 = Qfunc(
        obs_action_dim, qfunc_hiddens, with_layer_norm=qfunc_layer_norm
    )
    Q2 = Qfunc(
        obs_action_dim, qfunc_hiddens, with_layer_norm=qfunc_layer_norm
    )
    optimQ1 = torch.optim.Adam(Q1.parameters(), lr=qfunc_lr)
    optimQ2 = torch.optim.Adam(Q2.parameters(), lr=qfunc_lr)

    # init 2 target qnets with same parameters as q1 and q2;
    Q1t = Qfunc(
        obs_action_dim, qfunc_hiddens, with_layer_norm=qfunc_layer_norm
    )
    Q2t = Qfunc(
        obs_action_dim, qfunc_hiddens, with_layer_norm=qfunc_layer_norm
    )
    load_params_in_net(Q1t, Q1.parameters())
    load_params_in_net(Q2t, Q2.parameters())

    # target nets only track params of q-nets;
    # don't optimise them explicitly;
    Q1t.requires_grad_(False)
    Q2t.requires_grad_(False)

    # make agent;
    agent = agent(
        **agent_policy_kwargs["agent_kwargs"],
        **agent_policy_kwargs["policy_kwargs"],
    )

    # whether entropy and UT are used;
    agent.name = (
        agent.name
        + f"-{agent.policy.name}-UT-{int(UT_trick)}-entropy-{int(with_entropy)}-buffer-size-{buffer_len}-epochs-{num_epochs}-iters-{num_iters}"
    )

    # init replay buffer;
    qfunc1_losses, qfunc2_losses = [], []
    buffer = Buffer(buffer_len, obs_dim, action_dim, seed=seed)

    # see if presampling needed.
    if min_steps_to_presample > 0:
        buffer.collect_path(env, agent, min_steps_to_presample)

    eval_path_returns = []
    eval_path_lens = []
    # start running episodes;
    for _ in tqdm(range(num_epochs)):
        train_sac_one_epoch(
            env,
            agent,
            Q1,
            Q2,
            Q1t,
            Q2t,
            optimQ1,
            optimQ2,
            tau,
            discount,
            num_iters,
            num_grad_steps,
            num_steps_to_sample,
            num_eval_steps_to_sample,
            buffer,
            batch_size,
            qfunc1_losses,
            qfunc2_losses,
            UT_trick,
            with_entropy,
            for_graph,
            eval_path_returns,
            eval_path_lens,
        )

    # optionally save for this seed from all episodes;
    if save_returns_to:
        metric_names = [
            "policy-loss",
            "temperature-loss",
            "qfunc1-loss",
            "qfunc2-loss",
            "avg-reward",
            "path-lens",
            "ma-path-lens-30",
            "undiscounted-returns",
            "ma-returns-30",
            "eval-path-returns",
            "eval-ma-returns-30",
            "eval-path-lens",
            "eval-ma-path-lens-30",
        ]
        metrics = [
            agent.policy_losses,
            agent.temperature_losses,
            qfunc1_losses,
            qfunc2_losses,
            buffer.avg_rewards_per_episode,
            buffer.path_lens,
            get_moving_avgs(buffer.path_lens, 30),
            buffer.undiscounted_returns,
            get_moving_avgs(buffer.undiscounted_returns, 30),
            eval_path_returns,
            get_moving_avgs(eval_path_returns, 30),
            eval_path_lens,
            get_moving_avgs(eval_path_lens, 30),
        ]

        # save the metrics as numbers as well as plot;
        save_metrics(
            save_returns_to,
            metric_names,
            metrics,
            agent.name,
            env.spec.id,
            seed,
        )

    # return the q funcs and the agent;
    return Q1, Q2, agent
