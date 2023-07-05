import numpy as np
import torch
from policy import GaussPolicy, Qfunc
from torch import nn
from typing import Iterable
from pathlib import Path
import random
from buffer_v2 import Buffer
from vis_utils import save_metric_plots, see_one_episode
import time
import pickle

TEST_OUTPUTS_PATH = Path(".").absolute().parent / "test_output"
if not TEST_OUTPUTS_PATH.exists():
    TEST_OUTPUTS_PATH.mkdir()


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

    def get_policy_loss_and_temperature_loss(self, *args, **kwargs):
        pass

    def update_policy_and_temperature(self, *args, **kwargs):
        pass


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
        # if isinstance(policy_dist, dists.Normal):
        #     # you can access mean and stddev
        #     # with policy_dist.mean and policy_dist.stddev;
        return action

    def update_policy_and_temperature(self):
        # print(self.policy_loss, self.temperature_loss)
        # update policy params;
        self.policy_optim.zero_grad()
        self.policy_loss.backward()
        self.policy_optim.step()

        # update temperature;
        self.temperature_optim.zero_grad()
        self.temperature_loss.backward()
        self.temperature_optim.step()

    def get_policy_loss_and_temperature_loss(
        self, obs_t, qfunc1, qfunc2, use_entropy=False
    ):
        # self.policy.requires_grad_(True)
        # self.log_temperature.requires_grad_(True)

        # get policy;
        policy_density = self.policy(obs_t)

        # do reparam trick;
        repr_trick = policy_density.rsample()

        # for some policies, we can compute entropy;
        if use_entropy:
            with torch.no_grad():
                entropies = policy_density.entropy().sum(-1)

        # get log prob for policy optimisation;
        log_prob = policy_density.log_prob(repr_trick).sum(-1)

        # freeze q-nets and eval at current observation
        # and reparam trick action and choose min for policy loss;
        qfunc1.requires_grad_(False)
        qfunc2.requires_grad_(False)
        qfunc_in = torch.cat((obs_t, repr_trick), -1)
        q_est = torch.min(qfunc1(qfunc_in), qfunc2(qfunc_in)).view(-1)

        # get loss for policy;
        self.policy_loss = (
            np.exp(self.log_temperature.item()) * log_prob - q_est
        ).mean()

        # get loss for temperature;
        if use_entropy:
            self.temperature_loss = -(
                self.log_temperature.exp()
                * (entropies - self.entropy_lb).detach().mean()
            )
        else:
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
):
    qfunc1.requires_grad_(True)
    qfunc2.requires_grad_(True)

    # freeze policy and temperature;
    # agent.policy.requires_grad_(False)
    # agent.log_temperature.requres_grad_(False)

    # get predictions from q functions;
    obs_action_t = torch.cat((obs_t, action_t), -1)
    q1_est = qfunc1(obs_action_t).view(-1)
    q2_est = qfunc2(obs_action_t).view(-1)

    # see appropriate dist for obs_tp1;
    policy_density = agent.policy(obs_tp1)

    # sample future action;
    action_tp1 = policy_density.sample()

    # get log probs;
    log_probs = policy_density.log_prob(action_tp1).detach().sum(-1).view(-1)

    obs_action_tp1 = torch.cat((obs_tp1, action_tp1), -1)
    # use the values from the target net that
    # had lower value predictions;
    q_target = (
        torch.min(qt1(obs_action_tp1), qt2(obs_action_tp1)).view(-1)
        - agent.log_temperature.exp() * log_probs
    )
    # print(q_target.dtype)
    # print(reward_t.dtype, (1-terminated_tp1.int()).dtype, (discount * q_target).dtype)
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


def train_sac(
    env,
    num_iters,
    qfunc_hiddens,
    qfunc_layer_norm,
    lr,
    buffer_len,
    batch_size,
    discount,
    tau,
    entropy_lb,
    seed,
    policy,
    save_returns_to: Path = None,
    num_steps_to_sample=500,
    num_grad_steps=500,
    avg_the_returns=False,
    **policy_kwargs,
):
    """
    This is currently doing num_iters train iterations
    each iteration gathers trajectories given current
    agent and then does num_grad_steps gradient steps;

    might be good to add another loop over epochs
    as in the original sac paper.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    obs_action_dim = obs_dim + action_dim

    # init 2 q nets;
    Q1 = Qfunc(obs_action_dim, qfunc_hiddens, with_layer_norm=qfunc_layer_norm)
    Q2 = Qfunc(obs_action_dim, qfunc_hiddens, with_layer_norm=qfunc_layer_norm)
    optimQ1 = torch.optim.Adam(Q1.parameters(), lr=lr)
    optimQ2 = torch.optim.Adam(Q2.parameters(), lr=lr)

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
    agent = SACAgentMuJoCo(
        "SACAgentMuJoCo",  # name;
        policy,  # policy;
        lr,  # policy_lr;
        entropy_lb,  # entropy_lb
        lr,  # temperature_lr
        **policy_kwargs,
    )

    # init replay buffer of size=N deque?
    undiscounted_returns = []
    qfunc1_losses, qfunc2_losses = [], []
    buffer = Buffer(buffer_len, obs_dim, action_dim)

    # start running episodes;
    fifth = num_iters // 5
    now = time.time()
    # might be good to do another (outer) for loop over epochs
    # since num_iters corresponds to train loops per epoch;
    # i.e., currently this code does 100 train loops in this one epoch;
    # also might be good to clear the buffer at each epoch if
    # I do an outer loop over epochs;
    for it in range(num_iters):
        # sample paths;
        if (it + 1) % fifth == 0:
            print(
                f"{20 * (it + 1) // fifth}% train iterations processed\n"
                f"took {(time.time() - now) / 60:.3f} minutes"
            )
        buffer.collect_path(
            env,
            agent,
            num_steps_to_sample,
            undiscounted_returns,
            avg_the_returns,  # if true, gives avg reward in episode.
        )

        # do the gradient updates;
        for _ in range(num_grad_steps):
            obs_t, action_t, reward_t, obs_tp1, terminated_tp1 = buffer.sample(
                batch_size
            )
            # get temperature and policy loss;
            agent.get_policy_loss_and_temperature_loss(obs_t, Q1, Q2)

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

    # optionally save for this seed from all episodes;
    if save_returns_to:
        metric_names = [
            "policy-loss",
            "temperature-loss",
            "qfunc1-loss",
            "qfunc2-loss",
            "undiscounted-returns",
        ]
        metrics = [
            agent.policy_losses,
            agent.temperature_losses,
            qfunc1_losses,
            qfunc2_losses,
            undiscounted_returns,
        ]
        if avg_the_returns:
            metric_names[-1] = "avg-reward-per-episode"

        for metric_name, metric in zip(metric_names, metrics):
            file_name = (
                agent.name + f"-{env.spec.id}-{metric_name}-seed-{seed}.pkl"
            )
            file_name = save_returns_to / file_name
            with open(file_name, "wb") as f:
                pickle.dump(metric, f)
        save_metric_plots(
            agent.name,
            env.spec.id,
            metric_names,
            metrics,
            save_returns_to,
            seed,
        )
    return Q1, Q2, agent, undiscounted_returns


if __name__ == "__main__":
    import gymnasium as gym

    T = 1000

    env = gym.make("Hopper-v2", max_episode_steps=T)
    num_iters = 100  # this is the train iterations per epoch;
    Q1, Q2, agent, undiscounted_returns = train_sac(
        env,
        num_iters,
        qfunc_hiddens=[256, 256],
        qfunc_layer_norm=True,
        lr=3e-4,
        buffer_len=5_000,
        batch_size=250,
        discount=0.99,
        tau=0.05,
        entropy_lb=None,
        seed=0,
        policy=GaussPolicy,
        save_returns_to=TEST_OUTPUTS_PATH,
        num_steps_to_sample=500,
        num_grad_steps=500,
        avg_the_returns=True,
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hiddens=[256, 256],
        with_layer_norm=True,
    )

    env = gym.make(
        "Hopper-v2", max_episode_steps=T, render_mode="human"
    )
    see_one_episode(env, agent, seed=0)
