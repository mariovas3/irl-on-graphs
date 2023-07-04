import numpy as np
import torch
import torch.distributions as dists
from policy import GaussPolicy, Qfunc
from torch import nn
from typing import Iterable
from pathlib import Path
import random
from buffer import get_loader
# from copy import deepcopy
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
        **policy_kwargs
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
        **policy_kwargs
    ):
        super(SACAgentMuJoCo, self).__init__(
            name,
            policy,
            policy_lr,
            entropy_lb,
            temperature_lr,
            **policy_kwargs
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
            self.temperature_loss = - (
                self.log_temperature.exp() * (log_prob + self.entropy_lb).detach()
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


def get_q_losses(qfunc1, qfunc2, qt1, qt2, obs_t, action_t,
                 reward_t, obs_tp1, action_tp1, terminated_tp1, agent, discount,
                 resample_action_tp1=True):
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

    # see if I should resample action_tp1;
    if resample_action_tp1:   
        action_tp1 = policy_density.sample()
    
    # get log probs;
    log_probs = policy_density.log_prob(action_tp1).detach().sum(-1).view(-1)
    
    obs_action_tp1 = torch.cat((obs_tp1, action_tp1), -1)
    # use the values from the target net that
    # had lower value predictions;
    q_target = torch.min(
        qt1(obs_action_tp1),
        qt2(obs_action_tp1)
    ).view(-1) - agent.log_temperature.exp() * log_probs
    q_target = (reward_t + (1. - terminated_tp1.float()) * discount * q_target).detach()

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
    num_episodes,
    qfunc_hiddens,
    qfunc_layer_norm,
    policy_hiddens,
    policy_layer_norm,
    lr,
    buffer_len,
    batch_size,
    discount,
    tau,
    entropy_lb,
    seed,
    T=200,
    save_returns_to: Path = None,
):
    """
    This is currently some pseudo code for training SAC agent.
    """
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
        'SACAgentMuJoCo',  # name;
        lr,  # policy_lr;
        entropy_lb,  # entropy_lb
        lr,  # temperature_lr
        obs_dim=obs_dim,
        action_dim=action_dim,
        hiddens=policy_hiddens,
        with_layer_norm=policy_layer_norm,
    )

    # init replay buffer of size=N deque?
    undiscounted_returns = []
    buffer, idx = [], 0

    # start running episodes;
    for _ in range(num_episodes):
        sampled_return = 0.0
        done, t = False, 0

        # sample starting state;
        obs_t, info = env.reset(seed=seed)
        obs_t = torch.FloatTensor(obs_t).view(-1)

        # sample first action;
        action_t = agent.sample_action(obs_t)

        # buffer = deque([])
        while not done and t < T:
            # act with action_t and get state and reward;;
            obs_tp1, reward_t, terminated, truncated, info = env.step(
                action_t.numpy()
            )
            obs_tp1 = torch.FloatTensor(obs_tp1).view(-1)

            # update return;
            sampled_return += reward_t

            # sample next action;
            action_tp1 = agent.sample_action(obs_tp1)

            # add experience to buffer;
            if len(buffer) < buffer_len:
                buffer.append((obs_t, action_t, reward_t, obs_tp1, action_tp1, terminated))
            else:
                buffer_idx = idx % buffer_len
                buffer[buffer_idx] = (obs_t, action_t, reward_t, obs_tp1, action_tp1, terminated)
            idx += 1

            # update current observation and action;
            obs_t, action_t = obs_tp1, action_tp1
            done = terminated or truncated
            t += 1

        # add sampled episodic return to list;
        undiscounted_returns.append(sampled_return)

        # prep buffer to sample minibatches of experience.
        data_loader = get_loader(
            buffer, batch_size=batch_size, shuffle=False
        )

        # do the gradient updates;
        for i, (
            obs_t,
            action_t,
            reward_t,
            obs_tp1,
            action_tp1,
            terminated_tp1
        ) in enumerate(data_loader, start=1):
            # get temperature and policy loss;
            agent.get_policy_loss_and_temperature_loss(obs_t, Q1, Q2)
                        
            # value func updates;
            l1, l2 = get_q_losses(
                Q1, Q2, Q1t, Q2t, obs_t, 
                action_t, reward_t, obs_tp1, 
                action_tp1, terminated_tp1, agent, 
                discount, 
                resample_action_tp1=False
            )

            agent.update_policy_and_temperature()
            # agent.policy_optim.zero_grad()
            # agent.policy_loss.backward()
            # agent.policy_optim.step()
            
            # agent.temperature_optim.zero_grad()
            # agent.temperature_loss.backward()
            # agent.temperature_optim.step()
            
            # update q funcs;
            update_q_funcs(l1, l2, optimQ1, optimQ2)

            # target q funcs update;
            track_params(Q1t, Q1, tau)
            track_params(Q2t, Q2, tau)

            # return (0, 0, 0, 0)

            # update policy and temperature;
            # agent.update_policy_and_temperature()

    # optionally save for this seed from all episodes;
    if save_returns_to:
        file_name = agent.name + f"-{env.spec.id}-seed-{seed}.pkl"
        file_name = save_returns_to / file_name
        with open(file_name, "wb") as f:
            pickle.dump(undiscounted_returns, f)
    return Q1, Q2, agent, undiscounted_returns


if __name__ == "__main__":
    import gymnasium as gym
    # import matplotlib.pyplot as plt


    env = gym.make('Hopper-v2')
    num_episodes=200
    Q1, Q2, agent, undiscounted_returns = train_sac(
        env, num_episodes, [28, 28], False, [22, 22], True,
        3e-4, 10_000, 250, 0.99, 0.05, None, 0, 200, TEST_OUTPUTS_PATH
    )
