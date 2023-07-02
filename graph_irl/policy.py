import numpy as np
import random
import torch
from torch import nn
import torch.distributions as dists
from collections import deque
from typing import Iterable
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pickle


TEST_OUTPUTS_PATH = Path(".").absolute().parent / "test_output"
if not TEST_OUTPUTS_PATH.exists():
    TEST_OUTPUTS_PATH.mkdir()


class GaussPolicy(nn.Module):
    def __init__(
        self, obs_dim, action_dim, hiddens, with_layer_norm=False
    ):
        super(GaussPolicy, self).__init__()

        # init net;
        self.net = nn.Sequential()

        # add modules/Layers to net;
        for i in range(len(hiddens)):
            if i == 0:
                self.net.append(nn.Linear(obs_dim, hiddens[i]))
            else:
                self.net.append(nn.Linear(hiddens[i - 1], hiddens[i]))

            # ReLU activation;
            self.net.append(nn.ReLU())
            if with_layer_norm:
                self.net.append(nn.LayerNorm(hiddens[i]))

        # add Affine layer for mean and stds for indep Gauss vector.
        self.mu_net = nn.Linear(hiddens[-1], action_dim)
        self.std_net = nn.Linear(hiddens[-1], action_dim)

    def forward(self, obs):
        emb = self.net(obs)  # shared embedding for mean and std;
        return self.mu_net(emb), torch.log(1.0 + self.std_net(emb).exp())


class PGAgentBase:
    def __init__(
        self,
        name,
        obs_dim,
        action_dim,
        policy,
        with_baseline,
        lr,
        discount,
        **kwargs,
    ):
        self.rewards = []
        self.name = name
        self.log_probs = []
        self.baseline = 0.0
        self.policy = policy(obs_dim, action_dim, **kwargs)
        self.with_baseline = with_baseline
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.discount = discount

    def update_rewards(self, reward):
        pass

    def update_baseline(self):
        pass

    def update(self):
        pass

    def sample_action(self, obs):
        pass


class PGGauss(PGAgentBase):
    def __init__(
        self,
        name,
        obs_dim,
        action_dim,
        policy,
        with_baseline,
        lr,
        discount,
        **kwargs,
    ):
        super(PGGauss, self).__init__(
            name,
            obs_dim,
            action_dim,
            policy,
            with_baseline,
            lr,
            discount,
            **kwargs,
        )

    def update_rewards(self, reward):
        self.rewards.append(reward)

    def update_baseline(self):
        if self.with_baseline:
            self.baseline = self.baseline + (
                self.rewards[-1] - self.baseline
            ) / len(self.rewards)

    def sample_action(self, obs):
        mus, sigmas = self.policy(obs)
        policy_dist = dists.Normal(mus, sigmas)
        action = policy_dist.sample()
        self.log_probs.append(policy_dist.log_prob(action).sum())
        return action

    def update(self):
        curr_return = 0.0
        returns_to_go = torch.empty(len(self.rewards))

        for i, r in enumerate(self.rewards[::-1], start=1):
            if self.with_baseline:
                r = r - self.baseline
            curr_return = r + self.discount * curr_return
            returns_to_go[len(self.rewards) - i] = curr_return

        # get weighted loss;
        self.optim.zero_grad()
        loss = -torch.stack(self.log_probs) @ returns_to_go
        loss.backward()
        self.optim.step()

        # reset rewards, log_probs and baseline;
        self.rewards, self.log_probs, self.baseline = [], [], 0.0


class Qfunc(nn.Module):
    def __init__(self, obs_action_dim, hiddens, with_layer_norm=False):
        super(Qfunc, self).__init__()

        # init net;
        self.net = nn.Sequential()

        # add hidden layers;
        for i in range(len(hiddens)):
            if i == 0:
                self.net.append(nn.Linear(obs_action_dim, hiddens[i]))
            else:
                self.net.append(nn.Linear(hiddens[i - 1], hiddens[i]))

            # add ReLU non-linearity;
            self.net.append(nn.ReLU())
            if with_layer_norm:
                self.net.append(nn.LayerNorm(hiddens[i]))

        # Q-func maps to scalar;
        self.net.append(hiddens[-1], 1)

    def forward(self, obs_action):
        return self.net(obs_action)


def update_params(Q1, Q2, tau):
    """
    Updates parameters of Q1 to be:
    Q1_new = Q2 * tau + Q1_old * (1 - tau).
    """
    theta = nn.utils.parameters_to_vector(Q2.parameters()) * tau + (
        1 - tau
    ) * nn.utils.parameters_to_vector(Q1.parameters())

    # load theta as the new params of Q1;
    nn.utils.vector_to_parameters(theta, Q1.parameters())


# def train_sac(
#     obs_dim,
#     action_dim,
#     qfunc_hiddens,
#     qfunc_lns,
#     policy_hiddens,
#     policy_lns,
#     lr,
# ):
#     # init nets;
#     obs_action_dim = obs_dim + action_dim
#     Q1 = Qfunc(obs_action_dim, qfunc_hiddens, with_layer_norm=qfunc_lns)
#     Q2 = Qfunc(obs_action_dim, qfunc_hiddens, with_layer_norm=qfunc_lns)
#     Q3 = Qfunc(obs_action_dim, qfunc_hiddens, with_layer_norm=qfunc_lns)
#     Q4 = Qfunc(obs_action_dim, qfunc_hiddens, with_layer_norm=qfunc_lns)
#     policy = GaussPolicy(
#         obs_dim, action_dim, policy_hiddens, with_layer_norm=policy_lns
#     )
#     optim_q1 = torch.optim.Adam(Q1.parameters(), lr=lr)


class SACAgentBase:
    def __init__(self, policy, lr, **policy_kwargs):
        self.policy = policy(**policy_kwargs)
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def sample_action(self, obs):
        pass

    def update_policy(self, *args, **kwargs):
        pass


class SACAgentMuJoCo(SACAgentBase):
    def __init__(self, policy, lr, **policy_kwargs):
        super(SACAgentMuJoCo, self).__init__(policy, lr, **policy_kwargs)

    def sample_action(self, obs):
        mus, sigmas = self.policy(obs)
        policy_dist = dists.Normal(mus, sigmas)  # indep Gauss;
        action = policy_dist.sample()
        return action

    def update_policy(self, temperature, obs_t, qfunc, **kwargs):
        self.policy.require_grad_(True)

        # clear grads;
        self.optim.zero_grad()

        # get gauss params;
        mus, sigmas = self.policy(obs_t)

        # do reparam trick;
        repr_trick = torch.randn(mus.shape)  # samples from N(0, I);
        repr_trick = mus + repr_trick * sigmas

        # get Gauss density of policy;
        policy_density = dists.Normal(mus, sigmas)

        # compute entropies of gaussian to be passed
        # to optimisation step for temperature;
        with torch.no_grad():
            entropies = policy_density.entropy()

        # get avg log prob over batch dim for policy optimisation;
        avg_log_prob = policy_density.log_prob(repr_trick).sum(-1).mean()

        # freeze q-net;
        qfunc.require_grad_(False)

        # eval q-net at observations in minibatch and reparam trick actions.
        # take avg over batch dim;
        avg_q_vals = qfunc(torch.cat((obs_t, repr_trick), -1)).mean()
        # get loss for policy;
        loss = temperature * avg_log_prob - avg_q_vals
        loss.backward()
        self.optim.step()
        return entropies


def load_params_in_net(net: nn.Module, parameters: Iterable):
    """
    Loads parameters in the layers of net.
    """
    nn.utils.vector_to_parameters(
        nn.utils.parameters_to_vector(parameters), net.parameters()
    )


class Buffer(Dataset):
    def __init__(self, buffer):
        super(Buffer, self).__init__()
        self.buffer = buffer

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]


def get_loader(buffer: deque, batch_size, shuffle=False):
    return DataLoader(
        Buffer(buffer), batch_size=batch_size, shuffle=shuffle
    )


def update_q(
    Q,
    Qt,
    obs_t,
    action_t,
    reward_t,
    obs_tp1,
    action_tp1,
    optimQ,
    agent,
    temperature,
    discount,
):
    Q.requires_grad_(True)
    agent.policy.requires_grad_(False)

    # clear grads;
    optimQ.zero_grad()

    # get predictions of q net and target q net
    # averaged over batch dim;
    obs_action_t = torch.cat((obs_t, action_t), -1)
    obs_action_tp1 = torch.cat((obs_tp1, action_tp1), -1)
    avg_q_est = Q(obs_action_t).mean()
    avg_qt_est = Qt(obs_action_tp1)

    # get log prob of policy
    # averaged over batch dim;
    mus, sigmas = agent.policy(obs_tp1)
    avg_log_probs = (
        dists.Normal(mus, sigmas).log_prob(action_tp1).sum(-1).mean()
    )

    # get square loss;
    loss = (
        avg_q_est
        - reward_t
        - discount * (avg_qt_est - temperature * avg_log_probs)
    ) ** 2

    # get grad;
    loss.backward()

    # optimise q net;
    optimQ.step()

    # return avg q estimate;
    return avg_q_est.item()


# update_temperature(temperature, policy_entropies, entropy_lb)
def update_temperature(temperature, policy_entropies, entropy_lb, lr):
    return temperature - lr * (policy_entropies - entropy_lb)


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
    temperature,
    discount,
    tau,
    entropy_lb,
    seed,
    save_returns_to: Path = None,
):
    """
    This is currently some pseudo code for training SAC agent.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    obs_dim = env.state_space.shape[0]
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
        GaussPolicy,
        lr,
        obs_dim=obs_dim,
        action_dim=action_dim,
        hiddens=policy_hiddens,
        with_layer_norm=policy_layer_norm,
    )

    # init replay buffer of size=N deque?
    undiscounted_returns = []
    buffer = deque([])

    # start running episodes;
    for _ in range(num_episodes):
        sampled_return = 0.0
        done = False

        # sample starting state;
        obs_t, info = env.reset(seed=seed)
        obs_t = torch.FloatTensor(obs_t).view(-1)

        # sample first action;
        action_t = agent.sample_action(obs_t)

        # buffer = deque([])
        while not done:
            # act with action_t and get state and reward;;
            obs_tp1, reward_t, terminated, truncated, info = env.step(
                action_t.numpy()
            )
            obs_tp1 = torch.FloatTensor(obs_tp1).view(-1)

            # update return;
            sampled_return += reward_t

            # sample next action;
            action_tp1 = agent.sample_action(obs_tp1)

            # maintain the buffer with lenght N;
            if len(buffer) >= buffer_len:
                buffer.popleft()
            buffer.append((obs_t, action_t, reward_t, obs_tp1, action_tp1))

            # update current observation and action;
            obs_t, action_t = obs_tp1, action_tp1
            done = terminated or truncated

        # add sampled episodic return to list;
        undiscounted_returns.append(sampled_return)

        # prep buffer to sample minibatches of experience.
        data_loader = get_loader(
            buffer, batch_size=batch_size, shuffle=True
        )

        # do the gradient updates;
        for i, (
            obs_t,
            action_t,
            reward_t,
            obs_tp1,
            action_tp1,
        ) in enumerate(data_loader, start=1):
            # value func updates;
            avg_q_est1 = update_q(
                Q1,
                Q1t,
                obs_t,
                action_t,
                reward_t,
                obs_tp1,
                action_tp1,
                optimQ1,
                agent,
                temperature,
                discount,
            )
            avg_q_est2 = update_q(
                Q2,
                Q2t,
                obs_t,
                action_t,
                reward_t,
                obs_tp1,
                action_tp1,
                optimQ2,
            )

            # target q-func updates;
            update_params(Q1t, Q1, tau)
            update_params(Q2t, Q2, tau)

            # update policy params with the arg min (q1, q2) q func;
            if avg_q_est1 < avg_q_est2:
                policy_entropies = agent.update_policy(
                    temperature, obs_t, Q1
                )
                # agent.update_policy(Q1, obs_t, action_t)
            else:
                policy_entropies = agent.update_policy(
                    temperature, obs_t, Q2
                )
                # agent.update_policy(Q2, obs_t, action_t)

            # update temperature;
            temperature = update_temperature(
                temperature, policy_entropies, entropy_lb, lr
            )

    # optionally save for this seed from all episodes;
    if save_returns_to:
        file_name = agent.name + f"-{env.spec.id}-seed-{seed}.pkl"
        file_name = save_returns_to / file_name
        with open(file_name, "wb") as f:
            pickle.dump(undiscounted_returns, f)
    return Q1, Q2, agent, undiscounted_returns
