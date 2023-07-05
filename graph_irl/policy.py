import torch
from torch import nn
import torch.distributions as dists
from pathlib import Path


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

    def forward(self, obs) -> dists.Distribution:
        emb = self.net(obs)  # shared embedding for mean and std;
        mus, sigmas = self.mu_net(emb), torch.log(1.0 + self.std_net(emb).exp())
        return dists.Normal(mus, sigmas)


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
        self.net.append(nn.Linear(hiddens[-1], 1))

    def forward(self, obs_action):
        return self.net(obs_action)
