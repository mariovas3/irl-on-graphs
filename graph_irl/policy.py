import torch
from torch import nn
import torch.distributions as dists
from pathlib import Path
from distributions import TanhGauss, GaussDist
from torch_geometric.nn import GCNConv
from torch_geometric.utils import degree


TEST_OUTPUTS_PATH = Path(".").absolute().parent / "test_output"
if not TEST_OUTPUTS_PATH.exists():
    TEST_OUTPUTS_PATH.mkdir()


class GCN(nn.Module):
    def __init__(self, in_dim, hiddens, with_layer_norm=False):
        super(GCN, self).__init__()
        self.net = nn.ModuleList()
        hiddens = [in_dim] + hiddens
        self.hiddens = hiddens
        self.with_layer_norm = with_layer_norm
        for i in range(len(hiddens)-1):
            self.net.add_module(f"GCNCov{i+1}", GCNConv(hiddens[i], hiddens[i+1]))
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for i, f in enumerate(self.net):
            x = torch.relu(f(x, edge_index))
            if self.with_layer_norm:
                x = torch.layer_norm(x, (self.hiddens[i+1], ))
        degrees = degree(edge_index[0])
        # get a weighted avg of node features according to degree of nodes;
        return (x * (degrees / degrees.sum()).view(len(x), 1)).sum(0)


class GraphGaussPolicy(nn.Module):
    def __init__(
        self, obs_dim, action_dim, hiddens, with_layer_norm=False,
        encoder=None,
    ):
        super(GraphGaussPolicy, self).__init__()
        self.name = "GraphGaussPolicy"

        # set encoder;
        self.encoder = encoder

        # init net;
        self.net = nn.Sequential(nn.LayerNorm(obs_dim))

        # add modules/Layers to net;
        for i in range(len(hiddens)):
            if i == 0:
                self.net.append(nn.Linear(obs_dim, hiddens[i]))
            else:
                self.net.append(nn.Linear(hiddens[i - 1], hiddens[i]))

            if with_layer_norm:
                self.net.append(nn.LayerNorm(hiddens[i]))
            
            # ReLU activation;
            self.net.append(nn.ReLU())

        # add Affine layer for mean and stds for indep Gauss vector.
        self.mu_net = nn.Linear(hiddens[-1], action_dim)
        self.std_net = nn.Sequential(
            nn.Linear(hiddens[-1], action_dim),
            nn.Softplus()
        )

    def forward(self, obs) -> dists.Distribution:
        if self.encoder:
            obs = self.encoder(obs)  # (B, obs_dim)
        emb = self.net(obs)  # shared embedding for mean and std;
        mus, sigmas = self.mu_net(emb), torch.clamp(self.std_net(emb), .01, 4.)
        return GaussDist(dists.Normal(mus, sigmas))


class GaussPolicy(nn.Module):
    def __init__(
        self, obs_dim, action_dim, hiddens, with_layer_norm=False
    ):
        super(GaussPolicy, self).__init__()
        self.name = "GaussPolicy"
        # init net;
        self.net = nn.Sequential(nn.LayerNorm(obs_dim))

        # add modules/Layers to net;
        for i in range(len(hiddens)):
            if i == 0:
                self.net.append(nn.Linear(obs_dim, hiddens[i]))
            else:
                self.net.append(nn.Linear(hiddens[i - 1], hiddens[i]))

            if with_layer_norm:
                self.net.append(nn.LayerNorm(hiddens[i]))
            
            # ReLU activation;
            self.net.append(nn.ReLU())

        # add Affine layer for mean and stds for indep Gauss vector.
        self.mu_net = nn.Linear(hiddens[-1], action_dim)
        self.std_net = nn.Sequential(
            nn.Linear(hiddens[-1], action_dim),
            nn.Softplus()
        )

    def forward(self, obs) -> dists.Distribution:
        emb = self.net(obs)  # shared embedding for mean and std;
        mus, sigmas = self.mu_net(emb), torch.clamp(self.std_net(emb), .01, 4.)
        return GaussDist(dists.Normal(mus, sigmas))


class TanhGaussPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hiddens, with_layer_norm=False):
        super(TanhGaussPolicy, self).__init__()
        self.name = "TanhGaussPolicy"
        self.gauss_dist = GaussPolicy(
            obs_dim, action_dim, hiddens, with_layer_norm
        )
    
    def forward(self, obs):
        return TanhGauss(self.gauss_dist(obs))


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
