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
