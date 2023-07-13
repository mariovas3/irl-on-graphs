import torch
from torch import nn
import torch.distributions as dists
from pathlib import Path
from .distributions import TanhGauss, GaussDist
from torch_geometric.nn import GCNConv, global_mean_pool
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
        for i in range(len(hiddens) - 1):
            self.net.add_module(
                f"GCNCov{i+1}", GCNConv(hiddens[i], hiddens[i + 1])
            )

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        for i, f in enumerate(self.net):
            x = torch.relu(f(x, edge_index))
            if self.with_layer_norm:
                x = torch.layer_norm(x, (self.hiddens[i + 1],))
        return global_mean_pool(x, batch.batch)


class GaussPolicy(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        hiddens,
        with_layer_norm=False,
        encoder=None,
        two_actions=False,
    ):
        super(GaussPolicy, self).__init__()
        self.name = (
            "GaussPolicy" if self.encoder is None else "GraphGaussPolicy"
        )
        self.two_actions = two_actions

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
        if two_actions:
            out_dim = 2 * action_dim
        else:
            out_dim = action_dim
        self.mu_net = nn.Linear(hiddens[-1], out_dim)
        self.std_net = nn.Sequential(
            nn.Linear(hiddens[-1], out_dim), nn.Softplus()
        )

    def forward(self, obs):
        if self.encoder is not None:
            obs = self.encoder(obs)  # (B, obs_dim)
        emb = self.net(obs)  # shared embedding for mean and std;
        mus, sigmas = self.mu_net(emb), torch.clamp(
            self.std_net(emb), 0.01, 4.0
        )
        if self.encoder is not None:
            return GaussDist(dists.Normal(mus, sigmas)), obs
        return GaussDist(dists.Normal(mus, sigmas))


class TanhGaussPolicy(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        hiddens,
        with_layer_norm=False,
        encoder=None,
        two_actions=False,
    ):
        super(TanhGaussPolicy, self).__init__()
        self.name = (
            "TanhGaussPolicy"
            if self.encoder is None
            else "GraphTanhGaussPolicy"
        )
        self.gauss_dist = GaussPolicy(
            obs_dim,
            action_dim,
            hiddens,
            with_layer_norm,
            encoder,
            two_actions,
        )

    def forward(self, obs):
        if self.gauss_dist.encoder is not None:
            p, embeds = self.gauss_dist(obs)
            return TanhGauss(p), embeds
        return TanhGauss(self.gauss_dist(obs))


class Qfunc(nn.Module):
    def __init__(
        self, obs_action_dim, hiddens, with_layer_norm=False, encoder=None
    ):
        super(Qfunc, self).__init__()

        # set encoder;
        self.encoder = encoder

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
        """
        if self.encoder is not None, then obs_action is (batch, action).
        """
        if self.encoder is not None:
            obs = self.encoder(obs_action[0])
            obs_action = torch.cat((obs, obs_action[-1]), -1)
        return self.net(obs_action)
