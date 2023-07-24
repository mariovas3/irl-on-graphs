"""
TODO:
    (1): Implement Q function to work with the batch 
        format defined in the README.
"""


import torch
from torch import nn
import torch.distributions as dists
from pathlib import Path
from graph_irl.distributions import *
from graph_irl.graph_rl_utils import get_action_vector_from_idx
from torch_geometric.nn import GCNConv, global_mean_pool


TEST_OUTPUTS_PATH = Path(__file__).absolute().parent.parent / "test_output"
if not TEST_OUTPUTS_PATH.exists():
    TEST_OUTPUTS_PATH.mkdir()


class GCN(nn.Module):
    def __init__(
        self, in_dim, hiddens, with_layer_norm=False, final_tanh=False
    ):
        super(GCN, self).__init__()

        # set attributes from constructor;
        self.hiddens = hiddens
        self.with_layer_norm = with_layer_norm
        self.final_tanh = final_tanh

        # init network;
        self.net = nn.ModuleList()

        # create a dummy list for ease of creating net;
        temp = [in_dim] + hiddens

        for i in range(len(temp) - 1):
            self.net.add_module(f"GCNCov{i}", GCNConv(temp[i], temp[i + 1]))

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        for i, f in enumerate(self.net):
            # GNN pass;
            x = f(x, edge_index)

            # activation follow-up;
            if i == len(self.net) - 1:
                if self.final_tanh:
                    x = torch.tanh(x)
                else:
                    x = torch.relu(x)
            else:
                x = torch.relu(x)

            # see if layer norm is needed;
            if self.with_layer_norm:
                x = torch.layer_norm(x, (self.hiddens[i],))
        # return avg node embedding for each graph in the batch;
        # together with node embeddings;
        return global_mean_pool(x, batch.batch), x


class AmortisedGaussNet(nn.Module):
    def __init__(self, obs_dim, action_dim, hiddens, with_layer_norm=True):
        super(AmortisedGaussNet, self).__init__()

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

        # add mean and Cholesky of diag covariance net;
        self.mu_net = nn.Linear(hiddens[-1], action_dim)
        self.std_net = nn.Sequential(
            nn.Linear(hiddens[-1], action_dim), nn.Softplus()
        )

    def forward(self, obs):
        emb = self.net(obs)  # shared embedding for mean and std;
        return self.mu_net(emb), torch.clamp(self.std_net(emb), 0.01, 4.0)


class GaussPolicy(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        hiddens,
        with_layer_norm=False,
        encoder=None,
        two_action_vectors=False,
    ):
        """
        Args:
            obs_dim (int): Observation dimension if encoder is None,
                otherwise it's the embedding dimension from the encoder.
            action_dim (int): The dimension of the action.
            hiddens (List[int]): Sizes of hidden layers for
                amortised Gaussian net.
            with_layer_norm (bool): Whether to put layer norm
                transforms in the Amortised net.
            encoder (nn.Module): Intended to be GNN instance.
            two_action_vectors (bool): If True, output a single
                action vector that has size 2 * action_dim.
        """
        super(GaussPolicy, self).__init__()
        self.two_action_vectors = two_action_vectors

        # set encoder;
        self.encoder = encoder

        self.name = (
            "GaussPolicy" if self.encoder is None else "GraphGaussPolicy"
        )

        # get dimension for indep Gauss vector.
        if two_action_vectors:
            out_dim = 2 * action_dim
        else:
            out_dim = action_dim

        # init net;
        self.net = AmortisedGaussNet(
            obs_dim, out_dim, hiddens, with_layer_norm
        )

    def forward(self, obs):
        if self.encoder is not None:
            obs, node_embeds = self.encoder(obs)  # (B, obs_dim)
        mus, sigmas = self.net(obs)
        if self.encoder is not None:
            return (
                GaussDist(dists.Normal(mus, sigmas), self.two_action_vectors),
                node_embeds,
            )
        return GaussDist(dists.Normal(mus, sigmas))


class TwoStageGaussPolicy(nn.Module):
    def __init__(
        self,
        obs_dim,
        action_dim,
        hiddens1,
        hiddens2,
        encoder,
        with_layer_norm=False,
    ):
        super(TwoStageGaussPolicy, self).__init__()
        self.encoder = encoder
        self.name = "TwoStageGaussPolicy"

        self.net1 = AmortisedGaussNet(
            obs_dim, action_dim, hiddens1, with_layer_norm
        )
        self.net2 = AmortisedGaussNet(
            obs_dim + action_dim, action_dim, hiddens2, with_layer_norm
        )

    def forward(self, obs):
        obs, node_embeds = self.encoder(obs)
        mus, sigmas = self.net1(obs)
        return (
            TwoStageGaussDist(dists.Normal(mus, sigmas), obs, self.net2),
            node_embeds,
        )


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
        if two_actions:
            raise NotImplementedError(
                "Haven't implemented " "TanhGaussPolicy for " "graphs yet."
            )
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

    def forward(self, obs_action, action_is_index=False):
        """
        If self.encoder is not None, then obs_action is (batch, actions).

        If action_is_index is True, then the actions
            are a tensor of shape (B, 2) saying which nodes to connect.
        """
        if self.encoder is not None:
            batch, actions = obs_action
            obs, node_embeds = self.encoder(batch)
            if action_is_index:
                actions = get_action_vector_from_idx(
                    node_embeds, actions, batch.batch
                )
            else:
                actions = torch.cat(actions, -1)
            obs_action = torch.cat((obs, actions), -1)
        return self.net(obs_action)
