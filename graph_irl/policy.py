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
import torch_geometric.nn as tgnn


TEST_OUTPUTS_PATH = Path(__file__).absolute().parent.parent / "test_output"
if not TEST_OUTPUTS_PATH.exists():
    TEST_OUTPUTS_PATH.mkdir()


class GCN(nn.Module):
    def __init__(
        self, in_dim, 
        hiddens, heads=1,
        with_batch_norm=False, final_tanh=False,
        bet_on_homophily=False, net2_batch_norm=False,
    ):
        super(GCN, self).__init__()

        # set attributes from constructor;
        self.hiddens = hiddens.copy()
        self.with_batch_norm = with_batch_norm
        self.final_tanh = final_tanh
        self.bet_on_homophily = bet_on_homophily
        if self.bet_on_homophily:
            assert self.hiddens[-1] % 2 == 0
            self.hiddens[-1] = self.hiddens[-1] // 2

        # init network;
        tgnet = []
        if bet_on_homophily:
            self.net2 = nn.Sequential()

        # create a dummy list for ease of creating net;
        temp = [in_dim] + self.hiddens

        for i in range(len(temp) - 1):
            assert temp[i + 1] % heads == 0
            tgnet.append((tgnn.GATv2Conv(temp[i], temp[i + 1] // heads, 
                                         heads=heads), 
                          f"x, edge_index -> x"))
            if i < len(temp) - 2:
                tgnet.append(nn.ReLU())
                if self.with_batch_norm:
                    tgnet.append(nn.BatchNorm1d(temp[i + 1], affine=True))
            if bet_on_homophily:
                self.net2.append(nn.Linear(temp[i], temp[i + 1]))
                if i < len(temp) - 2:
                    self.net2.append(nn.ReLU())
                    if net2_batch_norm:
                        self.net2.append(nn.BatchNorm1d(temp[i + 1], affine=True))
        
        # get graph net in Sequential container;
        self.net = tgnn.Sequential('x, edge_index', tgnet)

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        if self.bet_on_homophily:
            mlp_x = self.net2(x)
        x = self.net(x, edge_index)
        # return avg node embedding for each graph in the batch;
        # together with node embeddings;
        if self.bet_on_homophily:
            x = torch.cat((x, mlp_x), -1)
        if self.final_tanh:
            x = torch.tanh(x)
        return tgnn.global_mean_pool(x, batch.batch), x


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
            nn.Linear(hiddens[-1], action_dim), 
            nn.Softplus()
        )

    def forward(self, obs):
        emb = self.net(obs)  # shared embedding for mean and std;
        return self.mu_net(emb), self.std_net(emb)


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
        two_action_vectors=False,
    ):
        super(TanhGaussPolicy, self).__init__()
        self.two_action_vectors = two_action_vectors
        self.encoder = encoder

        self.name = (
            "TanhGaussPolicy"
            if self.encoder is None
            else "GraphTanhGaussPolicy"
        )

        if two_action_vectors:
            out_dim = 2 * action_dim
        else:
            out_dim = action_dim
        
        self.net = AmortisedGaussNet(
            obs_dim, out_dim, hiddens, with_layer_norm
        )

    def forward(self, obs):
        if self.encoder is not None:
            obs, node_embeds = self.encoder(obs)
        mus, sigmas = self.net(obs)
        if self.encoder is not None:
            return (
                TanhGauss(dists.Normal(mus, sigmas), self.two_action_vectors),
                node_embeds
            )
        return TanhGauss(dists.Normal(mus, sigmas))


class Qfunc(nn.Module):
    def __init__(
        self, obs_action_dim, hiddens, with_layer_norm=False, 
        with_batch_norm=False, encoder=None,
    ):
        super(Qfunc, self).__init__()

        # set encoder;
        self.encoder = encoder

        assert not (with_batch_norm and with_layer_norm)

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
            if with_batch_norm:
                self.net.append(nn.BatchNorm1d(hiddens[i]))

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
                num_graphs = 1
                if hasattr(batch, 'num_graphs'):
                    num_graphs = batch.num_graphs
                
                # get actions -> vector of idxs of nodes;
                actions = get_action_vector_from_idx(
                    node_embeds, actions, num_graphs
                )
            # else:
                # actions = torch.cat(actions, -1)
            obs_action = torch.cat((obs, actions), -1)
        return self.net(obs_action)


class Vfunc(nn.Module):
    def __init__(
        self, embed_dim, encoder, hiddens, with_layer_norm=False,
    ):
        super(Vfunc, self).__init__()

        # set encoder;
        self.encoder = encoder

        # init net;
        self.net = nn.Sequential()

        # add hidden layers;
        for i in range(len(hiddens)):
            if i == 0:
                self.net.append(nn.Linear(embed_dim, hiddens[i]))
            else:
                self.net.append(nn.Linear(hiddens[i - 1], hiddens[i]))

            # add ReLU non-linearity;
            self.net.append(nn.ReLU())
            if with_layer_norm:
                self.net.append(nn.LayerNorm(hiddens[i]))

        # V-func maps to scalar;
        self.net.append(nn.Linear(hiddens[-1], 1))

    def forward(self, obs):
        """
        obs is batch of graphs or single torch_geometric.data.Data object.S
        """
        obs, _ = self.encoder(obs)
        return self.net(obs)
