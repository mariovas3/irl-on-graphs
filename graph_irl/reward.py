import torch
from torch import nn
from graph_irl.graph_rl_utils import get_action_vector_from_idx


class GraphReward(nn.Module):
    def __init__(
        self, 
        encoder: nn.Module, 
        embed_dim: int, 
        hiddens: list, 
        with_layer_norm: bool=False,
        with_batch_norm: bool=False,
    ):
        super(GraphReward, self).__init__()
        # set given attributes;
        self.encoder = encoder
        self.hiddens = hiddens

        assert not (with_batch_norm and with_layer_norm)

        # create net assuming input will be
        # concat(graph_embed, node_embed1, node_embed2)
        self.net = nn.Sequential()
        temp = [embed_dim * 3] + hiddens
        # if with_layer_norm:
                # self.net.append(nn.LayerNorm((temp[0], )))
        # if with_batch_norm:
            # self.net.append(nn.BatchNorm1d(temp[0]))
        for i in range(len(temp)-1):
            self.net.append(nn.Linear(temp[i], temp[i+1]))
            self.net.append(nn.ReLU())
            if with_layer_norm:
                self.net.append(nn.LayerNorm((temp[i+1], )))
            if with_batch_norm:
                self.net.append(nn.BatchNorm1d(temp[i+1]))
        
        # up to here this corresponds to getting the cost function;
        self.net.append(nn.Linear(hiddens[-1], 1))
        self.net.append(nn.Softplus())
    
    def reset(self):
        pass

    def verbose(self):
        pass
    
    def forward(
            self, 
            obs_action, 
            extra_graph_level_feats=None, 
            action_is_index=False,
            get_graph_embeds=False,
    ):
        batch, actions = obs_action
        obs, node_embeds = self.encoder(batch, extra_graph_level_feats)
        if action_is_index:
            num_graphs = 1
            if hasattr(batch, 'num_graphs'):
                num_graphs = batch.num_graphs
            
            # get actions -> vector of idxs of nodes;
            actions = get_action_vector_from_idx(
                node_embeds, actions, num_graphs
            )
        # else:
        #     actions = torch.cat(actions, -1)
        obs_action = torch.cat((obs, actions), -1)
        # return the negative of the cost -> reward;
        if get_graph_embeds:
            return -self.net(obs_action), obs
        return -self.net(obs_action)


class StateGraphReward(nn.Module):
    def __init__(
        self, 
        encoder: nn.Module, 
        embed_dim: int, 
        hiddens: list, 
        with_layer_norm: bool=False,
        with_batch_norm: bool=False,
    ):
        super(StateGraphReward, self).__init__()
        # set given attributes;
        self.encoder = encoder
        self.hiddens = hiddens

        assert not (with_batch_norm and with_layer_norm)

        # create net assuming input will be
        # graph_embed;
        self.net = nn.Sequential()
        temp = [embed_dim] + hiddens
        # if with_layer_norm:
                # self.net.append(nn.LayerNorm((temp[0], )))
        # if with_batch_norm:
            # self.net.append(nn.BatchNorm1d(temp[0]))
        for i in range(len(temp)-1):
            self.net.append(nn.Linear(temp[i], temp[i+1]))
            self.net.append(nn.ReLU())
            if with_layer_norm:
                self.net.append(nn.LayerNorm((temp[i+1], )))
            if with_batch_norm:
                self.net.append(nn.BatchNorm1d(temp[i+1]))
        
        # up to here this corresponds to getting the cost function;
        self.net.append(nn.Linear(hiddens[-1], 1))
        self.net.append(nn.Softplus())
    
    def reset(self):
        pass

    def verbose(self):
        pass
    
    def forward(self, graph_batch, extra_graph_level_feats=None,
                get_graph_embeds=False):
        obs, _ = self.encoder(graph_batch, extra_graph_level_feats)
        # return the negative of the cost -> reward;
        if get_graph_embeds:
            return -self.net(obs), obs
        return - self.net(obs)
