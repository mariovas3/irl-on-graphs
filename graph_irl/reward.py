import torch
from torch import nn
from graph_irl.graph_rl_utils import get_action_vector_from_idx


class GraphReward(nn.Module):
    def __init__(
        self, 
        encoder: nn.Module, 
        embed_dim: int, 
        hiddens: list, 
        with_layer_norm: bool=False
    ):
        super(GraphReward, self).__init__()
        # set given attributes;
        self.encoder = encoder
        self.hiddens = hiddens

        # create net assuming input will be
        # concat(graph_embed, node_embed1, node_embed2)
        self.net = nn.Sequential()
        temp = [embed_dim * 3] + hiddens
        for i in range(len(temp)-1):
            self.net.append(nn.Linear(temp[i], temp[i+1]))
            self.net.append(nn.ReLU())
            if with_layer_norm:
                self.net.append(nn.LayerNorm((temp[i+1], )))
        
        # up to here this corresponds to getting the cost function;
        self.net.append(nn.Linear(hiddens[-1], 1))
        self.net.append(nn.Softplus())
    
    def reset(self):
        pass

    def verbose(self):
        pass
    
    def forward(self, obs_action, action_is_index=False):
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
        else:
            actions = torch.cat(actions, -1)
        obs_action = torch.cat((obs, actions), -1)
        # return the negative of the cost -> reward;
        return -self.net(obs_action)
