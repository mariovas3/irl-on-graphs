"""
TODO:
    (1) Finish this file by essentially testing the
        functionality of the GraphBuffer class that 
        should be written in buffer_v2 or some other place.
"""
import sys
from pathlib import Path

p = Path(__file__).absolute().parent.parent
sys.path.append(str(p))

from graph_irl.graph_rl_utils import GraphEnv
from graph_irl.policy import TwoStageGaussPolicy, GCN

import torch


if __name__ == "__main__":
    n_nodes = 10
    encoder_hiddens = [20, 20]
    x = torch.eye(n_nodes)  # 10 nodes;
    encoder = GCN(n_nodes, encoder_hiddens,
                  with_layer_norm=True)
    policy = TwoStageGaussPolicy(
        encoder_hiddens[-1], encoder_hiddens[-1],
        [30, 30], [40, 40], encoder, with_layer_norm=True
    )
    env = GraphEnv(x, lambda data: data.x.mean(), 8, 16, 5)

