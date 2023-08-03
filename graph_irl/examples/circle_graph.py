import torch
import matplotlib.pyplot as plt
from networkx import Graph, draw_networkx
from typing import Callable


def create_circle_graph(n_nodes, node_dim, init_fn: Callable):
    # circular graph with n_nodes;
    a = [0] + torch.repeat_interleave(
            torch.arange(1, n_nodes), 2
        ).tolist() + [0]
    
    # get edge_index;
    edge_index = torch.tensor([
        a,
        ((
            torch.tensor([1, -1] * (len(a) // 2)) + torch.tensor(a)
        ) % n_nodes).tolist()
    ], dtype=torch.long)
    
    # create nodes;
    nodes = init_fn((edge_index.shape[-1] // 2, node_dim))

    return nodes, edge_index


if __name__ == "__main__":
    n_nodes, init_fn = 11, torch.ones
    nodes, edge_index = create_circle_graph(n_nodes, 5, init_fn)
    G = Graph()
    G.add_edges_from(list(zip(*edge_index.tolist())))
    fig = plt.figure(figsize=(8, 8))
    draw_networkx(G)
    plt.show()
