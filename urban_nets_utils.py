from pathlib import Path
from torch_geometric.data import Data
import torch


p = Path(__file__).absolute().parent
DATA_PATH = p / 'data/Subways'


def extract_names_and_edge_set(edge_set_path):
    names = set()  # names of stations;
    undirected_edges = set()  # undirected edges;
    with open(edge_set_path, 'r') as f:
        for l in f:
            tokens = l.split()
            # sort lexicographically;
            first, second = min(tokens), max(tokens)
            if (first, second) in undirected_edges:
                continue
            undirected_edges.add((first, second))
            names.add(first)
            names.add(second)
    return names, undirected_edges


def extract_raw_node_feats(tube_positions_path, names_to_keep):
    names, positions = {}, []
    with open(tube_positions_path, mode='r') as f:
        i = 0
        for l in f:
            tokens = l.split()
            if tokens[0] in names:
                continue
            if tokens[0] in names_to_keep:
                names[tokens[0]] = i
                # add lattitude and longitude;
                positions.append((float(tokens[-2]), float(tokens[-1])))
                i += 1
    assert len(names) == len(positions)
    return names, torch.tensor(positions, dtype=torch.float32)


def get_edge_index(undirected_edge_set, node_name_to_idx):
    edge_index = [[], []]
    for e1, e2 in undirected_edge_set:
        # make sure all stations from the edge set are
        # in the node index;
        if not (e1 in node_name_to_idx and e2 in node_name_to_idx):
            raise AssertionError(
                'station names from edge set not in node index\n'
            )
        # add (i, j) and (j, i) to the edge index since 
        # it's undirected graph;
        edge_index[0].append(node_name_to_idx[e1])
        edge_index[1].append(node_name_to_idx[e2])
        edge_index[0].append(node_name_to_idx[e2])
        edge_index[1].append(node_name_to_idx[e1])
    return torch.tensor(edge_index, dtype=torch.long).view(2, -1)


def get_city_graph(positions_path, edge_set_path, scaling=None):
    names_to_keep, undirected_edges = extract_names_and_edge_set(
        edge_set_path
    )
    node_name_to_idx, node_feats = extract_raw_node_feats(
        positions_path, names_to_keep
    )
    edge_index = get_edge_index(
        undirected_edges, node_name_to_idx
    )
    if scaling is not None:
        node_feats = scaling(node_feats)
    return Data(x=node_feats, edge_index=edge_index)


def get_min_max_scaled_feats(X, feat_range=(0., 1.)):
    low, high = feat_range
    mins = X.min(0, keepdims=True).values
    maxes = X.max(0, keepdims=True).values
    X_std = (X - mins) / (
        maxes - mins
    )
    return X_std * (high - low) + low


def get_standard_scaled_feats(X):
    return (X - X.mean(0, keepdims=True)) / X.std(0, keepdims=True)
