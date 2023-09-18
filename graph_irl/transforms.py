"""
TODO:
    (1): Dynamic node attributes should be features of the graph, 
        so should be computed before the GNN pass. It might also 
        be good to append graph-level info such as sum of distances
        to the output of the global_pool operator in the GNN model.
"""

import torch
from torch_geometric.utils import degree
from torch_geometric.nn import (
    global_mean_pool,
    global_max_pool,
    global_add_pool,
)

from itertools import groupby
from typing import Callable


class InplaceBatchNodeFeatTransform:
    """
    Concats columns to node features (batch.x API in torch_geometric).
    """

    def __init__(
        self,
        in_dim,
        transform_fn_: Callable,
        n_cols_append,
        col_names_ridxs,
        n_extra_cols_append,
        get_graph_level_feats_fn: Callable = None,
    ):
        """
        Note:
            transform_fn works on tg Batch or Data objects.
            col_names_ridxs should be dict[str, int] where the
                int part gives the reverse index <= 0.
        """
        self.in_dim = in_dim
        self.transform_fn_ = transform_fn_
        self.n_cols_append = n_cols_append
        self.col_names_ridxs = col_names_ridxs
        assert n_cols_append == len(col_names_ridxs)

        # see if anything to append to graph-level representation;
        self.n_extra_cols_append = n_extra_cols_append

        # compute extra graph-level-feats according to graph_level_aggr;
        self.get_graph_level_feats_fn = get_graph_level_feats_fn

    def __call__(self, batch):
        # don't transform stuff that are not of expected input dimension;
        if batch.x.shape[-1] == self.in_dim:
            self.transform_fn_(batch)


def get_columns_aggr(
    batch, col_idxs, aggr: str = "sum", check_in_dim: int = None
):
    if check_in_dim is not None:
        assert batch.x.shape[-1] == check_in_dim
    if aggr == "sum":
        aggr = global_add_pool
    elif aggr == "mean":
        aggr = global_mean_pool
    elif aggr == "max":
        aggr = global_max_pool
    else:
        raise ValueError(
            "aggr expected to be one of 'sum', 'mean' or 'max', "
            f"but {aggr} was found."
        )
    return aggr(batch.x[:, col_idxs], batch.batch).view(-1, len(col_idxs))


def append_degrees_(batch):
    if batch.edge_index.numel() == 0:
        b = torch.zeros((len(batch.x),)).view(-1, 1)
        # augment node features inplace;
        batch.x = torch.cat((batch.x, b), -1)
        return
    num_graphs = 1
    if hasattr(batch, "num_graphs"):
        num_graphs = batch.num_graphs
    n_nodes = len(batch.x)
    degrees = degree(batch.edge_index[0, :], num_nodes=n_nodes).view(-1, 1)
    assert len(batch.x) == len(degrees)
    batch.x = torch.cat((batch.x, degrees), -1)


def get_sum_count(group):
    c, s = 0, 0.0
    for it in group:
        c += 1
        s += it[1]
    return [c, s]


def append_distances_(batch, with_degrees=False):
    # prep vector to append as a node feature;
    dim = 1
    if with_degrees:
        dim = 2
    b = torch.zeros((len(batch.x), dim)).view(-1, dim)

    if batch.edge_index.numel() == 0:
        # augment node features inplace;
        batch.x = torch.cat((batch.x, b), -1)
        return

    if with_degrees:
        f = get_sum_count
    else:
        f = lambda group: [sum([v[1] for v in group])]

    # assumes batch of undirected graphs with edges
    # (from, to), (to, from) one after the other;
    idxs = batch.edge_index[:, ::2].tolist()
    # get square euclid norm;
    ds = (
        ((batch.x[idxs[0], :] - batch.x[idxs[1], :]) ** 2).sum(-1).tolist()
    )

    # get list of (node_idx, score) tuples sorted by node_idx;
    a = sorted(zip(idxs[0] + idxs[1], ds * 2), key=lambda x: x[0])

    # get node_idxs as keys, and sum of scores as vals;
    keys, vals = zip(
        *[(k, f(g)) for (k, g) in groupby(a, key=lambda x: x[0])]
    )

    # populate new columns with relevant values;
    b[keys, :] = torch.tensor(vals).view(-1, dim)

    # augment node features inplace;
    batch.x = torch.cat((batch.x, b), -1)
