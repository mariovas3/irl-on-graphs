"""
TODO:
    (1): Make batch processing functions take x, and edge_index 
        as input similarly to tgnn layers. This is so that I can 
        apply a transform exactly before global_agg_pool function 
        in the encoder pass.
    (2): Adapt existing code for change in (1).
    (3): Dynamic node attributes should be features of the graph, 
        so should be computed before the GNN pass. It might also 
        be good to append graph-level info such as sum of distances
        to the output of the global_pool operator in the GNN model.
    (4): To this end, transform should be passed as param to 
        the GNN instance rather than done outside of it.
"""

import torch
from torch_geometric.utils import degree

from itertools import groupby


class InplaceBatchNodeFeatTransform:
    """
    Concats columns to node features (batch.x API in torch_geometric).
    """
    def __init__(
        self,
        transform_fn_,
        n_cols_appended,
        col_names_ridxs,
        in_dim,
        graph_level_aggr: str=None
    ):
        """
        Note:
            transform_fn works on tg Batch or Data objects.
            col_names_ridxs should be dict[str, int] where the 
                int part gives the reverse index <= 0.
        """
        self.in_dim = in_dim
        self.transform_fn_ = transform_fn_
        self.n_cols_appended = n_cols_appended
        self.n_extra_cols_append = 0
        for k in col_names_ridxs.keys():
            if 'distance' in k:
                self.n_extra_cols_append += 1
        assert n_cols_appended == len(col_names_ridxs)
        self.col_names_ridxs = col_names_ridxs
        # this should be populated after transform_fn_ cal;
        self.graph_level_feats = None
        self.graph_level_aggr = graph_level_aggr
    
    def __call__(self, batch):
        # don't transform stuff that are not of expected
        # input dimension;
        if batch.x.shape[-1] == in_dim:
            self.transform_fn_(batch)
        
        # aggregate only new columns with distance substring;
        idxs = []
        for k, v in self.col_names_ridxs.items():
            if 'distance' in k:
                idxs.append(v)
        if self.graph_level_aggr == 'sum':
            self.graph_level_feats = batch.x[:, idxs].sum(
                0, keepdim=True
            )
        elif self.graph_level_aggr == 'mean':
            self.graph_level_feats = batch.x[:, idxs].mean(
                0, keepdim=True
            )


def append_degrees_(batch):
    if batch.edge_index.numel() == 0:
        b =  torch.zeros((len(batch.x), )).view(-1, 1)
        # augment node features inplace;
        batch.x = torch.cat((batch.x, b), -1)
        return
    num_graphs = 1
    if hasattr(batch, 'num_graphs'):
        num_graphs = batch.num_graphs
    n_nodes = len(batch.x)
    degrees = degree(batch.edge_index[0, :], num_nodes=n_nodes).view(-1, 1)
    assert len(batch.x) == len(degrees)
    batch.x = torch.cat((batch.x, degrees), -1)


def append_distances_(batch):
    # prep vector to append as a node feature;
    b =  - torch.ones((len(batch.x), )).view(-1, 1)

    if batch.edge_index.numel() == 0:
        # augment node features inplace;
        batch.x = torch.cat((batch.x, b), -1)
        return
    # assumes batch of undirected graphs with edges
    # (from, to), (to, from) one after the other;
    idxs = batch.edge_index[:, ::2].tolist()
    # get square euclid norm;
    ds = ((batch.x[idxs[0], :] - batch.x[idxs[1], :]) ** 2).sum(-1).tolist()
    
    # get list of (node_idx, score) tuples sorted by node_idx;
    a = sorted(zip(idxs[0] + idxs[1], ds * 2), key=lambda x: x[0])

    # get node_idxs as keys, and sum of scores as vals; 
    keys, vals = zip(*[
        (k, sum([v[1] for v in g])) 
        for (k, g) in groupby(a, key=lambda x: x[0])
    ])

    # subtract sum of distances in appropriate places;
    b[keys, :] = - torch.tensor(vals).view(-1, 1)

    # augment node features inplace;
    batch.x = torch.cat((batch.x, b), -1)
