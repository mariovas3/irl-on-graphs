from scipy.spatial import KDTree
import torch


def add_edge_dynamics(a, x, data):
    obs_dim = x.shape[-1]
    tree = KDTree(x.numpy())
    action_dim = len(a)
    
    # graph opt like edge addition;
    # connects point closes to a1 to point 
    # closest to a2, these could be very far apart.
    # also not clear to me how NN will figure out
    # which ones to connect since they share a common embedding;
    # and a1 e.g., picks a point to connect, and a2 should depend
    # on the selection for a1.
    # I think Victor's 2 stage mechanism makes more sense
    # i.e., first select x1 = closest(x, a1)
    # and then incorporate that info somehow (mby distance weight)
    # like softmax to select x2 = func(x, dist_weights);
    if action_dim == 2 * obs_dim:
        a1, a2 = a[:obs_dim], a[obs_dim:]
        first = tree.query(a1, k=1)[-1]
        second = tree.query(a2, k=1)[-1]
    
    # connect 2 points closest to a;
    else:
        idxs = tree.query(a, k=2)[-1]
        first, second = idxs[0], idxs[1]
    data.edge_index = torch.cat(
            (
                data.edge_index, 
                torch.tensor([(first, second), (second, first)])
            ), -1
        )
