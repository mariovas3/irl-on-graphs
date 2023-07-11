from scipy.spatial import KDTree
import torch


def get_knn(x, k=1):
    tree = KDTree(x.numpy())
    return torch.from_numpy(tree.query(x, k=k)[-1])
