import networkx as nx
import networkx.algorithms as nxalgos

from torch_geometric.data import Data
import pickle


"""
Stats computation section.
"""


def tgdata_to_nxgraph(tgdata):
    G = nx.Graph()
    G.add_edges_from(list(zip(*tgdata.edge_index.tolist())))
    G.add_nodes_from(range(tgdata.num_nodes))
    return G


def get_clustering_coefs(G: nx.Graph, sort=False):    
    coefs = nxalgos.cluster.clustering(G)
    if sort:
        return sorted(coefs.values())
    return list(coefs.values())


def get_degrees(G: nx.Graph, sort=False):
    if sort:
        return [d[1] for d in sorted(G.degree, key=lambda x: x[1])]
    return [d[1] for d in G.degree]


def get_triangle_count(G: nx.Graph, sort=False):
    if sort:
        return sorted(nxalgos.cluster.triangles(G).values())
    return list(nxalgos.cluster.triangles(G).values())


def get_graph_stats(tgdata: Data, sort=False):
    G = tgdata_to_nxgraph(tgdata)
    triangles = get_triangle_count(G, sort)
    degrees = get_degrees(G, sort)
    clust_coefs = get_clustering_coefs(G, sort)
    return triangles, degrees, clust_coefs


"""
Saving stuff to files.
"""


def save_graph_stats(
    target_graph_dir, 
    names_of_stats, 
    stats, 
    prefix_name_of_stats='og_target_'
):
    for n, s in zip(names_of_stats, stats):
        p = prefix_name_of_stats + n + '.pkl'
        p = target_graph_dir / p
        with open(p, 'wb') as f:
            pickle.dump(s, f)


def train_eval_new_policy(new_policy, num_epochs):
    new_policy.train_k_epochs(num_epochs)
    _, _, code, _, _, obs = new_policy.buffer.get_single_ep_rewards_and_weights(
        new_policy.env,
        new_policy,
    )
    print(f"new_policy eval code after {num_epochs} epochs: {code}")
    return obs


def eval_irl_policy(irl_policy, env):
    old_env = irl_policy.env
    irl_policy.env = env
    _, _, code, _, _, obs = irl_policy.buffer.get_single_ep_rewards_and_weights(
        irl_policy.env,
        irl_policy,
    )
    irl_policy.env = old_env
    print(f"irl_policy eval code on target_graph: {code}")
    return obs


def save_graph_stats_k_runs_GO1(
        irl_policy, 
        new_policy_constructor,
        num_epochs_new_policy,
        target_graph: Data,
        run_k_times: int,
        sort_metrics: bool=False,
        **new_policy_kwargs,
):
    """
    Save graph statistics of target_graph, as well as run_k_times
    constructions (1): directly from irl_learned policy,
    (2): from learning a new policy on irl_reward.

    Notes: 
        
    """
    target_graph_dir = irl_policy.save_to / 'target_graph_stats'
    if not target_graph_dir.exists():
        target_graph_dir.mkdir(parents=True)
    
    names_of_stats = [
        'triangles',
        'degrees',
        'clustcoefs'
    ]

    # get stats of original target graph;
    stats = get_graph_stats(target_graph, sort_metrics)

    # save stats of original target graph;
    save_graph_stats(
        target_graph_dir, names_of_stats, stats, 
        prefix_name_of_stats='og_target_'
    )

    # init empty edge set for target graph;
    for k in range(run_k_times):
        new_policy = new_policy_constructor(**new_policy_kwargs)
        
        # save to same dir;
        new_policy.save_to = irl_policy.save_to
        out_graph = train_eval_new_policy(
            new_policy, num_epochs_new_policy
        )
        stats = get_graph_stats(out_graph, sort_metrics)
        save_graph_stats(target_graph_dir, names_of_stats,
                         stats,
                         prefix_name_of_stats=f"newpolicy_{k}_")
        
        out_graph = eval_irl_policy(irl_policy, 
                                    new_policy.env)
        stats = get_graph_stats(out_graph, sort_metrics)
        save_graph_stats(
            target_graph_dir, names_of_stats, stats, 
            prefix_name_of_stats=f"irlpolicy_{k}_"
        )
