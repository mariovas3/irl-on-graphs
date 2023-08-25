import networkx as nx
import networkx.algorithms as nxalgos

from math import sqrt
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

from typing import Callable
from pathlib import Path

from torch_geometric.data import Data
import pickle

import torch


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
    prefix_name_of_stats='targetgraph_'
):
    for n, s in zip(names_of_stats, stats):
        p = prefix_name_of_stats + n + '.pkl'
        p = target_graph_dir / p
        with open(p, 'wb') as f:
            pickle.dump(s, f)


def train_eval_new_policy(new_policy, num_epochs, 
                          save_edge_index=False,
                          vis_graph=False, with_pos=False):
    new_policy.train_k_epochs(num_epochs, 
                              save_edge_index=save_edge_index,
                              vis_graph=vis_graph,
                              with_pos=with_pos)
    _, _, code, _, _, obs, _ = new_policy.buffer.get_single_ep_rewards_and_weights(
        new_policy.env,
        new_policy,
    )
    print(f"new_policy eval code after {num_epochs} epochs: {code}")
    return obs


def eval_irl_policy(irl_policy, env):
    old_env = irl_policy.env
    irl_policy.env = env
    _, _, code, _, _, obs, _ = irl_policy.buffer.get_single_ep_rewards_and_weights(
        irl_policy.env,
        irl_policy,
    )
    irl_policy.env = old_env
    print(f"irl_policy eval code on target_graph: {code}")
    return obs


def get_sum_euc_dist(tgdata, idxs: torch.Tensor=None):
    if idxs is None:
        idxs = torch.arange(tgdata.x.shape[-1]).view(1, -1)
    e = tgdata.edge_index[:, ::2]
    print(f"num edges x num features for euc dist:", tgdata.x[e[0].view(-1, 1), idxs].shape)
    ans = torch.norm(tgdata.x[e[0].view(-1, 1), idxs] - tgdata.x[e[1].view(-1, 1), idxs], 
                     p=2, dim=-1).sum().item()
    return ans


def save_graph_stats_k_runs_GO1(
        irl_policy, 
        irl_reward_fn,
        new_policy_constructor,
        num_epochs_new_policy,
        target_graph: Data,
        run_k_times: int,
        new_policy_param_getter_fn: Callable,
        sort_metrics: bool=False,
        euc_dist_idxs=None,
        save_edge_index=False,
        vis_graph=False,
        with_pos=False,
        **policy_extra_params,
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
        prefix_name_of_stats='targetgraph_'
    )

    # see if should calculate euclidean distances;
    if euc_dist_idxs is not None:
        if euc_dist_idxs is not None:
            ans = get_sum_euc_dist(target_graph,
                                euc_dist_idxs)
            save_graph_stats(target_graph_dir, ['eucdist'],
                            [ans],
                            prefix_name_of_stats='targetgraph_')
            print(f"og graph has sum of edge dists: {ans}")

    # init empty edge set for target graph;
    for k in range(run_k_times):
        # get params for new policy;
        _, new_policy_kwargs, _, irl_kwargs = new_policy_param_getter_fn()
        new_policy_kwargs['env_kwargs']['reward_fn'] = irl_reward_fn
        new_policy_kwargs['buffer_kwargs']['verbose'] = False
        
        # instantiate new policy/agent;
        new_policy = new_policy_constructor(
            **policy_extra_params, **new_policy_kwargs
        )
        if irl_kwargs['ortho_init']:
            new_policy.OI_init_nets()
        
        # save to same dir;
        new_policy.save_to = irl_policy.save_to
        
        # train new policy and then run eval and get constructed graph;
        out_graph = train_eval_new_policy(
            new_policy, num_epochs_new_policy,
            save_edge_index=save_edge_index,
            vis_graph=vis_graph,
            with_pos=with_pos,
        )

        # see if should calculate euclidean distances;
        if euc_dist_idxs is not None:
            ans = get_sum_euc_dist(out_graph, euc_dist_idxs)
            save_graph_stats(target_graph_dir, ['eucdist'],
                             [ans], 
                             prefix_name_of_stats=f"newpolicy_{k}_")
            print(f"new policy has sum of edge dists: {ans}")
        
        # get graph stats;
        stats = get_graph_stats(out_graph, sort_metrics)
        save_graph_stats(target_graph_dir, names_of_stats,
                         stats,
                         prefix_name_of_stats=f"newpolicy_{k}_")
        
        # eval policy from irl training directly on target graph;
        out_graph = eval_irl_policy(irl_policy, 
                                    new_policy.env)
        
        # see if should calculate euclidean distances;
        if euc_dist_idxs is not None:
            ans = get_sum_euc_dist(out_graph, euc_dist_idxs)
            save_graph_stats(target_graph_dir, ['eucdist'],
                             [ans], 
                             prefix_name_of_stats=f"newpolicy_{k}_")
            print(f"irl policy has sum of edge dists: {ans}")
        
        # get graph stats;
        stats = get_graph_stats(out_graph, sort_metrics)
        save_graph_stats(
            target_graph_dir, names_of_stats, stats, 
            prefix_name_of_stats=f"irlpolicy_{k}_"
        )


def get_five_num_summary(data):
    low, high = min(data), max(data)
    mean, median, std = np.mean(data), np.median(data), np.std(data)
    return [low, median, high, mean, std]


def get_means_and_stds(metrics):
    ms = np.array(metrics)
    if ms.ndim < 2:
        ms = ms.reshape(1, -1)
    return np.vstack(
        (ms.mean(0, keepdims=True), ms.std(0, keepdims=True))
    )


def get_stats_of_metrics_and_metrics_in_dict(read_from, verbose=False):
    groups = {}
    summary_stats_over_runs = {}
    for f in read_from.iterdir():
        file_name = str(f).split('/')[-1]
        tokens = file_name.split('_')
        who = tokens[0]
        metric_name = tokens[-1][:-4]
        g = who + '_' + metric_name
        with open(f, 'rb') as f:
            # data should be list;
            data = pickle.load(f)
        if hasattr(data, '__len__'):
            five_num_summary = get_five_num_summary(data)
            if g not in groups:
                groups[g] = data
                summary_stats_over_runs[g] = [five_num_summary]
            else:
                groups[g] += data
                summary_stats_over_runs[g].append(five_num_summary)
    if verbose:
        print("metric names:\n", list(groups.keys()))
    f = get_means_and_stds
    summary_stats_over_runs = {
        n: f(s) for (n, s)
        in summary_stats_over_runs.items()
    }
    for k, v in sorted(summary_stats_over_runs.items(), key=lambda x: x[0]):
        print(k, v, sep='\n', end="\n\n")
    return summary_stats_over_runs, groups


def plot_dists(file_name, groups, suptitle):
    rows = len(set([gr.split('_')[0] for gr in groups.keys()]))
    cols = int(len(groups) / rows)
    cols *= 2  # hist and kde -> 2 plots per metric;
    fig = plt.figure(figsize=(14, 8))
    for i, (metric_name, metric) in enumerate(
        sorted(groups.items(), key=lambda x: x[0])
    ):
        xlow, xhigh = min(metric) - .01, max(metric) + .01

        # get hist;
        plt.subplot(rows, cols, 2 * i + 1)
        ax = plt.gca()
        ax.hist(metric)
        ax.set_ylabel(metric_name)
        ax.set_xticks(np.arange(xlow, xhigh, (xhigh - xlow) / 5).round(1))
        ax.set_xticklabels(ax.get_xticks(), rotation=90)
        
        # get gauss kde;
        plt.subplot(rows, cols, 2 * (i + 1))
        ax = plt.gca()
        try:
            metric = sorted(metric)
            kde = gaussian_kde(metric)
            plt.plot(metric, kde(metric))
            ax.set_ylabel(metric_name)
            ax.set_title('scipy gauss_kde')
            ax.set_xticks(np.arange(xlow, xhigh, 
                                    (xhigh - xlow) / 5).round(1))
            ax.set_xticklabels(ax.get_xticks(), rotation=90)
        except np.linalg.LinAlgError:
            print(f'problem with covariance in {metric_name}'
                " probably all values are the same")
    if suptitle is not None:
        fig.suptitle(suptitle)
    fig.tight_layout()
    # plt.show()
    plt.savefig(file_name)
    plt.close()


def save_analysis_graph_stats(test_output_path,
                              file_name, 
                              suptitle=None, verbose=False):
    p = Path(__file__).absolute().parent.parent
    p = p / test_output_path
    summary_stats, groups = get_stats_of_metrics_and_metrics_in_dict(
        p, verbose
    )
    file_name = p / file_name
    plot_dists(file_name, groups, suptitle)


def save_stats_edge_index(
    path, names_of_stats, tgdata, prefix_name_of_stats
):
    target_graph_dir = path / 'target_graph_stats'
    if not target_graph_dir.exists():
        target_graph_dir.mkdir(parents=True)
    stats = get_graph_stats(tgdata)
    save_graph_stats(
        target_graph_dir, names_of_stats, stats, prefix_name_of_stats
    )
    with open(path / (prefix_name_of_stats + 'edgeidx.pkl'), 'wb') as f:
        pickle.dump(tgdata.edge_index.tolist(), f)


def run_on_train_nodes_k_times(irl_agent, names_of_stats, k):
    p = irl_agent.save_to
    train_graph_gen_dir = p / 'train_graph_gen_dir'
    if not train_graph_gen_dir.exists():
        train_graph_gen_dir.mkdir(parents=True)
    for i in range(k):
        out_graph = eval_irl_policy(irl_agent, 
                                        irl_agent.env)
        # get graph stats;
        stats = get_graph_stats(out_graph, sort=False)
        save_graph_stats(
            train_graph_gen_dir, names_of_stats, stats, 
            prefix_name_of_stats=f"irlpolicy_{i}_"
        )


def get_mrr_and_avg(
    irl_policy, env, positives_dict, num_runs=3, k_proposals=10
):
    rr_means = []
    avg_found_rates = []
    for j in range(num_runs):
        positives_dict = {k: 0 for k in positives_dict.keys()}
        _, _, code, steps_done, _, _, rrs = irl_policy.buffer.get_single_ep_rewards_and_weights(
            env, 
            irl_policy,
            with_mrr=True,
            with_auc=True,
            positives_dict=positives_dict,
            k_proposals=k_proposals,
        )
        rr_means.append(np.mean(rrs))
        t = sum(positives_dict.values()) / len(positives_dict)
        avg_found_rates.append(t)
        print(f"from MRR EVAL, irl-policy did {steps_done} steps and code is {code}")
    rr_stats = get_five_num_summary(rr_means)
    found_rates_stats = get_five_num_summary(avg_found_rates)
    return rr_stats, found_rates_stats
