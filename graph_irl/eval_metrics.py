import networkx as nx
import networkx.algorithms as nxalgos

import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

from graph_irl.graph_rl_utils import edge_index_to_adj_list, get_dfs_edge_order

from typing import Callable, Tuple
from pathlib import Path

from torch_geometric.data import Data, Batch
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
                          vis_graph=False, with_pos=False,
                          do_graphopt=False):
    new_policy.train_k_epochs(num_epochs, 
                              save_edge_index=save_edge_index,
                              vis_graph=vis_graph,
                              with_pos=with_pos)
    new_policy.policy.eval()
    _, _, code, _, _, obs, _, rewards, _ = new_policy.buffer.get_single_ep_rewards_and_weights(
        new_policy.env,
        new_policy,
        reward_encoder=new_policy.old_encoder if do_graphopt else None
    )
    print(f"new_policy eval code after {num_epochs} epochs: {code}")
    return obs, rewards


def eval_irl_policy(irl_policy, env, do_graphopt=False):
    old_env = irl_policy.env
    irl_policy.env = env
    _, _, code, _, _, obs, _, rewards, _ = irl_policy.buffer.get_single_ep_rewards_and_weights(
        irl_policy.env,
        irl_policy,
        reward_encoder=irl_policy.old_encoder if do_graphopt else None
    )
    irl_policy.env = old_env
    print(f"irl_policy eval code on target_graph: {code}")
    return obs, rewards


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
        do_graphopt=False,
):
    """
    Save graph statistics of target_graph, as well as run_k_times
    constructions (1): directly from irl_learned policy,
    (2): from learning a new policy on irl_reward.

    Notes: 
        
    """
    # build adjacency list from edge index;
    adj_list = edge_index_to_adj_list(target_graph.edge_index[:, ::2], len(target_graph.x))
    
    # dir to save topo stats;
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
    
    r_perms, r_dfss = [], []
    for _ in range(3):
        r_perm, r_dfs = get_perm_and_dfs_rewards(
            np.random.randint(0, len(target_graph.x)),
            adj_list,
            target_graph.edge_index,
            target_graph.x,
            irl_policy
        )
        r_perms.append(r_perm)
        r_dfss.append(r_dfs)

    save_graph_stats(irl_policy.save_to, ['RandPermRewards'],
                        [r_perms],
                        prefix_name_of_stats=f"irlpolicyOnExpert_")
    
    save_graph_stats(irl_policy.save_to, ['DFSRewards'],
                        [r_dfss],
                        prefix_name_of_stats=f"irlpolicyOnExpert_")

    # init empty edge set for target graph;
    for k in range(run_k_times):
        # get params for new policy;
        agent_kwargs, new_policy_kwargs, _, irl_kwargs = new_policy_param_getter_fn()
        new_policy_kwargs['env_kwargs']['reward_fn'] = irl_reward_fn
        new_policy_kwargs['buffer_kwargs']['verbose'] = False
        agent_kwargs['save_to'] = None
        
        # instantiate new policy/agent;
        new_policy = new_policy_constructor(
            **agent_kwargs, **new_policy_kwargs
        )
        if irl_kwargs['ortho_init']:
            new_policy.OI_init_nets()
        
        # save to same dir;
        new_policy.save_to = irl_policy.save_to
        
        # train new policy and then run eval and get constructed graph;
        out_graph, rewards = train_eval_new_policy(
            new_policy, num_epochs_new_policy,
            save_edge_index=save_edge_index,
            vis_graph=vis_graph,
            with_pos=with_pos,
            do_graphopt=do_graphopt,
        )
        # save rewards on construction path;
        save_graph_stats(irl_policy.save_to, ['rewards'],
                         [rewards],
                         prefix_name_of_stats=f"newpolicyOnTarget_{k}_")
        
        
        r_perms, r_dfss = [], []
        for _ in range(3):
            r_perm, r_dfs = get_perm_and_dfs_rewards(
                np.random.randint(0, len(target_graph.x)),
                adj_list,
                target_graph.edge_index,
                target_graph.x,
                new_policy
            )
            r_perms.append(r_perm)
            r_dfss.append(r_dfs)

        save_graph_stats(irl_policy.save_to, ['RandPermRewards'],
                         [r_perms],
                         prefix_name_of_stats=f"newpolicyOnExpert_{k}_")
        
        save_graph_stats(irl_policy.save_to, ['DFSRewards'],
                         [r_dfss],
                         prefix_name_of_stats=f"newpolicyOnExpert_{k}_")

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
        out_graph, rewards = eval_irl_policy(irl_policy, 
                                    new_policy.env,
                                    do_graphopt=do_graphopt)

        # save rewards along reconstruction path;
        save_graph_stats(irl_policy.save_to, ['rewards'],
                         [rewards],
                         prefix_name_of_stats=f"irlpolicyOnTarget_{k}_")
        # see if should calculate euclidean distances;
        if euc_dist_idxs is not None:
            ans = get_sum_euc_dist(out_graph, euc_dist_idxs)
            save_graph_stats(target_graph_dir, ['eucdist'],
                             [ans], 
                             prefix_name_of_stats=f"irlpolicy_{k}_")
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


def run_on_train_nodes_k_times(irl_agent, names_of_stats, k,
                               do_graphopt=False):
    p = irl_agent.save_to
    train_graph_gen_dir = p / 'train_graph_gen_dir'
    if not train_graph_gen_dir.exists():
        train_graph_gen_dir.mkdir(parents=True)
    for i in range(k):
        out_graph, rewards = eval_irl_policy(irl_agent, 
                                        irl_agent.env,
                                        do_graphopt=do_graphopt)
        # get graph stats;
        stats = get_graph_stats(out_graph, sort=False)
        save_graph_stats(
            train_graph_gen_dir, names_of_stats, stats, 
            prefix_name_of_stats=f"irlpolicy_{i}_"
        )
        save_graph_stats(p, ['rewards'],
                         [rewards],
                         prefix_name_of_stats=f"irlpolicyOnSource_{i}_")


def get_mrr_and_avg(
    irl_policy, env, positives_dict, num_runs=3, k_proposals=10,
    do_graphopt=False,
):
    rr_means = []
    avg_found_rates = []
    for j in range(num_runs):
        positives_dict = {k: 0 for k in positives_dict.keys()}
        _, _, code, steps_done, _, _, rrs, _, _ = irl_policy.buffer.get_single_ep_rewards_and_weights(
            env, 
            irl_policy,
            with_mrr=True,
            with_auc=True,
            positives_dict=positives_dict,
            k_proposals=k_proposals,
            reward_encoder=irl_policy.old_encoder if do_graphopt else None
        )
        rr_means.append(np.mean(rrs))
        t = sum(positives_dict.values()) / len(positives_dict)
        avg_found_rates.append(t)
        print(f"from MRR EVAL, irl-policy did {steps_done} steps and code is {code}")
    rr_stats = get_five_num_summary(rr_means)
    found_rates_stats = get_five_num_summary(avg_found_rates)
    return rr_stats, found_rates_stats


def save_mrr_and_avg(
    agent, train_edge_index, positives_dict, graph_source,
    k_proposals=10, do_graphopt=False,
):
    # make env start from subsampled edge index and sample 
    # remaining num of edges;
    agent.env.expert_edge_index = train_edge_index
    agent.env.num_edges_start_from = train_edge_index.shape[-1] // 2
    num_expert_steps = graph_source.edge_index.shape[-1] // 2
    agent.env.spec.max_episode_steps = num_expert_steps
    agent.env.num_expert_steps = num_expert_steps
    agent.env.max_repeats = agent.env.max_self_loops = num_expert_steps

    mrr_stats, found_rate_stats = get_mrr_and_avg(
        agent, agent.env, positives_dict, k_proposals=k_proposals,
        do_graphopt=do_graphopt,
    )
    mrr_stats_dir = agent.save_to / 'mrr_stats'
    if not mrr_stats_dir.exists():
        mrr_stats_dir.mkdir()
    with open(mrr_stats_dir / 'mrr_stats.pkl', 'wb') as f:
        pickle.dump(mrr_stats, f)
    with open(mrr_stats_dir / 'found_rates.pkl', 'wb') as f:
        pickle.dump(found_rate_stats, f)
    print("MRR stats:", mrr_stats, 
          "found rates stats:", found_rate_stats, sep='\n')


def _get_single_ep_expert_return(
    edge_index,
    nodes,
    agent,
) -> Tuple[float, torch.Tensor]:
        """
        Returns sum of rewards along single expert trajectory as well as all
        rewards along the trajectory in 1-D torch.Tensor.
        """
        # permute even indexes of edge index;
        # edge index is assumed to correspond to an undirected graph;
        perm = np.random.permutation(
            range(0, edge_index.shape[-1], 2)
        )
        # expert_edge_index[:, (even_index, even_index + 1)] should
        # correspond to the same edge in the undirected graph in torch_geom.
        # this is because in torch_geometric, undirected graph duplicates
        # each edge e.g., (from, to), (to, from) are both in edge_index;
        idxs = sum([(i, i + 1) for i in perm], ())
        return _get_return_and_rewards_on_path(
            edge_index, nodes, idxs,
            agent
        )


def _get_single_ep_dfs_return(
    start_node, 
    adj_list, 
    edge_index,
    nodes,
    agent,
):
    edge_index = get_dfs_edge_order(adj_list, start_node)
    idxs = list(range(edge_index.shape[-1]))
    return _get_return_and_rewards_on_path(
            edge_index, nodes, idxs,
            agent
        )

def _get_return_and_rewards_on_path(
    edge_index, nodes, idxs, agent,
):
    pointer = 0
    return_val = 0.0
    cached_rewards = []
    reward_fn = agent.env.reward_fn
    reward_scale = agent.buffer.reward_scale
    graphs_per_batch = agent.buffer.graphs_per_batch
    reward_encoder = agent.old_encoder

    # the * 2 multiplier is because undirected graph is assumed
    # and effectively each edge is duplicated in edge_index;
    while (
        pointer * graphs_per_batch * 2
        < edge_index.shape[-1]
    ):
        batch_list = []
        action_idxs = []
        for i in range(
            pointer * graphs_per_batch * 2,
            min(
                (pointer + 1) * graphs_per_batch * 2,
                edge_index.shape[-1],
            ),
            2,
        ):
            batch_list.append(
                Data(
                    x=nodes,
                    edge_index=edge_index[
                        :, idxs[pointer * graphs_per_batch * 2 : i]
                    ],
                )
            )
            first, second = (
                edge_index[0, idxs[i]],
                edge_index[1, idxs[i]],
            )
            action_idxs.append([first, second])
            if agent.buffer.state_reward:
                batch_list[-1].edge_index = torch.cat(
                    (
                        batch_list[-1].edge_index, 
                        torch.tensor([[first, second], 
                                        [second, first]], dtype=torch.long)
                    ), -1
                )

        # create batch of graphs;
        batch = Batch.from_data_list(batch_list)
        extra_graph_level_feats = None
        if agent.buffer.transform_ is not None:
            agent.buffer.transform_(batch)
            if agent.buffer.transform_.get_graph_level_feats_fn is not None:
                extra_graph_level_feats = agent.buffer.transform_.get_graph_level_feats_fn(batch)

        pointer += 1
        assert agent.buffer.state_reward
        # check if in graphopt setting;
        out = batch
        if reward_encoder is not None:
            out = reward_encoder(batch, extra_graph_level_feats)[0]
        curr_rewards = reward_fn(out)
        
        curr_rewards = curr_rewards.view(-1) * reward_scale
        return_val += curr_rewards.sum()
        cached_rewards.append(curr_rewards)
    return return_val, torch.cat(cached_rewards)


def get_perm_and_dfs_rewards(
    start_node, 
    adj_list, 
    edge_index,
    nodes,
    agent,
):
    _, r_perm = _get_single_ep_expert_return(
        edge_index,
        nodes,
        agent,
    )
    _, r_dfs = _get_single_ep_dfs_return(
        start_node, 
        adj_list, 
        edge_index,
        nodes,
        agent,
    )

    return r_perm.tolist(), r_dfs.tolist()
