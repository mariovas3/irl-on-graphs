import sys
from pathlib import Path
p = Path(__file__).absolute().parent.parent
if str(p) not in sys.path:
    sys.path.append(str(p))
from scipy.stats import ks_2samp
import pickle
import numpy as np
from heapq import heapify, heappop, heappush
import shutil


def process_topo_dir(topo_dir):
    metrics = {}
    metric_names = set()
    policy_names = set()
    flag = False
    for fi in topo_dir.iterdir():
        file_name = str(fi).split('/')[-1]
        # check if run in this experiment finished successfully;
        # if yes, there should be png file;
        if not flag and '.png' in file_name:
            flag = True
        if '.pkl' in file_name:
            tokens = file_name[:-4].split('_')
            # e.g., 'degrees' or 'triangles'
            metric_names.add(tokens[-1])
            # e.g., 'irlpolicy' or 'targetgraph'
            policy_names.add(tokens[0])
            # e.g., 'irlpolicy_degrees'
            metric_name = tokens[0] + '_' + tokens[-1]
            # read the data - should be list;
            with open(fi, 'rb') as f:
                data = pickle.load(f)
            # some metrics are over multiple runs;
            # e.g., 'irlpolicy_0_degrees.pkl'
            if metric_name in metrics:
                metrics[metric_name].extend(data)
            else:
                metrics[metric_name] = data
    return metrics, metric_names, policy_names, flag


def get_min_p_vals(metrics, metric_names):
    min_p1, min_p2 = 2, 2
    for n in metric_names:
        # n = 'degrees'
        n1 = 'irlpolicy' + '_' + n
        n2 = 'newpolicy' + '_' + n
        n3 = 'sourcegraph' + '_' + n
        n4 = 'targetgraph' + '_' + n
        # kolmogorov-smirnov tests;
        # first compare irlpolicy ran on target vs sourcegraph dist;
        # since irlpolicy was trained on sourcegraph;
        p1 = ks_2samp(metrics[n1], metrics[n3], method='exact').pvalue
        # second, compare newpolicy ran on target vs targetgraph dist;
        # since it was retrained on target, using irl_reward;
        p2 = ks_2samp(metrics[n2], metrics[n4], method='exact').pvalue
        # keep track of least pvalues, i.e., where the
        # dists matched the least;
        min_p1, min_p2 = min(min_p1, p1), min(min_p2, p2)
    return min_p1, min_p2


def select_good_candidates(experiments_dir, k_best):
    # maintain heap with experiments where
    # the worst performing experiment had the greatest
    # pvalue across experiments;
    source_priority = []
    target_priority = []
    heapify(source_priority)
    heapify(target_priority)
    for d in experiments_dir.iterdir():
        topo_dir = d / 'target_graph_stats'
        metrics, metric_names, policy_names, flag = process_topo_dir(topo_dir)
        if not flag:
            continue
        min_p1, min_p2 = get_min_p_vals(metrics, metric_names)
        # add results to min heap;
        heappush(source_priority, (min_p1, min_p2, d))
        heappush(target_priority, (min_p2, min_p1, d))
        if len(source_priority) > k_best:
            assert len(source_priority) == len(target_priority)
            heappop(source_priority)
            heappop(target_priority)
    return source_priority, target_priority


def get_agreement(h1, h2):
    sh1 = set([it[-1] for it in h1])
    return [it for it in h2 if it[-1] in sh1]


def copy_selected_dirs(heap, dest_dir, target_nums=False):
    for it in h:
        exp_dir_name = str(it[-1]).split('/')[-1]
        exp_dir_name = f"{exp_dir_name}-{it[0]:.4f}-{it[1]:.4f}"
        if target_nums:
            exp_dir_name = exp_dir_name + '-targetnums'
        shutil.copytree(it[-1], Path(dest_dir) / exp_dir_name)


if __name__ == "__main__":
    rel_path = input("provide relative path from project root: ")
    experiments_path = p / rel_path
    target_nums = True
    dest_dir = input("provide a path to save experiments: ")
    dest_dir = p / dest_dir

    source_priority, target_priority = select_good_candidates(experiments_path, k_best=10)
    if target_nums:
        h = get_agreement(source_priority, target_priority)
    else:
        h = get_agreement(target_priority, source_priority)
    # copy consensus paths;
    copy_selected_dirs(h, dest_dir, target_nums)

