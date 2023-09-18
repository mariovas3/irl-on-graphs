import pickle
import numpy as np
from scipy.stats import ks_2samp
from graph_irl.eval_metrics import get_five_num_summary
from pathlib import Path
import matplotlib.pyplot as plt


def softopt_vs_rand_perm_illustration(save_to: Path):
    # create "optimal control" example
    # and show softoptimal traj alongside it;
    x = np.linspace(0, 2 * np.pi, endpoint=True)
    y = np.sin(x)
    fig, axs = plt.subplots(figsize=(12, 4), nrows=1, ncols=2)
    axs[0].plot(x, y, label="optimal control")
    axs[0].scatter(
        (x[0], x[-1]), (y[0], y[-1]), marker="x", color="red", s=120
    )
    for _ in range(3):
        # add random Gauss noise to optimal control;
        offset = np.random.normal(0, 0.1, size=len(x))
        offset[0] = offset[-1] = 0
        axs[0].plot(x, y + offset, alpha=0.5, linestyle="--")
    axs[0].set_title("Soft-optimal paths alongside optimal path")
    axs[0].legend()

    # create "optimal control" example
    # and show rand permutations of it alongside it;
    axs[1].plot(x, y, label="optimal control")
    axs[1].scatter(
        (x[0], x[-1]), (y[0], y[-1]), marker="x", color="red", s=120
    )
    for _ in range(3):
        # permute optimal control traj;
        p_new = np.random.choice(y[1:-1], size=len(y) - 2, replace=False)
        p_new = [y[0]] + p_new.tolist() + [y[-1]]
        axs[1].plot(x, p_new, alpha=0.5)
    # axs[1].legend()
    axs[1].set_title("Random permutation of optimal control path")
    fig.tight_layout()
    plt.savefig(save_to / "gcl_vs_graphopt.png")
    plt.close()


def get_all_p_vals(metrics, metric_names):
    results = {}
    for n in metric_names:
        # n = 'degrees'
        n1 = "irlpolicy" + "_" + n
        n2 = "newpolicy" + "_" + n
        n3 = "sourcegraph" + "_" + n
        n4 = "targetgraph" + "_" + n
        # kolmogorov-smirnov tests;
        # first compare irlpolicy ran on target vs sourcegraph dist;
        # since irlpolicy was trained on sourcegraph;
        p1 = ks_2samp(metrics[n1], metrics[n3], method="exact").pvalue
        # second, compare newpolicy ran on target vs targetgraph dist;
        # since it was retrained on target, using irl_reward;
        p2 = ks_2samp(metrics[n2], metrics[n4], method="exact").pvalue
        # compare irl policy to target;
        p3 = ks_2samp(metrics[n1], metrics[n4], method="exact").pvalue
        results[n1 + "_KS_pval"] = p1
        results[n2 + "_KS_pval"] = p2
        results[n1 + "_target_KS_pval"] = p3
    return results, {
        k: get_five_num_summary(v) for k, v in metrics.items()
    }
