import numpy as np
import matplotlib.pyplot as plt


def ci_plot(ax, data: np.ndarray, **kwargs):
    """
    Plot mean +- std intervals along the rows of data.
    """
    x = np.arange(data.shape[-1])
    avg = np.mean(data, 0)
    sd = np.std(data, 0)
    ax.fill_between(x, avg - sd, avg + sd, alpha=0.2, **kwargs)
    ax.plot(x, avg, **kwargs)