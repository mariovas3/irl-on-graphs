from vis_utils import ci_plot
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

TEST_OUTPUTS_PATH = Path('.').absolute().parent / "test_output"
if not TEST_OUTPUTS_PATH.exists():
    TEST_OUTPUTS_PATH.mkdir()

file_name = TEST_OUTPUTS_PATH / 'test_ci_plot.png'


ranges = [0., 5., 6., 24., 28.]
data = [np.random.uniform(ranges[i], ranges[i+1], size=(30, 10)) for i in range(len(ranges)-1)]

fig, ax = plt.subplots(figsize=(8, 6))


def test_ci_plot():
    for i, c in enumerate(zip(['r', 'g', 'b', 'k'], ['-', '--', ':', '-.'])):
        ci_plot(ax, data[i], color=c[0], linestyle=c[1])
    fig.tight_layout()
    plt.savefig(file_name)
