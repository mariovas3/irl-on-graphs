from vis_utils import *
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import gymnasium as gym

TEST_OUTPUTS_PATH = Path('.').absolute().parent / "test_output"
if not TEST_OUTPUTS_PATH.exists():
    TEST_OUTPUTS_PATH.mkdir()

file_name = TEST_OUTPUTS_PATH / 'test_ci_plot.png'


class PlanningAgent:
    """Agent whose policy is an open-loop plan."""
    def __init__(self, actions):
        self.actions = actions
        self.idx = 0

    def sample_action(self, obs):
        self.idx = (self.idx + 1) % len(self.actions)
        return self.actions[self.idx-1]


# setup gym env to run actions in;
env = gym.make('LunarLander-v2', render_mode='human')
actions = np.random.randint(0, 2, size=(1000, ))
agent = PlanningAgent(actions)


# ci_plot setup;
ranges = [0., 5., 6., 24., 28.]
data = [np.random.uniform(ranges[i], ranges[i+1], size=(30, 10)) 
        for i in range(len(ranges)-1)]
fig, ax = plt.subplots(figsize=(8, 6))


def test_ci_plot():
    for i, c in enumerate(zip(['r', 'g', 'b', 'k'], ['-', '--', ':', '-.'])):
        ci_plot(ax, data[i], color=c[0], linestyle=c[1])
    fig.tight_layout()
    plt.savefig(file_name)


def test_visualise_episode():
    visualise_episode(env, agent, 0, 100)

