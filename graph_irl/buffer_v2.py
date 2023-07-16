"""
TODO:
    (1): Implement a Graph Buffer following the 
        description from the README.md file.
"""

import numpy as np
import torch


class Buffer:
    def __init__(self, max_size, obs_dim, action_dim, seed=None):
        self.idx, self.max_size = 0, max_size
        self.obs_t = np.empty((max_size, obs_dim))
        self.action_t = np.empty((max_size, action_dim))
        self.obs_tp1 = np.empty((max_size, obs_dim))
        self.terminal_tp1 = np.empty((max_size,))
        self.reward_t = np.empty((max_size,))
        self.seed = 0 if seed is None else seed
        self.undiscounted_returns = []
        self.path_lens = []
        self.avg_rewards_per_episode = []
        self.looped = False

    def add_sample(self, obs_t, action_t, reward_t, obs_tp1, terminal_tp1):
        idx = self.idx % self.max_size
        if not self.looped and self.idx and not idx:
            self.looped = True
        self.idx = idx
        self.obs_t[idx] = obs_t
        self.action_t[idx] = action_t
        self.reward_t[idx] = reward_t
        self.obs_tp1[idx] = obs_tp1
        self.terminal_tp1[idx] = terminal_tp1
        self.idx += 1

    def sample(self, batch_size):
        assert batch_size <= self.__len__()
        idxs = np.random.choice(
            self.__len__(), size=batch_size, replace=False
        )
        return (
            torch.tensor(self.obs_t[idxs], dtype=torch.float32),
            torch.tensor(self.action_t[idxs], dtype=torch.float32),
            torch.tensor(self.reward_t[idxs], dtype=torch.float32),
            torch.tensor(self.obs_tp1[idxs], dtype=torch.float32),
            torch.tensor(self.terminal_tp1[idxs], dtype=torch.float32),
        )

    def __len__(self):
        return self.max_size if self.looped else self.idx

    def collect_path(
        self, env, agent, num_steps,
    ):
        """
        Collect steps from MDP induced by env and agent.

        Args:
            env: Supports similar api to gymnasium.Env..
            agent: Supports sample_action(obs) api.
            num_steps: Number of (obs, action, reward, next_obs, terminated)
                tuples to be added to the buffer.
            returns: List where sampled returns are added.
            avg_the_returns: Record avg reward per episode.
        """
        num_steps = min(num_steps, self.max_size)
        t = 0
        obs_t, info = env.reset(seed=self.seed)
        self.seed += 1
        avg_reward, num_rewards = 0.0, 0.0
        undiscounted_return = 0.
        while t < num_steps:
            obs_t = torch.tensor(obs_t, dtype=torch.float32)

            # sample action;
            action_t = agent.sample_action(obs_t).numpy()

            # sample dynamics;
            obs_tp1, reward, terminal, truncated, info = env.step(action_t)

            # add sampled tuple to buffer;
            self.add_sample(obs_t, action_t, reward, obs_tp1, terminal)

            # house keeping for observed rewards.
            num_rewards += 1
            
            avg_reward = (
                avg_reward + (reward - avg_reward) / num_rewards
            )
            undiscounted_return += reward

            # restart env if episode ended;
            if terminal or truncated:
                obs_t, info = env.reset(seed=self.seed)
                self.seed += 1
                self.undiscounted_returns.append(undiscounted_return)
                self.avg_rewards_per_episode.append(avg_reward)
                self.path_lens.append(num_rewards)
                avg_reward = 0.0
                num_rewards = 0.0
                undiscounted_return = 0.
            else:
                obs_t = obs_tp1
            t += 1


def sample_eval_path(T, env, agent, seed):
    observations, actions, rewards = [], [], []
    obs, info = env.reset(seed=seed)
    observations.append(obs)
    for _ in range(T):
        obs = torch.tensor(obs, dtype=torch.float32)
        action = agent.sample_deterministic(obs).numpy()
        new_obs, reward, terminated, truncated, info = env.step(action)
        actions.append(action)
        observations.append(new_obs)
        rewards.append(reward)
        obs = new_obs
        if terminated and not truncated:
            return observations, actions, rewards, 0
        if terminated and truncated:
            return observations, actions, rewards, 1
        if truncated:
            return observations, actions, rewards, 2
    return observations, actions, rewards, 2


if __name__ == "__main__":
    import gymnasium as gym

    class DummyAgent:
        def __init__(self, env):
            self.env = env

        def sample_action(self, obs_t):
            return torch.tensor(
                self.env.action_space.sample(), dtype=torch.float32
            )

    env = gym.make("Hopper-v2", max_episode_steps=300)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_size = 1_000
    batch_size = 100
    buffer = Buffer(max_size, obs_dim, action_dim)
    returns = []
    agent = DummyAgent(env)
    buffer.collect_path(env, agent, 1200)
    print(len(buffer))
    for _ in range(5):
        batch = buffer.sample(batch_size)
