import numpy as np
import torch


class Buffer:
    def __init__(self, max_size, obs_dim, action_dim):
        self.idx, self.max_size = 0, max_size
        self.obs_t = np.empty((max_size, obs_dim))
        self.action_t = np.empty((max_size, action_dim))
        self.obs_tp1 = np.empty((max_size, obs_dim))
        self.terminal_tp1 = np.empty((max_size,))
        self.reward_t = np.empty((max_size,))
    
    def add_sample(self, obs_t, action_t, reward_t, obs_tp1, terminal_tp1):
        idx = self.idx % self.max_size
        self.obs_t[idx] = obs_t
        self.action_t[idx] = action_t
        self.reward_t[idx] = reward_t
        self.obs_tp1[idx] = obs_tp1
        self.terminal_tp1[idx] = terminal_tp1
        self.idx += 1
    
    def sample(self, batch_size):
        assert batch_size <= self.idx
        idxs = np.random.choice(min(self.idx, self.max_size), size=batch_size, replace=False)
        return (
            torch.tensor(self.obs_t[idxs], dtype=torch.float32),
            torch.tensor(self.action_t[idxs], dtype=torch.float32),
            torch.tensor(self.reward_t[idxs], dtype=torch.float32),
            torch.tensor(self.obs_tp1[idxs], dtype=torch.float32),
            torch.tensor(self.terminal_tp1[idxs], dtype=torch.float32)
        )
    
    def __len__(self):
        return min(self.idx, self.max_size)
    
    def collect_path(self, env, agent, num_steps, returns):
        t = 0
        obs_t, info = env.reset()
        sampled_return = 0.
        while t < num_steps:
            obs_t = torch.tensor(obs_t, dtype=torch.float32)
            action_t = agent.sample_action(obs_t).numpy()
            obs_tp1, reward, terminal, truncated, info = env.step(action_t)
            self.add_sample(obs_t, action_t, reward, obs_tp1, terminal)
            sampled_return += reward
            if terminal or truncated:
                obs_t, info = env.reset()
                returns.append(sampled_return)
                sampled_return = 0.
            else:
                obs_t = obs_tp1
            t += 1


if __name__ == "__main__":
    import gymnasium as gym
    

    class DummyAgent:
        def __init__(self, env):
            self.env = env
        
        def sample_action(self, obs_t):
            return torch.tensor(self.env.action_space.sample(), dtype=torch.float32)

       
    env = gym.make('Hopper-v2')

    # obs_t, info = env.reset(seed=0)
    # action_t = env.action_space.sample()
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_size = 1_000
    # batch_size = 100
    buffer = Buffer(max_size, obs_dim, action_dim)
    returns = []
    agent = DummyAgent(env)
    buffer.collect_path(env, agent, 100, returns)
    # while len(buffer) < 500:
    #     done = False
    #     while not done:
    #         obs_tp1, reward, terminal, truncated, info = env.step(action_t)
    #         buffer.add_sample(obs_t, action_t, reward, obs_tp1, terminal)
    #         done = terminal or truncated
    
    # print(len(buffer), buffer.terminal_tp1.sum())
    # for _ in range(10):
    #     batch = buffer.sample(batch_size)
    #     for item in batch:
    #         print(item.shape, item.dtype)
    #     break


        