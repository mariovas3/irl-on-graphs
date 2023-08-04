"""
    TODO:
        (1): Deprecate sample_eval_paths_graph since haven't implemented it 
                for case when drop self loop or repeated edge. It is recommended
                to use get_single_ep_rewards_and_weights.
"""

import numpy as np
import torch
from torch_geometric.data import Data, Batch

from graph_irl.graph_rl_utils import get_action_vector_from_idx

import warnings


class BufferBase:
    def __init__(self, max_size, seed=None):
        self.seed = 0 if seed is None else seed
        self.idx, self.max_size = 0, max_size
        self.reward_idx = 0
        self.looped = False

        # log variables;
        self.undiscounted_returns = []
        self.path_lens = []
        self.avg_rewards_per_episode = []

    def add_sample_without_reward(self, *args, **kwargs):
        pass

    def add_rewards(self, rewards, **kwargs):
        pass

    def add_sample(self, *args, **kwargs):
        pass

    def sample(self, batch_size):
        pass

    def __len__(self):
        return self.max_size if self.looped else self.idx

    def collect_path(self, env, agent, num_steps_to_collect):
        pass

    def clear_buffer(self):
        """Reset tracked logs and idx in buffer."""
        self.idx = 0
        self.reward_idx = 0
        self.looped = False

        self.undiscounted_returns = []
        self.path_lens = []
        self.avg_rewards_per_episode = []


class GraphBuffer(BufferBase):
    def __init__(
        self,
        max_size,
        nodes: torch.Tensor,
        seed=None,
        drop_repeats_or_self_loops=False,
        get_batch_reward=False,
        graphs_per_batch=None,
        action_is_index=True,
        per_decision_imp_sample=False,
        reward_scale=1.,
        log_offset=0.,
        verbose=False,
    ):
        super(GraphBuffer, self).__init__(max_size, seed)
        
        # see if should print stuff;
        self.verbose = verbose

        # place holder edge index;
        edge_index = torch.tensor([[], []], dtype=torch.long)
        
        # containers to store mdp info;
        self.obs_t = [
            Data(x=nodes, edge_index=edge_index) for _ in range(self.max_size)
        ]
        self.action_t = np.empty((max_size, 2), dtype=np.int64)
        self.obs_tp1 = [
            Data(x=nodes, edge_index=edge_index) for _ in range(self.max_size)
        ]
        self.terminal_tp1 = np.empty((max_size,), dtype=np.int8)
        self.reward_t = np.empty((max_size,))

        # keep nodes reference;
        self.nodes = nodes
        
        # log_offset is subtracted in
        # reward - log_prob_policy - log_offset
        # the reward is always in (-inf, 0)
        # as the reward and policy both become better, the 
        # reward will give high reward for sampled traj 
        # so reward -> 0, however log_prob_policy is generally 
        # negative and depends on dim of action vector; 
        # at that point our importance weights (reward - log_prob).exp()
        # might become too large so an offset might help;
        self.log_offset = log_offset
        
        # similar purpose as log offset, although has slightly different
        # interpretation as amplifying reward signal;
        self.reward_scale = reward_scale

        # extra config for reward computation;
        self.drop_repeats_or_self_loops = drop_repeats_or_self_loops
        self.get_batch_reward = get_batch_reward
        self.graphs_per_batch = graphs_per_batch
        self.action_is_index = action_is_index
        self.per_decision_imp_sample = per_decision_imp_sample
        
        # some proper init rules;
        if get_batch_reward:
            if graphs_per_batch is None:
                raise ValueError(
                    "get_batch_reward is True but graphs_per_batch not specified"
                )
            if self.graphs_per_batch > self.max_size:
                raise ValueError(
                    "graphs_per_batch expected to be less"
                    " than or equal to size of buffer"
                )

    def clear_buffer(self):
        super().clear_buffer()
        edge_index = torch.tensor([[], []], dtype=torch.long)

        # reset buffer;
        self.obs_t = [
            Data(x=self.nodes, edge_index=edge_index)
            for _ in range(self.max_size)
        ]
        self.action_t = np.empty((self.max_size, 2), dtype=np.int64)
        self.obs_tp1 = [
            Data(x=self.nodes, edge_index=edge_index)
            for _ in range(self.max_size)
        ]
        self.terminal_tp1 = np.empty((self.max_size,), dtype=np.int8)
        self.reward_t = np.empty((self.max_size,))

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

    def add_sample_without_reward(self, obs_t, action_t, obs_tp1, terminal_tp1):
        idx = self.idx % self.max_size
        if not self.looped and self.idx and not idx:
            self.looped = True
        self.idx = idx
        self.obs_t[idx] = obs_t
        self.action_t[idx] = action_t
        self.obs_tp1[idx] = obs_tp1
        self.terminal_tp1[idx] = terminal_tp1
        self.idx += 1

    def add_rewards(self, rewards):
        """
        Add 1-D numpy array of rewards to the buffer.
        """
        reward_idx = self.reward_idx % self.max_size
        idxs = (reward_idx + np.arange(len(rewards))) % self.max_size
        self.reward_t[idxs] = rewards
        self.reward_idx = idxs[-1] + 1

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.graphs_per_batch
        assert batch_size <= self.__len__()

        # always sample without replacement;
        idxs = np.random.choice(self.__len__(), size=batch_size, replace=False)
        return (
            Batch.from_data_list([self.obs_t[idx] for idx in idxs]),
            torch.from_numpy(self.action_t[idxs]),  # (B, 2) shape tensor
            torch.tensor(self.reward_t[idxs], dtype=torch.float32),
            Batch.from_data_list([self.obs_tp1[idx] for idx in idxs]),
            torch.tensor(self.terminal_tp1[idxs], dtype=torch.float32),
        )

    def _process_rewards_weights(self, rewards, weights, code, verbose=False):
        rewards = torch.cat(rewards, -1)
        weights = torch.cat(weights, -1)
        if verbose:
            print(weights)
            print(weights.shape, torch.cumprod(weights, 0), end='\n\n')
        return (
            rewards,
            torch.cumprod(weights, 0),
            code,
            len(rewards)
        )

    def get_single_ep_rewards_and_weights(self, env, agent, 
                                          lcr_reg=False, 
                                          verbose=False,
                                          unnorm_policy=False,
                                          be_deterministic=False):
        # init lcr regularisation term and last two rewards;
        lcr_reg_term = 0.
        r1, r2 = None, None
        
        if self.per_decision_imp_sample:
            weights, rewards = [], []
        else:
            imp_weight, return_val = 1.0, 0
        obs, info = env.reset(seed=self.seed)
        agent.policy.eval()

        # eval policy and reward fn in batch mode;
        batch_list, action_idxs = [], []
        if verbose:
            env.reward_fn.verbose()

        terminated, truncated = False, False

        # this steps variable is for debugging
        steps = 0

        # process the episode;
        while not (terminated or truncated):
            
            # sample stochastic actions;
            if be_deterministic:
                (a1, a2), node_embeds = agent.sample_deterministic(obs)
            else:
                (a1, a2), node_embeds = agent.sample_action(obs)

            # make env step;
            new_obs, reward, terminated, truncated, info = env.step(
                ((a1.numpy(), a2.numpy()), node_embeds.detach().numpy())
            )

            # resample action if repeated edge or self loop;
            if self.drop_repeats_or_self_loops and not (terminated or truncated):
                if info["self_loop"] or info["is_repeat"]:
                    continue
            
            skip_sample = (
                self.drop_repeats_or_self_loops 
                and (
                    info['max_repeats_reached'] 
                    or info['max_self_loops_reached']
                )
                # also check if current is self loop or repeated edge;
                # since if max repeats or max self loops are true 
                # but min_steps_to_do is not reached in env
                # we may discard good edges;
                and (
                    info['self_loop']
                    or info['is_repeat']
                )
            )

            steps -= int(skip_sample)
            
            # get idxs of nodes to connect;
            first, second = info["first"], info["second"]
            
            # add action to action list and observation in batch_list;
            if not skip_sample:
                if self.action_is_index:
                    action_idxs.append(
                        torch.tensor([first, second], dtype=torch.long)
                    )
                else:
                    assert a1.ndim == 1
                    action_idxs.append(torch.cat((a1, a2), -1))
                batch_list.append(obs)

            # if max batch length reached, compute rewards;
            if (
                len(batch_list) and (
                    len(batch_list) == self.graphs_per_batch
                    or terminated
                    or truncated
                )
            ):
                # make batch of graphs;
                batch = Batch.from_data_list(batch_list)
                action_idxs = torch.stack(action_idxs)

                # calculate rewards on batch;
                curr_rewards = env.reward_fn(
                    (batch, action_idxs), action_is_index=self.action_is_index
                ).view(-1) * self.reward_scale

                # see if lcr_reg should be calculated;
                # a single lcr reg term requires rewards from 3 time steps;
                if lcr_reg:
                    if r1 is not None and r2 is not None:
                        temp = torch.cat((
                            torch.stack((r1, r2)), curr_rewards
                        ))
                        lcr_reg_term += ((temp[2:] + temp[:-2] - 2 * temp[1:-1]) ** 2).sum()
                    elif len(curr_rewards) > 2:
                        lcr_reg_term += ((curr_rewards[2:] + curr_rewards[:-2] - 2 * curr_rewards[1:-1]) ** 2).sum()
                    if len(curr_rewards) > 1:
                        r1, r2 = curr_rewards[-2], curr_rewards[-1]

                # calculate probs;
                policy_dists, node_embeds = agent.policy(batch)

                # get action vectors from indexes by taking the node embeds
                # for the relevant indexes;
                if self.action_is_index:
                    action_idxs = get_action_vector_from_idx(
                        node_embeds, action_idxs, batch.num_graphs
                    )

                _, D = node_embeds.shape
                assert action_idxs.shape[-1] == 2 * D

                # get log_probs;
                if unnorm_policy:
                    log_probs = policy_dists.get_unnorm_log_prob(
                        action_idxs[:, :D], action_idxs[:, D:]
                    ).sum(-1)
                else:
                    log_probs = policy_dists.log_prob(
                        action_idxs[:, :D], action_idxs[:, D:]
                    ).sum(-1)
                assert log_probs.shape == curr_rewards.shape
                
                # see if should print stuff;
                if verbose:
                    print(f"max reward from within buffer sampling: {curr_rewards.max().item()}")
                    print(f"min log_prob: {log_probs.min().item()}")

                # rewards and weights
                if self.per_decision_imp_sample:
                    rewards.append(curr_rewards)
                    weights.append((curr_rewards - log_probs - self.log_offset).exp().detach())
                else:
                    imp_weight *= (
                        (curr_rewards - log_probs - self.log_offset).sum().exp().detach()
                    )
                    return_val += curr_rewards.sum()
                    if verbose:
                        print(f"step: {steps} of sampling, vanilla weight: {imp_weight}",
                            f"vanilla return: {return_val}\n"
                            "curr rewards from sampling\n", 
                            curr_rewards.tolist(), end='\n\n',)

                # reset batch list and action idxs list;
                batch_list, action_idxs = [], []

            # update current state;
            obs = new_obs
            steps += 1
            code = -1
            if terminated and not truncated:
                code = 0
            elif terminated and truncated:
                code = 1
            elif truncated:
                code = 2
            if code != -1:
                assert steps == env.steps_done
				# print(steps, env.steps_done, info['max_self_loops_reached'])
                if self.per_decision_imp_sample:
                    return self._process_rewards_weights(rewards, weights, code, verbose) + (lcr_reg_term, obs)
                return return_val, imp_weight, code, env.steps_done, lcr_reg_term, obs

        assert steps == env.steps_done
		# print(steps, env.steps_done, info['max_self_loops_reached'])
        if self.per_decision_imp_sample:
            # in the per dicision case, need to cumprod the weights
            # until the current time point;
            return self._process_rewards_weights(rewards, weights, 2, verbose) + (lcr_reg_term, obs)
        return return_val, imp_weight, 2, env.steps_done, lcr_reg_term, obs

    def collect_path(
        self,
        env,
        agent,
        num_steps_to_collect,
    ):
        """
        Collect steps from MDP induced by env and agent.

        Args:
            env: Supports similar api to gymnasium.Env..
            agent: Supports sample_action(obs) api.
            num_steps_to_collect: Number of (obs, action, reward, next_obs, terminated)
                tuples to be added to the buffer.
        """
        agent.policy.eval()
        num_steps_to_collect = min(num_steps_to_collect, self.max_size)
        t = 0
        obs_t, info = env.reset(seed=self.seed)
        self.seed += 1
        avg_reward, num_rewards = 0.0, 0.0
        undiscounted_return = 0.0

        # see if should do batch eval of reward;
        # this should be more efficient if deep reward net;
        if self.get_batch_reward:
            batch_list = []
            action_idxs = []

        while t < num_steps_to_collect or self.idx != self.reward_idx:
            # sample action;
            (a1, a2), node_embeds = agent.sample_action(obs_t)

            # sample dynamics;
            obs_tp1, reward, terminal, truncated, info = env.step(
                ((a1.numpy(), a2.numpy()), node_embeds.detach().numpy())
            )

            # resample action if repeated edge or self loop;
            if self.drop_repeats_or_self_loops and not (terminal or truncated):
                if info["self_loop"] or info["is_repeat"]:
                    continue
            
            skip_sample = (
                self.drop_repeats_or_self_loops 
                and (
                    info['max_repeats_reached'] 
                    or info['max_self_loops_reached']
                )
                and (
                    info['self_loop']
                    or info['is_repeat']
                )
            )

            # see if this step should be skipped;
            t -= int(skip_sample)

            # indexes of nodes to be connected;
            first, second = info["first"], info["second"]
            action_t = np.array([first, second])

            if self.get_batch_reward:
                if not skip_sample:
                    self.add_sample_without_reward(
                        obs_t, action_t, obs_tp1, terminal
                    )
                    batch_list.append(obs_t)
                    action_idxs.append([first, second])
                if (
                    len(batch_list) and (
                        len(batch_list) == self.graphs_per_batch
                        or terminal
                        or truncated
                        # or (t == num_steps_to_collect - 1)
                    )
                ):
                    # make batch of graphs;
                    batch = Batch.from_data_list(batch_list)

                    # calculate rewards on batch;
                    rewards = env.reward_fn(
                        (batch, torch.tensor(action_idxs, dtype=torch.long)),
                        action_is_index=True,
                    ).view(-1) * self.reward_scale

                    # add rewards to buffer;
                    rewards = rewards.detach().numpy()
                    self.add_rewards(rewards)

                    # reset batch list and action list;
                    batch_list = []
                    action_idxs = []

                    # update housekeeping;
                    avg_reward = avg_reward * (
                        num_rewards / (num_rewards + len(rewards))
                    ) + rewards.sum() / (num_rewards + len(rewards))
                    undiscounted_return += rewards.sum()
                    num_rewards += len(rewards)
            else:
                # add sampled tuple to buffer;
                if not skip_sample:
                    self.add_sample(obs_t, action_t, reward, obs_tp1, terminal)

                    # housekeeping for observed rewards.
                    num_rewards += 1
                    avg_reward = avg_reward + (reward - avg_reward) / num_rewards
                    undiscounted_return += reward

            # restart env if episode ended;
            if terminal or truncated:
                # print("repeats or self loops: "
                #       f"{info['max_repeats_reached']}, "
                #       f"{info['max_self_loops_reached']}")
                obs_t, info = env.reset(seed=self.seed)
                self.seed += 1
                self.undiscounted_returns.append(undiscounted_return)
                self.avg_rewards_per_episode.append(avg_reward)
                self.path_lens.append(num_rewards)
                avg_reward = 0.0
                num_rewards = 0.0
                undiscounted_return = 0.0
            else:
                obs_t = obs_tp1
            t += 1
        if self.get_batch_reward:
            assert self.idx == self.reward_idx


class Buffer(BufferBase):
    def __init__(self, max_size, obs_dim, action_dim, seed=None):
        super(Buffer, self).__init__(max_size, seed)
        self.obs_t = np.empty((max_size, obs_dim))
        self.action_t = np.empty((max_size, action_dim))
        self.obs_tp1 = np.empty((max_size, obs_dim))
        self.terminal_tp1 = np.empty((max_size,))
        self.reward_t = np.empty((max_size,))
        self.obs_dim, self.action_dim = obs_dim, action_dim

    def clear_buffer(self):
        super().clear_buffer()
        self.obs_t = np.empty((self.max_size, self.obs_dim))
        self.action_t = np.empty((self.max_size, self.action_dim))
        self.obs_tp1 = np.empty((self.max_size, self.obs_dim))
        self.terminal_tp1 = np.empty((self.max_size,))
        self.reward_t = np.empty((self.max_size,))

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
        idxs = np.random.choice(self.__len__(), size=batch_size, replace=False)
        return (
            torch.tensor(self.obs_t[idxs], dtype=torch.float32),
            torch.tensor(self.action_t[idxs], dtype=torch.float32),
            torch.tensor(self.reward_t[idxs], dtype=torch.float32),
            torch.tensor(self.obs_tp1[idxs], dtype=torch.float32),
            torch.tensor(self.terminal_tp1[idxs], dtype=torch.float32),
        )

    def collect_path(
        self,
        env,
        agent,
        num_steps_to_collect,
    ):
        """
        Collect steps from MDP induced by env and agent.

        Args:
            env: Supports similar api to gymnasium.Env..
            agent: Supports sample_action(obs) api.
            num_steps_to_collect: Number of (obs, action, reward, next_obs, terminated)
                tuples to be added to the buffer.
        """
        agent.policy.eval()
        num_steps_to_collect = min(num_steps_to_collect, self.max_size)
        t = 0
        obs_t, info = env.reset(seed=self.seed)
        self.seed += 1
        avg_reward, num_rewards = 0.0, 0.0
        undiscounted_return = 0.0
        while t < num_steps_to_collect:
            obs_t = torch.tensor(obs_t, dtype=torch.float32)

            # sample action;
            action_t = agent.sample_action(obs_t).numpy()

            # sample dynamics;
            obs_tp1, reward, terminal, truncated, info = env.step(action_t)

            # add sampled tuple to buffer;
            self.add_sample(obs_t, action_t, reward, obs_tp1, terminal)

            # house keeping for observed rewards.
            num_rewards += 1

            avg_reward = avg_reward + (reward - avg_reward) / num_rewards
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
                undiscounted_return = 0.0
            else:
                obs_t = obs_tp1
            t += 1


def sample_eval_path_graph(T, env, agent, seed, verbose=False):
    warnings.warn("sample_eval_path_graph is deprecated in favour of "
                  "the graph_buffer's get_ep_rewards_and_weights method")
    old_calculate_reward = env.calculate_reward
    env.reward_fn.eval()
    env.calculate_reward = True
    observations, actions, rewards = [], [], []
    obs, info = env.reset(seed=seed)
    agent.policy.eval()
    if verbose:
        env.reward_fn.verbose()
    observations.append(obs)
    for _ in range(T):
        (mus1, mus2), node_embeds = agent.sample_deterministic(obs)
        new_obs, reward, terminated, truncated, info = env.step(
            ((mus1.numpy(), mus2.numpy()), node_embeds.detach().numpy())
        )
        a = np.array([info["first"], info["second"]], dtype=np.int64)
        actions.append(a)
        observations.append(new_obs)
        if isinstance(reward, torch.Tensor):
            reward = reward.detach().item()
        rewards.append(reward)
        obs = new_obs
        if terminated and not truncated:
            env.calculate_reward = old_calculate_reward
            return observations, actions, rewards, 0
        if terminated and truncated:
            env.calculate_reward = old_calculate_reward
            return observations, actions, rewards, 1
        if truncated:
            env.calculate_reward = old_calculate_reward
            return observations, actions, rewards, 2
    env.calculate_reward = old_calculate_reward
    # env.min_steps_to_do = old_min_steps_to_do
    env.reward_fn.train()
    return observations, actions, rewards, 2


def sample_eval_path(T, env, agent, seed):
    agent.policy.eval()
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
