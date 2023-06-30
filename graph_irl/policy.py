import torch
from torch import nn
import torch.distributions as dists
from collections import deque


class GaussPolicy(nn.Module):
    def __init__(
        self, obs_dim, action_dim, hiddens, with_layer_norm=False
    ):
        super(GaussPolicy, self).__init__()
        self.net = nn.Sequential()
        for i in range(len(hiddens)):
            if i == 0:
                self.net.append(nn.Linear(obs_dim, hiddens[i]))
            else:
                self.net.append(nn.Linear(hiddens[i - 1], hiddens[i]))
            self.net.append(nn.ReLU())
            if with_layer_norm:
                self.net.append(nn.LayerNorm(hiddens[i]))
        self.mu_net = nn.Linear(hiddens[-1], action_dim)
        self.std_net = nn.Linear(hiddens[-1], action_dim)

    def forward(self, obs):
        emb = self.net(obs)
        return self.mu_net(emb), torch.log(
            1.0 + self.std_net(emb).exp()
        )


class PGAgentBase:
    def __init__(
        self,
        name,
        obs_dim,
        action_dim,
        policy,
        with_baseline,
        lr,
        discount,
        **kwargs
    ):
        self.rewards = []
        self.name = name
        self.log_probs = []
        self.baseline = 0.0
        self.policy = policy(obs_dim, action_dim, **kwargs)
        self.with_baseline = with_baseline
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.discount = discount

    def update_rewards(self, reward):
        pass

    def update_baseline(self):
        pass

    def update(self):
        pass

    def sample_action(self, obs):
        pass


class PGGauss(PGAgentBase):
    def __init__(
        self,
        name,
        obs_dim,
        action_dim,
        policy,
        with_baseline,
        lr,
        discount,
        **kwargs
    ):
        super(PGGauss, self).__init__(
            name,
            obs_dim,
            action_dim,
            policy,
            with_baseline,
            lr,
            discount,
            **kwargs
        )

    def update_rewards(self, reward):
        self.rewards.append(reward)

    def update_baseline(self):
        if self.with_baseline:
            self.baseline = self.baseline + (
                self.rewards[-1] - self.baseline
            ) / len(self.rewards)

    def sample_action(self, obs):
        mus, sigmas = self.policy(obs)
        policy_dist = dists.Normal(mus, sigmas)
        action = policy_dist.sample()
        self.log_probs.append(policy_dist.log_prob(action).sum())
        return action

    def update(self):
        curr_return = 0.0
        returns_to_go = torch.empty(len(self.rewards))

        for i, r in enumerate(self.rewards[::-1], start=1):
            if self.with_baseline:
                r = r - self.baseline
            curr_return = r + self.discount * curr_return
            returns_to_go[len(self.rewards) - i] = curr_return

        # get weighted loss;
        self.optim.zero_grad()
        loss = -torch.stack(self.log_probs) @ returns_to_go
        loss.backward()
        self.optim.step()

        # reset rewards, log_probs and baseline;
        self.rewards, self.log_probs, self.baseline = [], [], 0.0


class Qfunc(nn.Module):
    def __init__(self, obs_action_dim, hiddens, with_layer_norm=False):
        super(Qfunc, self).__init__()
        self.net = nn.Sequential()
        for i in range(len(hiddens)):
            if i == 0:
                self.net.append(nn.Linear(obs_action_dim, hiddens[i]))
            else:
                self.net.append(nn.Linear(hiddens[i - 1], hiddens[i]))
            self.net.append(nn.ReLU())
            if with_layer_norm:
                self.net.append(nn.LayerNorm(hiddens[i]))
        self.net.append(hiddens[-1], 1)

    def forward(self, obs_action):
        return self.net(obs_action)


def update_params(Q1, Q2, tau):
    """
    Updates parameters of Q1 to be:
    Q1_new = Q2 * tau + Q1_old * (1 - tau).
    """
    theta = nn.utils.parameters_to_vector(Q2.parameters()) * tau + (
        1 - tau
    ) * nn.utils.parameters_to_vector(Q1.parameters())

    # load theta as the new params of Q1;
    nn.utils.vector_to_parameters(theta, Q1.parameters())


# def train_sac(
#     obs_dim,
#     action_dim,
#     qfunc_hiddens,
#     qfunc_lns,
#     policy_hiddens,
#     policy_lns,
#     lr,
# ):
#     # init nets;
#     obs_action_dim = obs_dim + action_dim
#     Q1 = Qfunc(obs_action_dim, qfunc_hiddens, with_layer_norm=qfunc_lns)
#     Q2 = Qfunc(obs_action_dim, qfunc_hiddens, with_layer_norm=qfunc_lns)
#     Q3 = Qfunc(obs_action_dim, qfunc_hiddens, with_layer_norm=qfunc_lns)
#     Q4 = Qfunc(obs_action_dim, qfunc_hiddens, with_layer_norm=qfunc_lns)
#     policy = GaussPolicy(
#         obs_dim, action_dim, policy_hiddens, with_layer_norm=policy_lns
#     )
#     optim_q1 = torch.optim.Adam(Q1.parameters(), lr=lr)


class SACAgentBase:
    def __init__(self, policy, lr, **policy_kwargs):
        self.policy = policy(**policy_kwargs)
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def sample_action(self, obs):
        pass

    def update_policy(self, *args, **kwargs):
        pass


class SACAgentMuJoCo(SACAgentBase):
    def __init__(self, policy, lr, **policy_kwargs):
        super(SACAgentMuJoCo, self).__init__(policy, lr, **policy_kwargs)

    def sample_action(self, obs):
            mus, sigmas = self.policy(obs)
            policy_dist = dists.Normal(mus, sigmas)
            action = policy_dist.sample()
            return action

    def update_policy(self, temperature, obs_t, qfunc, **kwargs):
        self.optim.zero_grad()
        # get gauss params;
        mus, sigmas = self.policy(obs_t)
        # do reparam trick;
        repr_trick = torch.randn(mus.shape)
        repr_trick = mus + repr_trick * sigmas
        # average log prob of gauss policy over minibatch of observations
        # and evaled at reparam trick;
        avg_log_prob = dists.Normal(mus, sigmas).log_prob(repr_trick).sum(-1).mean()
        # freeze q-net;
        qfunc.require_grad_(False)
        # eval q-net at observations in minibatch and reparam trick actions.
        # take avg over batch dim;
        avg_q_vals = qfunc(torch.cat((obs_t, repr_trick), -1)).mean()
        # get loss for policy;
        loss = temperature * avg_log_prob - avg_q_vals
        loss.backward()
        self.optim.step()


def train_sac(num_episodes, seed):
    """
    This is currently some pseudo code for training SAC agent.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    # init 2 q nets;
    # init 2 target qnets with same parameters as q1 and q2;
    # init policy net;
    # init replay boofer of size=N deque?
    undiscounted_returns = []
    boofer = deque([])
    for ep in range(num_episodes):
        sampled_return = 0.
        # sample trajectory;
        done = False
        # sample starting state;
        obs_t, info = env.step()
        # sample first action;
        action_t = agent.sample_action(
            torch.FloatTensor(obs_t).view(-1)
        )
        # boofer = deque([])
        while not done:
            # act with action_t and get state and reward;;
            obs_tp1, reward_t, terminated, truncated, info = env.step(
                action_t.numpy()
            )
            # update return;
            sampled_return += reward_t
            # sample next action;
            action_tp1 = agent.sample_action(obs_tp1)
            # maintain the boofer with lenght N;
            if len(boofer) >= N:
                boofer.popleft()
            boofer.append(
                (obs_t, action_t, reward_t, obs_tp1, action_tp1)
            )
            obs_t, action_t = obs_tp1, action_tp1
            done = terminated or truncated
        # add sampled episodic return to list;
        undiscounted_returns.append(sampled_return)

        # prep boofer to sample minibatches of experience.
        data_loader = get_loader(boofer, batch_size=batch_size)

        # do the gradient updates;
        for i, (obs_t, action_t, reward_t, obs_tp1, action_tp1) in enumerate(data_loader, start=1):
            # value func updates;
            q_est1 = update_q(Q1, Q1t, obs_t, action_t, reward_t, obs_tp1, action_tp1, optimq1)
            q_est2 = update_q(Q2, Q2t, obs_t, action_t, reward_t, obs_tp1, action_tp1, optimq2)
            
            # target q-func updates;
            update_params(Q1t, Q1, tau)
            update_params(Q2t, Q2, tau)

            # update policy params with the arg min (q1, q2) q func;
            if q_est1 < q_est2:
                policy_entropies = agent.update_policy(Q1, obs_t, action_t)
            else:
                policy_entropies = agent.update_policy(Q2, obs_t, action_t)

            # update temperature;
            update_temperature(temperature, policy_entropies, entropy_target)
    # optionally save for this seed from all episodes;
    with open(file_name, 'wb'):
        pickle.dump(undiscounted_returns, f)
    return Q1, Q2, agent, undiscounted_returns

