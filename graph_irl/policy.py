import torch
from torch import nn
import torch.distributions as dists


class GaussPolicy(nn.Module):
    def __init__(self, in_dim, out_dim, hiddens, with_layer_norm=False):
        super(GaussPolicy, self).__init__()
        self.net = nn.Sequential()
        for i in range(len(hiddens)):
            if i == 0:
                self.net.append(nn.Linear(in_dim, hiddens[i]))
            else:
                self.net.append(nn.Linear(hiddens[i-1], hiddens[i]))
            self.net.append(nn.ReLU())
            if with_layer_norm:
                self.net.append(nn.LayerNorm(hiddens[i]))
        self.mu_net = nn.Linear(hiddens[-1], out_dim)
        self.std_net = nn.Linear(hiddens[-1], out_dim)

    def forward(self, obs):
        emb = self.net(obs)
        return self.mu_net(emb), torch.log(1. + self.std_net(emb).exp())


class PGAgentBase:
    def __init__(self, name, obs_dim, action_dim, policy, with_baseline, lr, discount, **kwargs):
        self.rewards = []
        self.name = name
        self.log_probs = []
        self.baseline = 0.
        self.policy = policy(obs_dim, action_dim, **kwargs)
        self.with_baseline = with_baseline
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.discount = discount
    
    def update_rewards(self, reward):
        pass

    def update_baseline(self, reward):
        pass

    def update(self):
        pass

    def sample_action(self, obs):
        pass


class PGGauss(PGAgentBase):
    def __init__(self, name, obs_dim, action_dim, policy, with_baseline, lr, discount, **kwargs):
        super(PGGauss, self).__init__(name, obs_dim, action_dim, policy, with_baseline, lr, discount, **kwargs)
    
    def update_rewards(self, reward):
        self.rewards.append(reward)

    def update_baseline(self):
        if self.with_baseline:
            self.baseline = self.baseline + (self.rewards[-1] - self.baseline) / len(self.rewards)
    
    def sample_action(self, obs):
        mus, sigmas = self.policy(obs)
        policy_dist = dists.Normal(mus, sigmas)
        action = policy_dist.sample()
        self.log_probs.append(policy_dist.log_prob(action))
        return action
    
    def update(self):
        curr_return = 0.
        returns_to_go = torch.empty(len(self.rewards))

        for i, r in enumerate(self.rewards[::-1], start=1):
            if self.with_baseline:
                r = r - self.baseline
            curr_return = r + self.discount * curr_return
            returns_to_go[len(self.rewards) - i] = curr_return
        
        # get weighted loss;
        self.optim.zero_grad()
        loss = - torch.cat(self.log_probs) @ returns_to_go
        loss.backward()
        self.optim.step()

        # reset rewards, log_probs and baseline;
        self.rewards, self.log_probs, self.baseline = [], [], 0.
