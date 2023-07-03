import numpy as np
import torch
import torch.distributions as dists
from policy import GaussPolicy, Qfunc
from torch import nn


class SACAgentBase:
    def __init__(
        self,
        name,
        policy,
        policy_lr,
        entropy_lb,
        temperature_lr,
        **policy_kwargs
    ):
        self.name = name
        self.policy = policy(**policy_kwargs)
        self.policy_optim = torch.optim.Adam(
            self.policy.parameters(), lr=policy_lr
        )
        self.log_temperature = torch.tensor(0.0, requires_grad=True)
        self.temperature_optim = torch.optim.Adam(
            [self.log_temperature], lr=temperature_lr
        )
        self.entropy_lb = (
            entropy_lb if entropy_lb else -policy_kwargs["action_dim"]
        )
        self.temperature_loss = None
        self.policy_loss = None

    def sample_action(self, obs):
        pass

    def get_policy_loss_and_temperature_loss(self, *args, **kwargs):
        pass

    def update_policy_and_temperature(self, *args, **kwargs):
        pass


class SACAgentMuJoCo(SACAgentBase):
    def __init__(
        self,
        name,
        policy_lr,
        entropy_lb,
        temperature_lr,
        **policy_kwargs
    ):
        """
        policy_kwargs contain arguments for GaussPolicy constructor;
        """
        super(SACAgentMuJoCo, self).__init__(
            name,
            GaussPolicy,
            policy_lr,
            entropy_lb,
            temperature_lr,
            **policy_kwargs
        )

    def sample_action(self, obs):
        mus, sigmas = self.policy(obs)
        policy_dist = dists.Normal(mus, sigmas)  # indep Gauss;
        action = policy_dist.sample()
        return action

    def update_policy_and_temperature(self):
        # update policy params;
        self.policy_optim.zero_grad()
        self.policy_loss.backward()
        self.policy_optim.step()

        # update temperature;
        self.temperature_optim.zero_grad()
        self.temperature_loss.backward()
        self.temperature_optim.step()

    def get_policy_loss_and_temperature_loss(
        self, log_temperature, obs_t, qfunc1, qfunc2
    ):
        # get gauss params;
        mus, sigmas = self.policy(obs_t)

        # do reparam trick;
        repr_trick = torch.randn(mus.shape)  # samples from N(0, I);
        repr_trick = mus + repr_trick * sigmas

        # get Gauss density of policy;
        policy_density = dists.Normal(mus, sigmas)

        # compute entropies of gaussian to be passed
        # to optimisation step for temperature;
        # in the papaer they use .log_prob() but
        # gauss policy has closed form entropy, so I use it;
        with torch.no_grad():
            entropies = policy_density.entropy().sum(-1)

        # get log prob for policy optimisation;
        log_prob = policy_density.log_prob(repr_trick).sum(-1)

        # freeze q-nets and eval at current observation
        # and reparam trick action and choose min for policy loss;
        qfunc1.requires_grad_(False)
        qfunc2.requires_grad_(False)
        qfunc_in = torch.cat((obs_t, repr_trick), -1)
        q_est = torch.min(qfunc1(qfunc_in), qfunc2(qfunc_in))

        # get loss for policy;
        self.policy_loss = (
            np.exp(log_temperature.item()) * log_prob - q_est
        ).mean()

        # get loss for temperature;
        self.temperature_loss = (
            self.log_temperature.exp()
            * (entropies - self.entropy_lb).mean()
        )


def update_params(Q1t, Q1, tau):
    """
    Updates parameters of Q1t to be:
    Q1t = Q1 * tau + Q1t * (1 - tau).
    """
    theta = nn.utils.parameters_to_vector(Q1.parameters()) * tau + (
        1 - tau
    ) * nn.utils.parameters_to_vector(Q1t.parameters())

    # load theta as the new params of Q1;
    nn.utils.vector_to_parameters(theta, Q1t.parameters())


def load_params_in_net(net: nn.Module, parameters: Iterable):
    """
    Loads parameters in the layers of net.
    """
    nn.utils.vector_to_parameters(
        nn.utils.parameters_to_vector(parameters), net.parameters()
    )
