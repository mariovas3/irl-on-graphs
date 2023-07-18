import torch
from torch import nn
import torch.distributions as dists


class GaussInputDist:
    def __init__(self, diag_gauss):
        self.diag_gauss = diag_gauss

    def log_prob(self, x):
        pass

    def sample(self):
        pass

    def rsample(self):
        pass

    @property
    def mean(self):
        return self.diag_gauss.mean

    @property
    def stddev(self):
        return self.diag_gauss.stddev

    def get_UT_trick_input(self, offset=1.0):
        mus = self.diag_gauss.mean
        sigmas = self.diag_gauss.stddev
        B, D = mus.shape

        # make to (B, D, D) shape;
        diags = sigmas.unsqueeze(1) * torch.eye(D)

        # make to (B, 2D + 1, D) shape;
        diags = torch.cat((diags, torch.zeros((B, 1, D)), -diags), 1)

        # return (B, 2D + 1, D) shape;
        return mus.unsqueeze(1) + offset * diags

    def log_prob_UT_trick(self):
        pass


class GaussDist(GaussInputDist):
    def __init__(self, diag_gauss, two_action_vectors=False):
        super(GaussDist, self).__init__(diag_gauss)
        self.two_action_vectors = two_action_vectors

    def log_prob(self, x):
        return self.diag_gauss.log_prob(x)

    def sample(self):
        a = self.diag_gauss.sample()
        mid = a.shape[-1] // 2
        if self.two_action_vectors:
            return a[:, :mid].squeeze(), a[:, mid:].squeeze()
        return a

    def rsample(self):
        a = self.diag_gauss.rsample()
        mid = a.shape[-1] // 2
        if self.two_action_vectors:
            return a[:, :mid].squeeze(), a[:, mid:].squeeze()
        return a

    def log_prob_UT_trick(self):
        f_in = self.get_UT_trick_input().permute((1, 0, 2))
        return self.log_prob(f_in).mean(0)  # (B, D)

    def entropy(self):
        return self.diag_gauss.entropy()


class TwoStageGaussDist(GaussInputDist):
    def __init__(self, diag_gauss, obs, net):
        super(TwoStageGaussDist, self).__init__(diag_gauss)
        self.obs = obs
        self.net = net

    def log_prob(self, a1, a2):
        diag_gauss2 = self._get_a2_dist(a1)
        return self.diag_gauss.log_prob(a1) + diag_gauss2.log_prob(a2)

    def sample(self):
        a1 = self.diag_gauss.sample()
        diag_gauss2 = self._get_a2_dist(a1)
        return a1.squeeze(), diag_gauss2.sample().squeeze()

    def rsample(self):
        a1 = self.diag_gauss.rsample()
        diag_gauss2 = self._get_a2_dist(a1)
        return a1.squeeze(), diag_gauss2.rsample().squeeze()

    def _get_a2_dist(self, a1):
        mus, sigmas = self.net(torch.cat((self.obs, a1), -1))
        return dists.Normal(mus, sigmas)

    def _get_a2_dist_from_obs_a1(self, obs_a1):
        mus, sigmas = self.net(obs_a1)
        return dists.Normal(mus, sigmas)

    def entropy(self):
        """
        Need to calculate:
            H[p(a1|s)] + E[H[p(a2|s, a1)]]_p(a1|s)
        Since p(a2|s, a1) = N(a2| mu(s, a1), sigma(s, a1)),
        Calculating H[p(a2|s, a1)] is a function of a1 and is
        easy to calculate as Gaussian entropy. However, Calculating
        expectation of H[p(a2|s, a1)] wrt p(a1|s) is intractable
        and would require some numerical trick - e.g., UT.
        """
        h1 = self.diag_gauss.entropy()
        # (B, 2D + 1, D);
        samples = super().get_UT_trick_input()
        f = lambda x: self._get_a2_dist_from_obs_a1(x).entropy()
        h2_UT = batch_UT_trick_from_samples(f, self.obs, samples)
        # returns (B, D) shape;
        return h1 + h2_UT

    def log_prob_UT_trick(self):
        raise NotImplementedError(
            "(2D + 1) ^ 2 samples needed" " -> too much compute"
        )

    @property
    def mean(self):
        mus1 = self.diag_gauss.mean
        mus2, _ = self.net(torch.cat((self.obs, mus1), -1))
        return mus1.squeeze(), mus2.squeeze()


class TanhGauss(GaussInputDist):
    def __init__(self, diag_gauss):
        super(TanhGauss, self).__init__(diag_gauss)

    def log_prob(self, tanh_domain_x):
        gauss_domain_x = (
            torch.log(1.0 + tanh_domain_x) / 2
            - torch.log(1.0 - tanh_domain_x) / 2
        )
        return self._log_prob_from_gauss(gauss_domain_x)

    def _log_prob_from_gauss(self, x):
        """
        x can be (*, B, x_dim)
        """
        tanh_term = (1.0 - torch.tanh(x) ** 2).log().sum(-1)
        return self.diag_gauss.log_prob(x).sum(-1) - tanh_term

    def sample(self):
        return torch.tanh(self.diag_gauss.sample())

    def rsample(self):
        return torch.tanh(self.diag_gauss.rsample())

    def get_UT_trick_input(self):
        return torch.tanh(super().get_UT_trick_input())

    @property
    def mean(self):
        """Return tanh at the mean of the gaussian."""
        return torch.tanh(self.diag_gauss.mean)

    @property
    def stddev(self):
        raise NotImplementedError

    def log_prob_UT_trick(self):
        f_in = super().get_UT_trick_input().permute((1, 0, 2))
        return self._log_prob_from_gauss(f_in).mean(0)


def batch_UT_trick_from_samples(f, obs, samples):
    """
    Args:
        f: Callable.
        obs: Tensor of shape (B, obs_dim)
        samples: Tensor of shape (B, 2 * action_dim + 1, action_dim).
    """
    obs = obs.unsqueeze(1)  # (B, 1, obs_dim);
    sizes = [-1, samples.size(1), -1]
    # note torch.expand, is memory efficient because it does views;
    # this means the same memory might be accessed multiple
    # times if the value has to be used in different places;
    # this is fine here since obs has requires_grad=False;
    # also in the torch.expand api, -1 means don't change current dim
    # rather than infer dim;
    f_in = torch.cat((obs.expand(*sizes), samples), -1)
    return f(f_in).mean(1)  # avg across 2D + 1 points;


def batch_UT_trick(f, obs, mus, sigmas, offset=1.0):
    """
    Assumes the latent variable component of the input of f
    is diagonal Gaussian with Batch of mean vectors in mus and
    batch of standard deviations in sigmas.

    Args:
        f: Callable that maps last dim of input to output_dim.
        obs: Tensor of shape (B, obs_dim).
        mus: Tensor of shape (B, action_dim).
        sigmas: Tensor of shape (B, action_dim).

    Returns:
        Tensor of shape (B, out_dim).

    Note:
        This performs the Unscented transform trick. For diagonal
        Gaussian latents, the eigenvectors are the axis aligned
        coordinate vectors with eigenvalues being the squared
        standard deviations. To do the UT, I eval f at the mean
        and the positive and negative pivots. The pivots
        are mean +- sqrt(eig_val) * eig_vec -> leading to
        2 * action_dim + 1 inputs per mean vector.
    """
    obs_dim, action_dim = obs.shape[-1], mus.shape[-1]
    B = len(obs)  # batch_dim

    # concat obs and mus -> (B, obs_action_dim);
    obs_mus = torch.cat((obs, mus), -1)

    # shape of diags is (B, action_dim, action_dim)
    diags = sigmas.unsqueeze(1) * torch.eye(action_dim)

    # pad inner most axis
    # on the left to make (B, action_dim, obs_action_dim)
    diags = nn.functional.pad(diags, (obs_dim, 0))

    # concat negative pivots with row of zeros;
    diags = torch.cat(
        (diags, torch.zeros((B, 1, diags.shape[-1])), -diags), 1
    )

    # return shape (B, out_dim)
    return f(obs_mus.unsqueeze(1) + offset * diags).mean(1)


def latent_only_batch_UT_trick(
    f, mus, sigmas, with_log_prob=False, offset=1.0
):
    B, D = mus.shape

    # make to (B, D, D) shape;
    diags = sigmas.unsqueeze(1) * torch.eye(D)

    # make to (B, 2D + 1, D) shape;
    diags = torch.cat((diags, torch.zeros((B, 1, D)), -diags), 1)

    # return (B, out_dim) shape;
    if with_log_prob:
        f_in = (mus.unsqueeze(1) + diags).permute((1, 0, 2))
        # print(f_in.detach().min(), f_in.detach().max())
        return f(f_in).mean(0)
    return f(mus.unsqueeze(1) + offset * diags).mean(1)
