import torch
from torch import nn


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

    def get_UT_trick_input(self):
        mus = self.diag_gauss.mean
        sigmas = self.diag_gauss.stddev
        B, D = mus.shape

        # make to (B, D, D) shape;
        diags = sigmas.unsqueeze(1) * torch.eye(D)
        
        # make to (B, 2D + 1, D) shape;
        diags = torch.cat(
            (
                diags, torch.zeros((B, 1, D)), -diags
            ), 1
        )

        # return (B, 2D + 1, D) shape;
        return mus.unsqueeze(1) + diags

    def log_prob_UT_trick(self):
        pass


class GaussDist(GaussInputDist):
    def __init__(self, diag_gauss):
        super(GaussDist, self).__init__(diag_gauss)
    
    def log_prob(self, x):
        return self.diag_gauss.log_prob(x)
    
    def sample(self):
        return self.diag_gauss.sample()
    
    def rsample(self):
        return self.diag_gauss.rsample()
    
    def log_prob_UT_trick(self):
        f_in = self.get_UT_trick_input().permute((1, 0, 2))
        return self.log_prob(f_in).mean(0)  # (B, D)


class TanhGauss(GaussInputDist):
    def __init__(self, diag_gauss):
        """
        This distribution does not have stddev attribute.
        If UT_trick is to be applied, can try sth like:
        
        >>> mus = tanh_dist.diag_gauss.mean
        >>> sigmas = tang_dist.diag_gauss.stddev
        >>> f = lambda x: net(torch.tanh(x))
        >>> result = UT_trick(f, mus, sigmas)
        """
        super(TanhGauss, self).__init__(diag_gauss)

    def log_prob(self, tanh_domain_x):
        gauss_domain_x = torch.log(1. + tanh_domain_x) / 2 - torch.log(1. - tanh_domain_x) / 2
        return self._log_prob_from_gauss(gauss_domain_x)

    def _log_prob_from_gauss(self, x):
        """
        x can be (*, B, x_dim)
        """
        tanh_term = (1. - torch.tanh(x) ** 2).log().sum(-1)
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
        samples: Tensor of shape (B, 2D + 1, D).
    """
    obs = obs.unsqueeze(1)  # (B, 1, obs_dim);
    sizes = [-1, samples.size(1), -1]
    # note torch.expand, is memory efficient because it does views;
    # this means the same memory might be accessed multiple
    # times if the value has to be used in different places;
    # this is fine here since obs has requires_grad=False;
    # also in the torch.expand api, -1 means don't change current dim
    # rather than infer dim;
    f_in = torch.cat(
            (
                obs.expand(*sizes), samples
            ), -1
        )
    return f(f_in).mean(1) # avg across 2D + 1 points;


def batch_UT_trick(f, obs, mus, sigmas):
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
        (
        diags, torch.zeros((B, 1, diags.shape[-1])), - diags
        ), 1
    )

    # return shape (B, out_dim)
    return f(obs_mus.unsqueeze(1) + diags).mean(1)


def latent_only_batch_UT_trick(f, mus, sigmas, with_log_prob=False):
    B, D = mus.shape

    # make to (B, D, D) shape;
    diags = sigmas.unsqueeze(1) * torch.eye(D)
    
    # make to (B, 2D + 1, D) shape;
    diags = torch.cat(
        (
        diags, torch.zeros((B, 1, D)), -diags
        ), 1
    )

    # return (B, out_dim) shape;
    if with_log_prob:
        f_in = (mus.unsqueeze(1) + diags).permute((1, 0, 2))
        # print(f_in.detach().min(), f_in.detach().max())
        return f(f_in).mean(0)
    return f(mus.unsqueeze(1) + diags).mean(1)
