import torch
from torch import distributions
from torch.distributions.gamma import Gamma
from torch.distributions.half_normal import HalfNormal
from torch.distributions.log_normal import LogNormal
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli

import torch.functional as F
from torch.nn.modules.loss import GaussianNLLLoss, PoissonNLLLoss
from scipy import stats

from torch.nn.modules.loss import _Loss
from torch import Tensor
from typing import Callable, Optional
# torch.autograd.set_detect_anomaly(True)
import numpy as np

from torch.distributions import constraints
from torch.distributions.transforms import ExpTransform
from torch.distributions.transformed_distribution import TransformedDistribution

# Distributions


class LogNormalHurdle():
    r"""
    Creates a log-normal distribution parameterized by
    :attr:`loc` and :attr:`scale` where::

        X ~ Normal(loc, scale)
        Y = exp(X) ~ LogNormal(loc, scale)

    Example::

        >>> m = LogNormal(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # log-normal distributed with mean=0 and stddev=1
        tensor([ 0.1046])

    Args:
        loc (float or Tensor): mean of log of distribution
        scale (float or Tensor): standard deviation of log of the distribution
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(self, loc, scale, prob, validate_args=None):
        
        self.bernoulli_dist = Bernoulli(prob, validate_args=validate_args)
        self.lognormal_dist = LogNormal(loc, scale, validate_args=validate_args)

    def sample(self,sample_size=(1,)):
        raise NotImplementedError # Check method below allows different mean and var to suppled to each varaible
        rain_prob = self.bernoulli_dist.sample( sample_size )

        sampled_rain = torch.where(rain_prob==1, self.lognormal_dist.sample( (1,) ), 0  )

        return sampled_rain

    # @property
    # def loc(self):
    #     return self.base_dist.loc

    # @property
    # def scale(self):
    #     return self.base_dist.scale

    # @property
    # def mean(self):
    #     return (self.loc + self.scale.pow(2) / 2).exp()

    # @property
    # def variance(self):
    #     return (self.scale.pow(2).exp() - 1) * (2 * self.loc + self.scale.pow(2)).exp()

    # def entropy(self):
    #     return self.base_dist.entropy() + self.loc



#Mean functions
class Inverse(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.eps = 1e-6

    def forward(self, x ):
        
        x = x.clone()
        with torch.no_grad():
            x.clamp_(min=self.eps)

        outp = 1/x

        return outp

class Shift(torch.nn.Module):
    def __init__(self, shift=1) -> None:
        super().__init__()
        self.shift = shift

    def forward(self, x ):
        x = x + self.shift
        return x


MAP_LINK_INVFUNC = {
    'identity': torch.nn.Identity(),
    'shift': Shift,
    'negative_inverse': lambda val: -torch.pow(val,-1),
    'inverse': Inverse(),
    'relu':torch.nn.ReLU(),
    'relu_inverse':torch.nn.Sequential( torch.nn.ReLU(), Inverse() ),
    'sigmoid':torch.nn.Sigmoid()
}

# Maps the distribution name to a list of canonical/common inverse link functions. 
MAP_DISTRIBUTION_MEANLINKFUNC = {
    'normal': ['identity'],
    'lognormal' : ['identity','shift'],
    'lognormal_hurdle' : ['identity'],
    'exponential':['negative_inverse','inverse'],
    'gamma':['negative_inverse','log','inverse'],
    'inverse_guassian':['inverse_squared'],
    'bernouilli': ['logit'],
    'binomial': ['logit'],
    'categorical': ['logit'],
    'multinomial': ['logit'],
    'poisson':['log']
}

MAP_DISTRIBUTION_DISPERSIONLINKFUNC = {
    'normal': ['negative_inverse','inverse','relu'],
    'lognormal': ['negative_inverse','inverse','relu','relu_inverse'],
    'lognormal_hurdle':['negative_inverse','inverse','relu','relu_inverse']
}

MAP_NAME_DISTRIBUTION = {
    'normal': Normal ,
    'lognormal': LogNormal ,
    'gamma': Gamma,
    'HalfNormal': HalfNormal,
    'lognormal_hurdle':LogNormalHurdle,
}

class GLMMixin:

    def _get_inv_link(self, link_name, **kwargs):
        if len(kwargs)== 0:
            invfunc =   MAP_LINK_INVFUNC[link_name]
        else:
            invfunc =   MAP_LINK_INVFUNC[link_name](**kwargs)
            
        return invfunc
    def check_distribution_mean_link(self, distribution_name, link_name):
        #TODO: implement code that checks if the distribution and link function chosen by a user match
        bool_check = link_name in MAP_DISTRIBUTION_MEANLINKFUNC.get(distribution_name,[])
        return bool_check

    def check_distribution_dispersion_link(self, distribution_name, link_name):
        #TODO: implement code that checks if the distribution and link function chosen by a user match
        bool_check = link_name in MAP_DISTRIBUTION_DISPERSIONLINKFUNC.get(distribution_name,[])
        return bool_check

    def _get_distribution(self, distribution_name):
        return MAP_NAME_DISTRIBUTION[distribution_name]
    
    def _get_mean_range(self, distribution_name, standardized):

        if standardized:
            min = float((-self.scaler_targets.data_min_[0])/(self.scaler_targets.data_max_[0] - self.scaler_targets.data_min_[0]) )
            max = None
        else:
            raise NotImplementedError
        
        return min, max
        
    def _get_dispersion_range(self, distribution_name, standardized, **kwargs):

        if standardized:
            if distribution_name == "lognormal":
                min = 0
                max = None
            elif distribution_name == "lognormal_hurdle":
                min = kwargs.get('eps',1e-7)
                max = None
        else:
            raise NotImplementedError
        
        return min, max