import torch
from torch import distributions
from torch.distributions.gamma import Gamma
from torch.distributions.half_normal import HalfNormal
from torch.distributions.log_normal import LogNormal
from torch.distributions.normal import Normal

import torch.functional as F
from torch.nn.modules.loss import GaussianNLLLoss, PoissonNLLLoss
from scipy import stats

from torch.nn.modules.loss import _Loss
from torch import Tensor
from typing import Callable, Optional
# torch.autograd.set_detect_anomaly(True)

#NOTES: Cannonical link function are sometimes inappropriate

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



MAP_LINK_INVFUNC = {
    'identity': torch.nn.Identity(),
    'negative_inverse': lambda val: -torch.pow(val,-1),
    'inverse': Inverse(),
    'relu':torch.nn.ReLU(),
    'relu_inverse':torch.nn.Sequential( torch.nn.ReLU(), Inverse() )

}

# Maps the distribution name to a list of canonical/common inverse link functions. 
MAP_DISTRIBUTION_MEANLINKFUNC = {
    'normal': ['identity'],
    'lognormal' : ['identity'],
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
}

MAP_NAME_DISTRIBUTION = {
    'normal': Normal ,
    'lognormal': LogNormal ,
    'gamma': Gamma,
    'HalfNormal': HalfNormal,

}


class GLMMixin:

    def _get_inv_link(self, link_name):
        return MAP_LINK_INVFUNC[link_name]

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
    
