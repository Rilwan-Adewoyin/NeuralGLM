import torch
from torch.autograd.grad_mode import set_grad_enabled
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
from typing import Callable, Optional, Union
# torch.autograd.set_detect_anomaly(True)
import numpy as np

from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.transforms import ExpTransform
from torch.distributions.transformed_distribution import TransformedDistribution
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler, FunctionTransformer, MaxAbsScaler
from loss_utils import *

import distributions

from typing import List


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
    'gamma_hurdle':distributions.GammaHurdle,
    'lognormal_hurdle':distributions.LogNormalHurdle,
    'compound_poisson':distributions.CompoundPoisson
}

MAP_DISTRIBUTION_LOSS = {

    'poisson': PoissonNLLLoss, #s(log_input, target, log_input=self.log_input, full=self.full,eps=self.eps, reduction=self.reduction)
    'normal': GaussianNLLLoss, # (input, target, var, full=self.full, eps=self.eps, reduction=self.reduction)
    'lognormal_hurdle':LogNormalHurdleNLLLoss,
    'compound_poisson':CPNLLLoss,
    'CP_GLM':CPGLMNLLLoss,
    'compound_poisson':CompoundPoissonGammaNLLLoss,
    'gamma_hurdle':GammaHurdleNLLLoss
}


class GLMMixin:

    def _get_inv_link(self, link_name):
        invfunc =   MAP_LINK_INVFUNC[link_name]            
        return invfunc

    def check_distribution_mean_link(self, distribution_name, link_name):
        #TODO: implement code that checks if the distribution and link function chosen by a user match
        bool_check = link_name in MAP_DISTRIBUTION_MEANLINKFUNC.get(distribution_name,[])
        return bool_check

    def check_distribution_dispersion_link(self, distribution_name, link_name):
        #TODO: implement code that checks if the distribution and link function chosen by a user match
        bool_check = link_name in MAP_DISTRIBUTION_DISPERSIONLINKFUNC.get(distribution_name,[])
        return bool_check

    def _get_distribution(self, distribution_name:str) -> Distribution:
        """Retrieves the distribution class when passed the name of the distributino

        Args:
            distribution_name ([type]): [description]

        Returns:
            [type]: [description]
        """
        return MAP_NAME_DISTRIBUTION[distribution_name]
    
    def _get_mean_range(self, distribution_name, scaler=None):
        
        if scaler==None:
            min = 0
            max = None
            pass

        elif isinstance(scaler,MinMaxScaler):
            min = scaler.feature_range[0]
            max = None
        
        elif isinstance(scaler, MaxAbsScaler):
            min = 0
            max = None
        else:
            raise NotImplementedError
        
        return min, max
        
    def _get_dispersion_range(self, distribution_name, **kwargs):

    
        if distribution_name == "lognormal":
            min = kwargs.get('eps',1e-3)
            max = None
        elif distribution_name == "lognormal_hurdle":
            min = kwargs.get('eps',1e-3 )
            max = None
        
        return min, max
    
    def destandardize(self, mean: Union[Tensor, np.ndarray], scaler: Union[MinMaxScaler,StandardScaler] ):
        
        if isinstance(scaler, FunctionTransformer):
            # mean = mean - scaler.
            mean = scaler.inverse_transform(mean)
        elif scaler == None:
            pass
        

        mean = mean

        return mean

    def _get_loglikelihood_loss_func(self,  distribution_name ):
        return MAP_DISTRIBUTION_LOSS[distribution_name]