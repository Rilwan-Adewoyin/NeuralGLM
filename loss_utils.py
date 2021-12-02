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
import math

#TODO: Add to notes: Dropout Inference in Bayesian Neural Networks, page 3, column 2, last paragraph of section 2.3

#TODO: Consider using symmetric JS divergence and assuming each datapoint to be the mean, and have variance=1, or whichever term would reduce it to 

class LogNormalNLLLoss(_Loss):
    __constants__ = ['full', 'eps', 'reduction']
    full: bool  
    eps: float

    def __init__(self, *, full: bool = False, eps: float = 1e-6, reduction: str = 'mean') -> None:
        super(LogNormalNLLLoss, self).__init__(None, None, reduction)
        self.full = full
        self.eps = eps
        

    def forward(self, input: Tensor, target: Tensor, var: Tensor) -> Tensor:
        # return F.gaussian_nll_loss(input, target, var, full=self.full, eps=self.eps, reduction=self.reduction)

        # Check var size
        # If var.size == input.size, the case is heteroscedastic and no further checks are needed.
        # Otherwise:
        if var.size() != input.size():

            # If var is one dimension short of input, but the sizes match otherwise, then this is a homoscedastic case.
            # e.g. input.size = (10, 2, 3), var.size = (10, 2)
            # -> unsqueeze var so that var.shape = (10, 2, 1)
            # this is done so that broadcasting can happen in the loss calculation
            if input.size()[:-1] == var.size():
                var = torch.unsqueeze(var, -1)

            # This checks if the sizes match up to the final dimension, and the final dimension of var is of size 1.
            # This is also a homoscedastic case.
            # e.g. input.size = (10, 2, 3), var.size = (10, 2, 1)
            elif input.size()[:-1] == var.size()[:-1] and var.size(-1) == 1:  # Heteroscedastic case
                pass

            # If none of the above pass, then the size of var is incorrect.
            else:
                raise ValueError("var is of incorrect size")

        # Check validity of reduction mode
        if self.reduction != 'none' and self.reduction != 'mean' and self.reduction != 'sum':
            raise ValueError(self.reduction + " is not valid")

        # Entries of var must be non-negative
        if torch.any(var < 0):
            raise ValueError("var has negative entry/entries")

        # Clamp for stability
        var = var.clone()
        with torch.no_grad():
            var.clamp_(min=self.eps)

        # Calculate the loss
        loss = 0.5 * (torch.log(var) + (torch.log(input) - target)**2 / var) + torch.log(input)
        if self.full:
            loss += 0.5 * math.log(2 * math.pi)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
            

MAP_DISTRIBUTION_LOSS = {

    'poisson': PoissonNLLLoss, #s(log_input, target, log_input=self.log_input, full=self.full,eps=self.eps, reduction=self.reduction)
    'normal': GaussianNLLLoss, # (input, target, var, full=self.full, eps=self.eps, reduction=self.reduction
    'lognormal': LogNormalNLLLoss,
}

class LossMixin:

    def _get_loglikelihood_loss_func(self,  distribution_name ):
        return MAP_DISTRIBUTION_LOSS[distribution_name]