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
        # input = input.clone()
        with torch.no_grad():
            var.clamp_(min=self.eps)
            # input.clamp_(min=self.eps)

        # Calculate the loss
        loss = 0.5 * (torch.log(var) + (torch.log(input) - target)**2 / var) 
        if self.full:
            loss += 0.5 * math.log(2 * math.pi) + torch.log(input)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class LogNormalHurdleNLLLoss(_Loss):
    __constants__ = ['full', 'eps', 'reduction']
    full: bool
    eps: float

    def __init__(self, *, full: bool = False, eps: float = 1e-6, reduction: str = 'mean', pos_weight=torch.ones([1]) ) -> None:
        super(LogNormalHurdleNLLLoss, self).__init__(None, None, reduction)
        self.full = full
        self.eps = eps
        self.bce_logits = torch.nn.BCEWithLogitsLoss(reduction='sum')
        self.bce_probs = torch.nn.BCEWithLogitsLoss(reduction='sum')
        self.pos_weight = pos_weight
        #TODO: later add increased weights to improve classification for rainy days (precision)
        
        assert reduction == 'mean'

    def forward(self, input: Tensor, did_rain:Tensor, target: Tensor, var: Tensor, prob: Tensor, **kwargs) -> Tensor:
        """
            Input : true rain fall
            target: predicted mean
            var: predicted variance
        """
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
        input = input.clone()
        with torch.no_grad():
            var.clamp_(min=self.eps)
            input.clamp_(min=self.eps)

        #  probit / logit / etc loss
        # did_rain = torch.where( input>0, 1.0, 0.0 )
        
        # Calculate the hurdle burnouilli loss
        if kwargs.get('use_logits',False):
            loss_bce = self.bce_logits( prob, did_rain)
        else:
            loss_bce = self.bce_probs(prob, did_rain)

        # Calculate the ll-loss
        indices_rainydays = torch.where(did_rain.view( (1,-1))>0)
        
        input_rainydays = input.view((1,-1))[indices_rainydays]
        target_rainydays = target.view((1,-1))[indices_rainydays]
        var_rainydays = var.view((1,-1))[indices_rainydays]

        # NOTE: input_rainydays is essentially a sample from a lognormal distribution. 
            #   Therefore we do not have to do the log() operation on input_rainydays
        # loss_cont = 0.5 * (torch.log(var_rainydays) + (torch.log(input_rainydays) - target_rainydays)**2 / var_rainydays) 
        loss_cont = 0.5 * (torch.log(var_rainydays) + (input_rainydays - target_rainydays)**2 / var_rainydays)  
            #NOTE(add to papers to write):This initialization with high variance acts as regularizer to smoothen initial loss funciton
        loss_cont = self.pos_weight * loss_cont
        loss_cont = loss_cont.sum()

        if self.reduction == 'mean':
            loss_bce = loss_bce/input.numel()
            loss_cont = loss_cont/input.numel()
            loss = loss_bce + loss_cont 

        return loss, {'loss_bce':loss_bce.detach() , 'loss_cont':loss_cont.detach()}            

MAP_DISTRIBUTION_LOSS = {

    'poisson': PoissonNLLLoss, #s(log_input, target, log_input=self.log_input, full=self.full,eps=self.eps, reduction=self.reduction)
    'normal': GaussianNLLLoss, # (input, target, var, full=self.full, eps=self.eps, reduction=self.reduction
    'lognormal': LogNormalNLLLoss,
    'lognormal_hurdle':LogNormalHurdleNLLLoss
}

class LossMixin:

    def _get_loglikelihood_loss_func(self,  distribution_name ):
        return MAP_DISTRIBUTION_LOSS[distribution_name]