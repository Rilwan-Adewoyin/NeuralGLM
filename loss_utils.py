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
import torchtyping
import numpy as np
from mpmath import *

#TODO: Add to notes: Dropout Inference in Bayesian Neural Networks, page 3, column 2, last paragraph of section 2.3


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

    def __init__(self, *, full: bool = False, eps: float = 1e-6, reduction: str = 'mean', pos_weight=1 ) -> None:
        super(LogNormalHurdleNLLLoss, self).__init__(None, None, reduction)
        self.full = full
        self.eps = eps
        self.bce_logits = torch.nn.BCEWithLogitsLoss(reduction='sum')
        self.register_buffer('pos_weight',torch.tensor([pos_weight]) )
        self.register_buffer('pi',torch.tensor(math.pi) )

        #TODO: later add increased weights to improve classification for rainy days (precision)
        
        assert reduction == 'mean'

    def forward(self, input: Tensor, did_rain:Tensor, target: Tensor, var: Tensor, logit: Tensor, **kwargs) -> Tensor:
        """
            Input : true rain fall
            target: predicted mean
            var: predicted variance
        """
        if did_rain.dtype != logit.dtype:
            did_rain = did_rain.to(logit.dtype)

        if input.dtype != target.dtype:
            input = input.to(target.dtype)

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

        # logit / etc loss
       
        # Calculate the hurdle burnouilli loss
        loss_bce = self.bce_logits( logit, did_rain)

        # Calculate the ll-loss
        indices_rainydays = torch.where(did_rain.view( (1,-1))>0)
        
        input_rainydays = input.view((1,-1))[indices_rainydays]
        target_rainydays = target.view((1,-1))[indices_rainydays]
        var_rainydays = var.view((1,-1))[indices_rainydays]

        loss_cont = 0.5 * (torch.log(var_rainydays) + (torch.log(input_rainydays) - target_rainydays)**2 / var_rainydays) \
                        + torch.log(input_rainydays) + 0.5*torch.log(2*self.pi)

            #NOTE(add to papers to write):This initialization with high variance acts as regularizer to smoothen initial loss funciton
        loss_cont = self.pos_weight * loss_cont
        loss_cont = loss_cont.sum()

        if self.reduction == 'mean':
            loss_bce = loss_bce/input.numel()
            loss_cont = loss_cont/input.numel()
            loss = loss_bce + loss_cont 

        return loss, {'loss_bce':loss_bce.detach() , 'loss_cont':loss_cont.detach()}            


#In this version of the loss, the parameters we learn are those of a regular CP model - not the GLM CP model
class CPNLLLoss(_Loss):
    __constants__ = ['full', 'eps', 'reduction']
    full: bool
    eps: float

    def __init__(self, *, full: bool = False, eps: float = 1e-6, reduction: str = 'mean', pos_weight=1 ) -> None:
        super(CPNLLLoss, self).__init__(None, None, reduction)
        self.full = full
        self.eps = eps
        self.bce_logits = torch.nn.BCEWithLogitsLoss(reduction='sum')
        self.register_buffer('pos_weight',torch.tensor([pos_weight]) )
        self.register_buffer('pi',torch.tensor(math.pi) )

        #TODO: later add increased weights to improve classification for rainy days (precision)
        
        assert reduction == 'mean'

    #For Compound poisson GLM, the mean is CP and the dispersion is Gamma
    #The loss for fitting the CP is:

    def ComPois(a,   #alpha
            b,   #beta
            d,   #delta
            data,#rainfall obs
            ):
        mp.dps = 15  #precision for infinite sum
        n=len(data)  
        S=-n*d       #initializing 
        for y in data:
            if y!=0:
                #This expression is defined in terms of regular CP parameters, not mean/dispersion/p as in a CP-GLM.
                S+=-(b*y)+log(nsum(lambda z: (((b**(z*a))*(y**(z*a-1))*(d**z))/(gamma(z*a)*factorial(z))), [1, inf]))           
        return -S

    #the values maximizing this Loss then need to be used to find the mean and dispersion corresponding to the GLM form of a CP model:
    
    def CP_param(a,b,d):
        p=(2+a)/(1+a)
        miu=(d*a)/b
        phi=(a+1)/((b**(2-p))*((a*d)**(p-1)))
        return [miu,phi]

    def forward(self, input: Tensor, did_rain:Tensor, target: Tensor, var: Tensor, logit: Tensor, **kwargs) -> Tensor:
        """
            Input : true rain fall
            target: predicted mean
            var: predicted variance
        """
        if did_rain.dtype != logit.dtype:
            did_rain = did_rain.to(logit.dtype)

        if input.dtype != target.dtype:
            input = input.to(target.dtype)

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

        # loss
       
        # Calculate the hurdle burnouilli loss
        loss_bce = self.bce_logits( logit, did_rain)

        # Calculate the ll-loss
        indices_rainydays = torch.where(did_rain.view( (1,-1))>0)
        
        input_rainydays = input.view((1,-1))[indices_rainydays]
        target_rainydays = target.view((1,-1))[indices_rainydays]
        var_rainydays = var.view((1,-1))[indices_rainydays]

        # Expression for the CP model loss
        loss_cont = ComPois(a=alpha,b=beta,d=delta,data=input_rainydays)

            #NOTE(add to papers to write):This initialization with high variance acts as regularizer to smoothen initial loss funciton
        loss_cont = self.pos_weight * loss_cont
        loss_cont = loss_cont.sum()

        if self.reduction == 'mean':
            loss_bce = loss_bce/input.numel()
            loss_cont = loss_cont/input.numel()
            loss = loss_bce + loss_cont 

        return loss, {'loss_bce':loss_bce.detach() , 'loss_cont':loss_cont.detach()}        


#In this version, the parameters we learn are those of the CP GLM - mean and dispersion.
class CPGLMNLLLoss(_Loss):
    __constants__ = ['full', 'eps', 'reduction']
    full: bool
    eps: float

    def __init__(self, *, full: bool = False, eps: float = 1e-6, reduction: str = 'mean', pos_weight=1 ) -> None:
        super(CPNLLLoss, self).__init__(None, None, reduction)
        self.full = full
        self.eps = eps
        self.bce_logits = torch.nn.BCEWithLogitsLoss(reduction='sum')
        self.register_buffer('pos_weight',torch.tensor([pos_weight]) )
        self.register_buffer('pi',torch.tensor(math.pi) )

        #TODO: later add increased weights to improve classification for rainy days (precision)
        
        assert reduction == 'mean'

    #For Compound poisson GLM, the mean is CP and the dispersion is Gamma
    #The loss for fitting the CP is:

    #CP log lokelihood in GLM form
    
    #first define a value used in the CP-GLM loss clalculations
    def W_CP(y,a,var,z):
        return(((y)**(z*a))/((var**(z*(1+a)))*((1/(1+a))**z)*((1/(1+a))**(z*a))*(factorial(z))*(gamma(z*a))))
    def CP_GLM_NLLLoss(a,       #alpha - a Gama dist parameter for both laten variables used in CP
                        rain_in,   
                        rain_out,
                        var):   #The variance of rain data
        mp.dps = 15  #precision for infinite sum
        n=len(rain_in)
        k=np.count_nonzero(a, axis=None, *, keepdims=False)
        #initialise loss value by adding the terms not dependent on the rain data.
        S= (n-k)*log(-((1+a)*(rain_out**(a/(1+a))))/(var*a))-(((1+a)*(rain_out**(a/(1+a))))/(var*a))
        #add the 'sum' part 
        for y in rain_in:
            if y!=0:
                S+=log(nsum(lambda z:W_CP(y=rain_in,a=a,var=var,z=z)))-log(rain_in)-((rain_in*(1+a))/(var*(rain_out**(1/(1+a)))))
        return -S

    def forward(self, input: Tensor, did_rain:Tensor, target: Tensor, var: Tensor, logit: Tensor, **kwargs) -> Tensor:
        """
            Input : true rain fall
            target: predicted mean
            var: predicted variance
        """
        if did_rain.dtype != logit.dtype:
            did_rain = did_rain.to(logit.dtype)

        if input.dtype != target.dtype:
            input = input.to(target.dtype)

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

        # loss
       
        # Calculate the hurdle burnouilli loss
        loss_bce = self.bce_logits( logit, did_rain)

        # Calculate the ll-loss
        indices_rainydays = torch.where(did_rain.view( (1,-1))>0)
        
        input_rainydays = input.view((1,-1))[indices_rainydays]
        target_rainydays = target.view((1,-1))[indices_rainydays]
        var_rainydays = var.view((1,-1))[indices_rainydays]

        # Expression for the CP model loss
        loss_cont = CP_GLM_NLLLoss(a=,rain_in=input_rainydays,rain_out=target_rainydays,var=var_rainydays)

        #NOTE(add to papers to write):This initialization with high variance acts as regularizer to smoothen initial loss funciton
        loss_cont = self.pos_weight * loss_cont
        loss_cont = loss_cont.sum()

        if self.reduction == 'mean':
            loss_bce = loss_bce/input.numel()
            loss_cont = loss_cont/input.numel()
            loss = loss_bce + loss_cont 

        return loss, {'loss_bce':loss_bce.detach() , 'loss_cont':loss_cont.detach()}

class GammaNLLLoss(_Loss):
    __constants__ = ['full', 'eps', 'reduction']
    full: bool
    eps: float
# Define a function to compute a Gamma negative loglikelihood
    def Gamma_NLL(obs,#observations
                    m, #mean
                    d): #dipersion
        #initialise S
        S=-n*(log(gamma(1/d)+(log(d*m)/d)))
        #itterate through observations
        for y in obs:
            S+=log(y)*((1/d)-1)+(y/(d*m))
        return -S

    def __init__(self, *, full: bool = False, eps: float = 1e-6, reduction: str = 'mean', pos_weight=1 ) -> None:
        super(LogNormalHurdleNLLLoss, self).__init__(None, None, reduction)
        self.full = full
        self.eps = eps
        self.bce_logits = torch.nn.BCEWithLogitsLoss(reduction='sum')
        self.register_buffer('pos_weight',torch.tensor([pos_weight]) )
        self.register_buffer('pi',torch.tensor(math.pi) )

        #TODO: later add increased weights to improve classification for rainy days (precision)
        
        assert reduction == 'mean'

    def forward(self, input: Tensor, did_rain:Tensor, target: Tensor, var: Tensor, logit: Tensor, **kwargs) -> Tensor:
        """
            Input : true rain fall
            target: predicted mean
            var: predicted variance
        """
        if did_rain.dtype != logit.dtype:
            did_rain = did_rain.to(logit.dtype)

        if input.dtype != target.dtype:
            input = input.to(target.dtype)

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

        # logit / etc loss
       
        # Calculate the hurdle burnouilli loss
        loss_bce = self.bce_logits( logit, did_rain)

        # Calculate the ll-loss
        indices_rainydays = torch.where(did_rain.view( (1,-1))>0)
        
        input_rainydays = input.view((1,-1))[indices_rainydays]
        target_rainydays = target.view((1,-1))[indices_rainydays]
        var_rainydays = var.view((1,-1))[indices_rainydays]

        loss_cont =Gamma_NLL(obs=input_rainydays,m=target_rainydays,d=var_rainydays)

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
    'normal': GaussianNLLLoss, # (input, target, var, full=self.full, eps=self.eps, reduction=self.reduction)
    'lognormal': LogNormalNLLLoss,
    'lognormal_hurdle':LogNormalHurdleNLLLoss,
    'compound_poisson':CPNLLLoss,
    'CP_GLM':CPGLMNLLLoss,
    'gamma':GammaNLLLoss
}

class LossMixin:

    def _get_loglikelihood_loss_func(self,  distribution_name ):
        return MAP_DISTRIBUTION_LOSS[distribution_name]