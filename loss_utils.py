from cmath import log
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
#from mpmath import *

#TODO: Add to notes: Dropout Inference in Bayesian Neural Networks, page 3, column 2, last paragraph of section 2.3

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
    
    def lognormal_nll(self, obs, mu, disp ):

        ll  = -0.5 * (torch.log(disp) + (torch.log(obs) - mu)**2 / disp) 
                        
        if self.full:
            ll += -0.5*torch.log(2*self.pi) - torch.log(obs)

        return -ll

    def forward(self, rain: Tensor, did_rain:Tensor, mean: Tensor, disp: Tensor, logits: Tensor, **kwargs) -> Tensor:
        """
            rain : true rain fall
            did_rain: whether or not it rained
            mean: predicted mean
            disp: predicted dispiance
            logits: predicted prob of rain
        """
        if did_rain.dtype != logits.dtype:
            did_rain = did_rain.to(logits.dtype)

        if rain.dtype != mean.dtype:
            rain = rain.to(mean.dtype)

        # Check disp size
        # If disp.size == rain.size, the case is heteroscedastic and no further checks are needed.
        # Otherwise:
        if disp.size() != rain.size():

            # If disp is one dimension short of rain, but the sizes match otherwise, then this is a homoscedastic case.
            # e.g. rain.size = (10, 2, 3), disp.size = (10, 2)
            # -> unsqueeze disp so that disp.shape = (10, 2, 1)
            # this is done so that broadcasting can happen in the loss calculation
            if rain.size()[:-1] == disp.size():
                disp = torch.unsqueeze(disp, -1)

            # This checks if the sizes match up to the final dimension, and the final dimension of disp is of size 1.
            # This is also a homoscedastic case.
            # e.g. rain.size = (10, 2, 3), disp.size = (10, 2, 1)
            elif rain.size()[:-1] == disp.size()[:-1] and disp.size(-1) == 1:  # Heteroscedastic case
                pass

            # If none of the above pass, then the size of disp is incorrect.
            else:
                raise ValueError("disp is of incorrect size")

        # Check validity of reduction mode
        if self.reduction != 'none' and self.reduction != 'mean' and self.reduction != 'sum':
            raise ValueError(self.reduction + " is not valid")

        # Entries of disp must be non-negative
        if torch.any(disp < 0):
            raise ValueError("disp has negative entry/entries")

        # Clamp for stability
        disp = disp.clone()
        rain = rain.clone()
        with torch.no_grad():
            disp.clamp_(min=self.eps)
            rain.clamp_(min=self.eps)

        # logits / etc loss
       
        # Calculate the hurdle burnouilli loss
        loss_norain = self.bce_logits( logits, did_rain)

        # Calculate the ll-loss
        indices_rainydays = torch.where(did_rain.view( (1,-1))>0)
        
        rain_rainydays = rain.view((1,-1))[indices_rainydays]
        mean_rainydays = mean.view((1,-1))[indices_rainydays]
        disp_rainydays = disp.view((1,-1))[indices_rainydays]

        loss_rain = self.lognormal_nll(rain_rainydays, mean_rainydays, disp_rainydays)

        loss_rain = self.pos_weight * loss_rain
        loss_rain = loss_rain.sum()

        if self.reduction == 'mean':
            loss_norain = loss_norain/rain.numel()
            loss_rain = loss_rain/rain.numel()
            loss = loss_norain + loss_rain 

        return loss, {'loss_norain':loss_norain.detach() , 'loss_rain':loss_rain.detach()}            

class GammaHurdleNLLLoss(_Loss):
    """Gamma distribution model with an Exponetial Dispersion parameterization.
        To model rainfall y
            We predict value p between 0 and 1  .
            If p<0.5, then y=0
            if p>0.5, then we use a Gamma distribution to sample the rainfall
    """
    __constants__ = ['full', 'eps', 'reduction']
    full: bool #Often extra parameters work to regularise the parameters
    eps: float


    def __init__(self, *, full: bool = True, eps: float = 1e-6, reduction: str = 'mean', pos_weight=1 ) -> None:
        super(GammaHurdleNLLLoss, self).__init__(None, None, reduction)
        self.full = full
        self.eps = eps
        self.bce_logits = torch.nn.BCEWithLogitsLoss(reduction='sum')
        self.register_buffer('pos_weight',torch.tensor([pos_weight]) )
        
        assert reduction == 'mean'
    
    def gamma_nll(self, obs,#observations
                    mu,     # mean
                    disp):  # dipersion - \alpha^-1 is dispersion term
        # Uses the Gamma(\mu, \sigma^2) parameterization. Where \mu = \frac{\alpha}{\beta} and \sigma^2 = \frac{1}{\alpha}
        # original parameteriszation was Gamma( \alpha, \beta )
        
        # #initialise S
        # S=-n*(log(gamma(1/d)+(log(d*m)/d)))
        # #itterate through observations
        # for y in obs:
        #     S+=log(y)*((1/d)-1)+(y/(d*m))

        alpha = disp.pow(-1)
        
        ll = (alpha-1)*torch.log(obs) - mu.pow(-1)*obs*alpha
        if self.full:
            ll += alpha*torch.log(alpha) - alpha*torch.log(mu) - torch.lgamma(alpha)

        return -ll


    def forward(self, rain: Tensor, did_rain:Tensor, mean: Tensor, disp: Tensor, logits: Tensor, **kwargs) -> Tensor:
        """
            rain : true rain fall
            did_rain: whether or not it rained
            mean: predicted mean
            disp: predicted dispersion - note dispersion term is the inverse of the \alpha value from the traditional gamma distribution. 
            logits: predicted prob of rain
        """
        if did_rain.dtype != logits.dtype:
            did_rain = did_rain.to(logits.dtype)

        if rain.dtype != mean.dtype:
            rain = rain.to(mean.dtype)

        # Check disp size
        # If disp.size == rain.size, the case is heteroscedastic and no further checks are needed.
        # Otherwise:
        if disp.size() != rain.size():

            # If disp is one dimension short of rain, but the sizes match otherwise, then this is a homoscedastic case.
            # e.g. rain.size = (10, 2, 3), disp.size = (10, 2)
            # -> unsqueeze disp so that disp.shape = (10, 2, 1)
            # this is done so that broadcasting can happen in the loss calculation
            if rain.size()[:-1] == disp.size():
                disp = torch.unsqueeze(disp, -1)

            # This checks if the sizes match up to the final dimension, and the final dimension of disp is of size 1.
            # This is also a homoscedastic case.
            # e.g. rain.size = (10, 2, 3), disp.size = (10, 2, 1)
            elif rain.size()[:-1] == disp.size()[:-1] and disp.size(-1) == 1:  # Heteroscedastic case
                pass

            # If none of the above pass, then the size of disp is incorrect.
            else:
                raise ValueError("disp is of incorrect size")

        # Check validity of reduction mode
        if self.reduction != 'none' and self.reduction != 'mean' and self.reduction != 'sum':
            raise ValueError(self.reduction + " is not valid")

        # Entries of disp must be non-negative
        if torch.any(disp < 0):
            raise ValueError("disp has negative entry/entries")

        # Clamp for stability
        disp = disp.clone()
        rain = rain.clone()
        # mean = mean.clone()
        with torch.no_grad():
            disp.clamp_(min=self.eps)
            rain.clamp_(min=self.eps)
            # mean.clamp_(min=0.5)

        # logits / etc loss
       
        # Calculate the hurdle burnouilli loss
        loss_norain = self.bce_logits( logits, did_rain )

        # Calculate the ll-loss
        indices_rainydays = torch.where(did_rain.view( (1,-1))>0)
        
        rain_rainydays = rain.view((1,-1))[indices_rainydays]
        mean_rainydays = mean.view((1,-1))[indices_rainydays]
        disp_rainydays = disp.view((1,-1))[indices_rainydays]

        # continuous loss 
        loss_rain = self.gamma_nll(obs=rain_rainydays, mu=mean_rainydays, disp=disp_rainydays)

        loss_rain = self.pos_weight * loss_rain
        loss_rain = loss_rain.sum()

        if self.reduction == 'mean':
            loss_norain = loss_norain/rain.numel()
            loss_rain = loss_rain/rain.numel()
            loss = loss_norain + loss_rain 

        return loss, {'loss_norain':loss_norain.detach() , 'loss_rain':loss_rain.detach()}            

class CompoundPoissonGammaNLLLoss(_Loss):
    """The CompoundPoisson Distribution.

        The underlying distribution is Gamma distribution

        We use the Tweedie parameterization that uses a mean and variance concept.
        
        In the Normal CompoundPoissonGamma distribution:
            - there are three parameters: \alpha, \beta, \lambda
            - X is distributed Gamma(\alpha, \beta) - \alpha is the shape, \beta is the rate
            - N is distributed Poisson(\lambda)
            
        In the Tweedie parameterisation:
            - there are three parameters \mu, \theta, \p
                :\mu = \lambda \times \frac{\alpha}{\beta} : can be interpreted as the mean of Y (our target variable)
                :\theta = \lambda * \alpha * (1-\alpha) * \beta^(-2) * \mu^-p : can be interpreted as dispersion
                :\p = \alpha \times (1-\alpha)^-1 and 1<p<2 : can be interpreted as the 'distribution parameters'
            - statistics:
                :
            - note: if \mu ==0, then \lambda ==0
                :  \mu is independent of \lambda 
                : \p interpretation
                    : can be interpreted as p = -( q + 1)
                    : as q tends to 1, \lambda tends to 0?? (i.e. Poisson(0) so the minium amount of samples is chosen)
                    : as q tends to 0, then \alpha (shape of gamm distributions) tends to 0, meaning mode and mean of tend to 0

    """
    __constants__ = ['full', 'eps', 'reduction']
    full: bool
    eps: float

    def __init__(self, *, full: bool = True, eps: float = 1e-6, reduction: str = 'mean', pos_weight=1 ) -> None:
        super(CompoundPoissonGammaNLLLoss, self).__init__(None, None, reduction)
        self.full = full
        self.eps = eps
        self.bce_logits = torch.nn.BCEWithLogitsLoss(reduction='sum')
        self.register_buffer('pos_weight',torch.tensor([pos_weight]) )
        self.register_buffer('pi',torch.tensor(math.pi) )

        
        # Check validity of reduction mode
        if self.reduction != 'none' and self.reduction != 'mean' and self.reduction != 'sum':
            raise ValueError(self.reduction + " is not valid")
        
    
    def nll_zero(self, mu, disp, p ):
        #L=0
        ll = -mu.pow(2-p) * disp.pow(-1) * (2-p).pow(-1)
        return -ll

    def nll_positive(self, obs, mu, disp, p ):
        #L>0
        # using approximation from https://www.hindawi.com/journals/jps/2018/1012647/
        
        beta = (2-p) / (p-1)  #from the gamm distribution
        L = obs
        theta = disp

         

        ll = L * (1-p).pow(-1) * mu.pow(p-1) 
        ll += mu.pow(2-p) * (2-p).pow(-1)
        
        jmax = L.pow(2-p) * (2-p).pow(-1) * theta.pow(-1)
        logW_jmax = jmax * ( torch.log(L**beta) + 
                            torch.log((p-1)**beta) + 
                            -torch.log( (theta)**(1-beta) ) + 
                            -torch.log( (2-p) ) + 
                            (1+beta) - beta*torch.log(beta) + 
                            - (1-beta)*(2-p)*torch.log(L) +
                            - (1-beta)*-torch.log(2-beta) +
                            - (1-beta)*-torch.log(theta)
                        )
                            
        logW_jmax += -torch.log(jmax)

        if self.full:
            logW_jmax += -torch.log(2*self.pi) - 0.5*torch.log(beta)
        
        ll +=  logW_jmax

        return -ll 


    def forward(self, rain: Tensor, did_rain:Tensor, mean: Tensor, disp: Tensor, p: Tensor, **kwargs) -> Tensor:
        
        # Ensuring correct dtypes 
        if did_rain.dtype != p.dtype:
            did_rain = did_rain.to(p.dtype)

        if rain.dtype != mean.dtype:
            rain = rain.to(mean.dtype)

        # Check disp size
            # If disp.size == rain.size, the case is heteroscedastic and no further checks are needed.
            # Otherwise the case is homoscedastic and we need to expand the last dimension
        if disp.size() != rain.size():

            # If disp is one dimension smaller than rain, but the sizes match otherwise, then this is a homoscedastic case.
            # e.g. rain.size = (10, 2, 3), disp.size = (10, 2)
            # -> unsqueeze disp so that disp.shape = (10, 2, 1)
            # this is done so that broadcasting can happen in the loss calculation
            if rain.size()[:-1] == disp.size():
                disp = torch.unsqueeze(disp, -1)

            # This checks if the sizes match up to the final dimension, and the final dimension of disp is of size 1.
            # This is also a homoscedastic case.
            # e.g. rain.size = (10, 2, 3), disp.size = (10, 2, 1)
            elif rain.size()[:-1] == disp.size()[:-1] and disp.size(-1) == 1:  # Heteroscedastic case
                pass

            # If none of the above pass, then the size of disp is incorrect.
            else:
                raise ValueError("disp is of incorrect size")

        # Entries of disp must be non-negative
        if torch.any(disp < 0):
            raise ValueError("disp has negative entry/entries") 

        # Clamping for stability
        disp = disp.clone()
        # rain = rain.clone()
        p = p.clone()
        with torch.no_grad():
            disp.clamp_(min=self.eps)
            # rain.clamp_(min=self.eps)  
            p.clamp_(min=1+self.eps, max=2-self.eps)

        # Gathering indices to seperate days of no rain from days of rain
        count = did_rain.numel()
        indices_rainydays = torch.where(did_rain.view( (1,-1))>0.5)
        indices_non_rainydays = torch.where(did_rain.view( (1,-1))<=0.5)

        
        # Gathering seperate tensors for days with and without rain
        rain_rainydays = rain.view((1,-1))[indices_rainydays]
        mean_rainydays = mean.view((1,-1))[indices_rainydays]
        disp_rainydays = disp.view((1,-1))[indices_rainydays]
        p_rainydays = p.view((1,-1))[indices_rainydays]

        rain_non_rainydays = rain.view((1,-1))[indices_non_rainydays]
        mean_non_rainydays = mean.view((1,-1))[indices_non_rainydays]
        disp_non_rainydays = disp.view((1,-1))[indices_non_rainydays]
        p_non_rainydays = p.view((1,-1))[indices_non_rainydays]

        # Calculating loss for rainy days


        # Calculating loss for non rainy days

        loss_norain = self.nll_zero(mu=mean_non_rainydays, disp=disp_non_rainydays, p = p_non_rainydays  )
        
        loss_rain = self.nll_positive(obs= rain_rainydays, mu=mean_rainydays, disp=disp_rainydays, p=p_rainydays)
        
        loss_rain = self.pos_weight * loss_rain
        loss_rain = loss_rain.sum()
        loss_norain = loss_norain.sum()


        if self.reduction == 'mean':
            loss_norain = loss_norain/count
            loss_rain = loss_rain/count

            loss = loss_rain + loss_norain
                    
        return loss, {'loss_norain':loss_norain.detach() , 'loss_rain':loss_rain.detach()}


#In this version of the loss, the parameters we learn are those of a regular CP model - not the GLM CP model
# class CPNLLLoss(_Loss):
#     __constants__ = ['full', 'eps', 'reduction']
#     full: bool
#     eps: float

#     def __init__(self, *, full: bool = False, eps: float = 1e-6, reduction: str = 'mean', pos_weight=1 ) -> None:
#         super(CPNLLLoss, self).__init__(None, None, reduction)
#         self.full = full
#         self.eps = eps
#         self.bce_logits = torch.nn.BCEWithLogitsLoss(reduction='sum')
#         self.register_buffer('pos_weight',torch.tensor([pos_weight]) )
#         self.register_buffer('pi',torch.tensor(math.pi) )

#         #TODO: later add increased weights to improve classification for rainy days (precision)
        
#         assert reduction == 'mean'

#     #For Compound poisson GLM, the mean is CP and the dispersion is Gamma
#     #The loss for fitting the CP is:

#     def ComPois(a,   #alpha
#             b,   #beta
#             d,   #delta
#             data,#rainfall obs
#             ):
#         mp.dps = 15  #precision for infinite sum
#         n=len(data)  
#         S=-n*d       #initializing 
#         for y in data:
#             if y!=0:
#                 #This expression is defined in terms of regular CP parameters, not mean/dispersion/p as in a CP-GLM.
#                 S+=-(b*y)+log(nsum(lambda z: (((b**(z*a))*(y**(z*a-1))*(d**z))/(gamma(z*a)*factorial(z))), [1, inf]))           
#         return -S

#     #the values maximizing this Loss then need to be used to find the mean and dispersion corresponding to the GLM form of a CP model:
    
#     def CP_param(a,b,d):
#         p=(2+a)/(1+a)
#         miu=(d*a)/b
#         phi=(a+1)/((b**(2-p))*((a*d)**(p-1)))
#         return [miu,phi]

#     def forward(self, input: Tensor, did_rain:Tensor, target: Tensor, var: Tensor, logits: Tensor, **kwargs) -> Tensor:
#         """
#             Input : true rain fall
#             target: predicted mean
#             var: predicted variance
#         """
#         if did_rain.dtype != logits.dtype:
#             did_rain = did_rain.to(logits.dtype)

#         if input.dtype != target.dtype:
#             input = input.to(target.dtype)

#         # Check var size
#         # If var.size == input.size, the case is heteroscedastic and no further checks are needed.
#         # Otherwise:
#         if var.size() != input.size():

#             # If var is one dimension short of input, but the sizes match otherwise, then this is a homoscedastic case.
#             # e.g. input.size = (10, 2, 3), var.size = (10, 2)
#             # -> unsqueeze var so that var.shape = (10, 2, 1)
#             # this is done so that broadcasting can happen in the loss calculation
#             if input.size()[:-1] == var.size():
#                 var = torch.unsqueeze(var, -1)

#             # This checks if the sizes match up to the final dimension, and the final dimension of var is of size 1.
#             # This is also a homoscedastic case.
#             # e.g. input.size = (10, 2, 3), var.size = (10, 2, 1)
#             elif input.size()[:-1] == var.size()[:-1] and var.size(-1) == 1:  # Heteroscedastic case
#                 pass

#             # If none of the above pass, then the size of var is incorrect.
#             else:
#                 raise ValueError("var is of incorrect size")

#         # Check validity of reduction mode
#         if self.reduction != 'none' and self.reduction != 'mean' and self.reduction != 'sum':
#             raise ValueError(self.reduction + " is not valid")

#         # Entries of var must be non-negative
#         if torch.any(var < 0):
#             raise ValueError("var has negative entry/entries")

#         # Clamp for stability
#         var = var.clone()
#         input = input.clone()
#         with torch.no_grad():
#             var.clamp_(min=self.eps)
#             input.clamp_(min=self.eps)

#         # loss
       
#         # Calculate the hurdle burnouilli loss
#         loss_norain = self.bce_logits( logits, did_rain)

#         # Calculate the ll-loss
#         indices_rainydays = torch.where(did_rain.view( (1,-1))>0)
        
#         input_rainydays = input.view((1,-1))[indices_rainydays]
#         target_rainydays = target.view((1,-1))[indices_rainydays]
#         var_rainydays = var.view((1,-1))[indices_rainydays]

#         # Expression for the CP model loss
#         loss_rain = ComPois(a=alpha,b=beta,d=delta,data=input_rainydays)

#             #NOTE(add to papers to write):This initialization with high variance acts as regularizer to smoothen initial loss funciton
#         loss_rain = self.pos_weight * loss_rain
#         loss_rain = loss_rain.sum()

#         if self.reduction == 'mean':
#             loss_norain = loss_norain/input.numel()
#             loss_rain = loss_rain/input.numel()
#             loss = loss_norain + loss_rain 

#         return loss, {'loss_norain':loss_norain.detach() , 'loss_rain':loss_rain.detach()}        

# #In this version, the parameters we learn are those of the CP GLM - mean and dispersion.
# class CPGLMNLLLoss(_Loss):
#     __constants__ = ['full', 'eps', 'reduction']
#     full: bool
#     eps: float

#     def __init__(self, *, full: bool = False, eps: float = 1e-6, reduction: str = 'mean', pos_weight=1 ) -> None:
#         super(CPNLLLoss, self).__init__(None, None, reduction)
#         self.full = full
#         self.eps = eps
#         self.bce_logits = torch.nn.BCEWithLogitsLoss(reduction='sum')
#         self.register_buffer('pos_weight',torch.tensor([pos_weight]) )
#         self.register_buffer('pi',torch.tensor(math.pi) )

#         #TODO: later add increased weights to improve classification for rainy days (precision)
        
#         assert reduction == 'mean'

#     #For Compound poisson GLM, the mean is CP and the dispersion is Gamma
#     #The loss for fitting the CP is:

#     #CP log lokelihood in GLM form
    
#     #first define a value used in the CP-GLM loss clalculations
#     def W_CP(y,a,var,z):
#         return(((y)**(z*a))/((var**(z*(1+a)))*((1/(1+a))**z)*((1/(1+a))**(z*a))*(factorial(z))*(gamma(z*a))))
    
#     def CP_GLM_NLLLoss(a,       #alpha - a Gama dist parameter for both laten variables used in CP
#                         rain_in,   
#                         rain_out,
#                         var):   #The variance of rain data
#         mp.dps = 15  #precision for infinite sum
#         n=len(rain_in)
#         k=np.count_nonzero(a, axis=None, *, keepdims=False)
#         #initialise loss value by adding the terms not dependent on the rain data.
#         S= (n-k)*log(-((1+a)*(rain_out**(a/(1+a))))/(var*a))-(((1+a)*(rain_out**(a/(1+a))))/(var*a))
#         #add the 'sum' part 
#         for y in rain_in:
#             if y!=0:
#                 S+=log(nsum(lambda z:W_CP(y=rain_in,a=a,var=var,z=z)))-log(rain_in)-((rain_in*(1+a))/(var*(rain_out**(1/(1+a)))))
#         return -S

#     def forward(self, input: Tensor, did_rain:Tensor, target: Tensor, var: Tensor, logits: Tensor, **kwargs) -> Tensor:
#         """
#             Input : true rain fall
#             target: predicted mean
#             var: predicted variance
#         """
#         if did_rain.dtype != logits.dtype:
#             did_rain = did_rain.to(logits.dtype)

#         if input.dtype != target.dtype:
#             input = input.to(target.dtype)

#         # Check var size
#         # If var.size == input.size, the case is heteroscedastic and no further checks are needed.
#         # Otherwise:
#         if var.size() != input.size():

#             # If var is one dimension short of input, but the sizes match otherwise, then this is a homoscedastic case.
#             # e.g. input.size = (10, 2, 3), var.size = (10, 2)
#             # -> unsqueeze var so that var.shape = (10, 2, 1)
#             # this is done so that broadcasting can happen in the loss calculation
#             if input.size()[:-1] == var.size():
#                 var = torch.unsqueeze(var, -1)

#             # This checks if the sizes match up to the final dimension, and the final dimension of var is of size 1.
#             # This is also a homoscedastic case.
#             # e.g. input.size = (10, 2, 3), var.size = (10, 2, 1)
#             elif input.size()[:-1] == var.size()[:-1] and var.size(-1) == 1:  # Heteroscedastic case
#                 pass

#             # If none of the above pass, then the size of var is incorrect.
#             else:
#                 raise ValueError("var is of incorrect size")

#         # Check validity of reduction mode
#         if self.reduction != 'none' and self.reduction != 'mean' and self.reduction != 'sum':
#             raise ValueError(self.reduction + " is not valid")

#         # Entries of var must be non-negative
#         if torch.any(var < 0):
#             raise ValueError("var has negative entry/entries")

#         # Clamp for stability
#         var = var.clone()
#         input = input.clone()
#         with torch.no_grad():
#             var.clamp_(min=self.eps)
#             input.clamp_(min=self.eps)

#         # loss
       
#         # Calculate the hurdle burnouilli loss
#         loss_norain = self.bce_logits( logits, did_rain)

#         # Calculate the ll-loss
#         indices_rainydays = torch.where(did_rain.view( (1,-1))>0)
        
#         input_rainydays = input.view((1,-1))[indices_rainydays]
#         target_rainydays = target.view((1,-1))[indices_rainydays]
#         var_rainydays = var.view((1,-1))[indices_rainydays]

#         # Expression for the CP model loss
#         loss_rain = CP_GLM_NLLLoss(a=,rain_in=input_rainydays,rain_out=target_rainydays,var=var_rainydays)

#         #NOTE(add to papers to write):This initialization with high variance acts as regularizer to smoothen initial loss funciton
#         loss_rain = self.pos_weight * loss_rain
#         loss_rain = loss_rain.sum()

#         if self.reduction == 'mean':
#             loss_norain = loss_norain/input.numel()
#             loss_rain = loss_rain/input.numel()
#             loss = loss_norain + loss_rain 

#         return loss, {'loss_norain':loss_norain.detach() , 'loss_rain':loss_rain.detach()}
