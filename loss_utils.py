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
import numpy as np

# Distributional Losses Torch Classes
class LogNormalHurdleNLLLoss(_Loss):
    """A log normal distribution LN(\mu, disp) with \mu,disp \in [0, \inf] 

    """
    __constants__ = ['full', 'eps', 'reduction']
    full: bool
    eps: float

    def __init__(self, *, full: bool = False, eps: float = 1e-6, reduction: str = 'mean', pos_weight=1, **kwargs ) -> None:
        super(LogNormalHurdleNLLLoss, self).__init__(None, None, reduction)
        self.full = full
        self.eps = eps
        self.register_buffer('pos_weight',torch.tensor([pos_weight]))
        self.register_buffer('pi',torch.tensor(math.pi))
        self.bce_logits = torch.nn.BCEWithLogitsLoss(reduction='sum', pos_weight=self.pos_weight)
        assert reduction == 'mean'
    
    def lognormal_nll(self, obs, mu, disp ):

  
        logobs = torch.log(obs)
        logobs = logobs.clone()
        with torch.no_grad():
            if logobs.numel()>0:
                logobs.clamp_(min=mu.min()) 
            
        ll  = -0.5 * (torch.log(disp) + (logobs - mu )**2 / disp ) 
                        
        if self.full:
            ll += -0.5*torch.log(2*self.pi) - logobs

        return -ll

    def forward(self, rain: Tensor, did_rain:Tensor, mu: Tensor, disp: Tensor, logits: Tensor, **kwargs) -> Tensor:
        """
            rain : true rain fall
            did_rain: whether or not it rained
            mu: predicted mu
            disp: predicted dispiance
            logits: predicted prob of rain
        """
        if did_rain.dtype != logits.dtype:
            did_rain = did_rain.to(logits.dtype)

        if rain.dtype != mu.dtype:
            rain = rain.to(mu.dtype)

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

        # logits / etc loss
       
        # Calculate the hurdle burnouilli loss
        loss_norain = self.bce_logits( logits, did_rain)

        # Calculate the ll-loss
        indices_rainydays = torch.where( did_rain.view( (1,-1))==1 )
        

        rain_rainydays = rain.view((1,-1))[indices_rainydays]
        mu_rainydays = mu.view((1,-1))[indices_rainydays]
        disp_rainydays = disp.view((1,-1))[indices_rainydays]

        loss_rain = self.lognormal_nll(rain_rainydays, mu_rainydays, disp_rainydays)

        loss_rain = loss_rain.sum()
        loss_rain = self.pos_weight * loss_rain

        if self.reduction == 'mean':
            loss_norain = loss_norain/rain.numel()
            loss_rain = loss_rain/rain.numel()
            loss = loss_norain + loss_rain 

        return loss, {'loss_norain':loss_norain.detach() , 'loss_rain':loss_rain.detach()}            

    def prediction_metrics(self, rain, did_rain, mean, logits, **kwargs):
        if mean.numel() == 0:
            return {
                'pred_acc':mean.new_tensor(0.0),
                'pred_rec':mean.new_tensor(0.0),
                'pred_mse':mean.new_tensor(0.0),
                'pred_r10mse':mean.new_tensor(0.0)

            }
        #Classification losses
        pred_rain_bool = torch.where( logits>=0.0, 1.0, 0.0)
        pred_acc = torch.mean( pred_rain_bool*did_rain + (pred_rain_bool-1)*(did_rain-1) )
        pred_rec = (did_rain*pred_rain_bool).sum() / did_rain.sum()

        #Continuous losses
        indices_rainydays = torch.where(did_rain.view( (1,-1))>0)
        
        pred_mse = torch.nn.functional.mse_loss(
            mean.view((1,-1))[indices_rainydays],
            rain.view((1,-1))[indices_rainydays]
        ) 


        indices_r10days = torch.where(rain.view( (1,-1))>10)
        pred_r10mse = torch.nn.functional.mse_loss(
            mean.view((1,-1))[indices_r10days],
            rain.view((1,-1))[indices_r10days]
        )

        #MSE on days it did rain
        
        pred_metrics = {'pred_acc': pred_acc.detach(),
                        'pred_rec':pred_rec.detach(),
                            'pred_mse': pred_mse.detach(),
                            'pred_r10mse': pred_r10mse.detach()
 }
        return pred_metrics

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

    def __init__(self, *, full: bool = True, eps: float = 1e-6, reduction: str = 'mean', pos_weight=1, **kwargs ) -> None:
        super(GammaHurdleNLLLoss, self).__init__(None, None, reduction)
        self.full = full
        self.eps = eps
        self.register_buffer('pos_weight',torch.tensor([pos_weight]) )
        self.bce_logits = torch.nn.BCEWithLogitsLoss(reduction='sum', pos_weight=self.pos_weight)
        
        assert reduction == 'mean'
    
    def gamma_nll(self, obs,#observations
                    mu,     # mu
                    disp):  # dipersion - \alpha^-1 is dispersion term
        # Uses the Gamma(\mu, \sigma^2) parameterization. Where \mu = \frac{\alpha}{\beta} and \sigma^2 = \frac{1}{\alpha}
        # original parameteriszation was Gamma( \alpha, \beta )

        d = disp
        # Clamp for stability
        obs = obs.clone()
        with torch.no_grad():
            obs.clamp_(min=self.eps)

        ll = (d.pow(-1)-1)*torch.log(obs) - obs*mu.pow(-1)*d.pow(-1) - torch.lgamma(d.pow(-1)) - d.pow(-1) * torch.log(d*mu)
                   
        return -ll

    def forward(self, rain: Tensor, did_rain:Tensor, mu: Tensor, disp: Tensor, logits: Tensor, **kwargs) -> Tensor:
        """
            rain : true rain fall
            did_rain: whether or not it rained
            mu: predicted mu
            disp: predicted dispersion - note dispersion term is the inverse of the \alpha value from the traditional gamma distribution. 
            logits: predicted prob of rain
        """
        if did_rain.dtype != logits.dtype:
            did_rain = did_rain.to(logits.dtype)

        if rain.dtype != mu.dtype:
            rain = rain.to(mu.dtype)

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

        # Calculate the hurdle burnouilli loss
        loss_norain = self.bce_logits( logits, did_rain )

        # Calculate the ll-loss
        indices_rainydays = torch.where(did_rain.view( (1,-1))>0)
        
        rain_rainydays = rain.view((1,-1))[indices_rainydays]
        mu_rainydays = mu.view((1,-1))[indices_rainydays]
        disp_rainydays = disp.view((1,-1))[indices_rainydays]

        # continuous loss 
        loss_rain = self.gamma_nll(obs=rain_rainydays, mu=mu_rainydays, disp=disp_rainydays)
        loss_rain = loss_rain.sum()
        loss_rain = self.pos_weight * loss_rain

        if self.reduction == 'mean':
            loss_norain     = loss_norain/rain.numel()
            loss_rain = loss_rain/rain.numel()
            loss = loss_norain + loss_rain 
        
        return loss, {'loss_norain':loss_norain.detach() , 'loss_rain':loss_rain.detach()}            

    def prediction_metrics(self, rain, did_rain, mean, logits, **kwargs):
        if mean.numel() == 0:
            return {
                'pred_acc':None,
                'pred_rec':None,
                'pred_mse':None,
                'pred_r10mse':None
            }

        pred_rain_bool = torch.where( logits>=0.0, 1.0, 0.0)
        pred_acc = torch.mean(  torch.where( pred_rain_bool==did_rain, 1.0, 0.0) ).detach()
        pred_rec = (did_rain*pred_rain_bool).sum() / did_rain.sum() if did_rain.sum()>0 else torch.as_tensor(1.0, device=pred_rain_bool.device).detach()

        indices_rainydays = torch.where(did_rain==1.0)

        if indices_rainydays[0].numel()>0.5:
            pred_mse = torch.nn.functional.mse_loss(
                mean[indices_rainydays],
                rain[indices_rainydays]
                ).detach() #MSE on days it did rain
        else:
            pred_mse = None
        
        indices_r10days = torch.where(rain>10.0)
        if indices_r10days[0].numel()>0:
            pred_r10mse = torch.nn.functional.mse_loss(
                mean[indices_r10days],
                rain[indices_r10days]
            ).detach()
        else:
            pred_r10mse = None

        pred_metrics = {'pred_acc': pred_acc,
                            'pred_rec': pred_rec,
                            'pred_mse': pred_mse,
                            'pred_r10mse': pred_r10mse

                             }
        return pred_metrics
        
class CompoundPoissonGammaNLLLoss(_Loss):
    """The CompoundPoisson Distribution.

        The underlying distribution is Gamma distribution

        We use the Tweedie parameterization that uses a mu and variance concept.
        
        In the Normal CompoundPoissonGamma distribution:
            - there are three parameters: \alpha, \beta, \lambda
            - X is distributed Gamma(\alpha, \beta) - \alpha is the shape, \beta is the rate
            - N is distributed Poisson(\lambda)
            
        In the Tweedie parameterisation:
            - there are three parameters \mu, \disp, \p
                :\mu = \lambda \times \frac{\alpha}{\beta} : can be interpreted as the mu of Y (our target variable)
                :\disp = \lambda * \alpha * (1-\alpha) * \beta^(-2) * \mu^-p : can be interpreted as dispersion
                :\p = \alpha \times (1-\alpha)^-1 and 1<p<2 : can be interpreted as the 'distribution parameters'
            - statistics:
                :
            - note: if \mu ==0, then \lambda ==0
                :  \mu is independent of \lambda 
                : \p interpretation
                    : can be interpreted as p = -( q + 1)
                    : as q tends to 1, \lambda tends to 0?? (i.e. Poisson(0) so the minium amount of samples is chosen)
                    : as q tends to 0, then \alpha (shape of gamm distributions) tends to 0, muing mode and mean of tend to 0

    """
    __constants__ = ['full', 'eps', 'reduction']
    full: bool
    eps: float

    def __init__(self, *, full: bool = True, eps: float = 1e-6, reduction: str = 'mean', pos_weight=1, **kwargs ) -> None:
        super(CompoundPoissonGammaNLLLoss, self).__init__(None, None, reduction)
        self.full = full
        self.eps = torch.finfo(torch.float32).eps
        self.bce_logits = torch.nn.BCEWithLogitsLoss(reduction='sum')
        self.register_buffer('pos_weight',torch.tensor([pos_weight]) )
        self.register_buffer('pi',torch.tensor(math.pi) )
        self.register_buffer('e',torch.tensor(math.e) )
        self.tblogger = kwargs.get('tblogger',None)
        
        self.j_window_size = kwargs.get('j_window_size', 24)
        self.register_buffer('j_window_size_float', torch.as_tensor( kwargs.get('j_window_size', 24.0), dtype=torch.float) )
        self.register_buffer('j_window', torch.arange(start=-self.j_window_size+1, end=self.j_window_size, step=1, dtype=torch.float, requires_grad=False) )

        self.approx_method = kwargs.get('approx_method', 'gosper')
        
        # Check validity of reduction mode
        if self.reduction not in  ['none','mean','sum']:
            raise ValueError(self.reduction + " is not valid")

    def gosper_gamma(self, x):
        x = x - 1 
        outp = torch.sqrt((2*x + 1/3) * self.pi) * ((x/self.e) ** x)
        return outp
    
    def log_gosper_gamma(self, x):
        x = x-1
        outp = 0.5*torch.log(2*x + 1/3)  + 0.5*torch.log(self.pi) + x*torch.log(x) - x
        return outp

    def nll_zero(self, mu, disp, p ):
        #y=0
        ll = -mu.pow(2-p) * disp.pow(-1) * (2-p).pow(-1)

        # log_ll = (2-p)*torch.log(-mu) + -1*torch.log(disp) + -1*torch.log(2-p)

        # ll = torch.exp(log_ll)

        return -ll

    def nll_positive(self, obs, mu, disp, p, **kwargs  ):
        """Negative log likelihood of the CompoundPoissonGamma distribution for positive observations."""
        
        y = obs        

        # # Poisson and Gamma params
        # lambda_ = l = mu.pow(2-p) / ( disp * (2-p) )
        alpha = a = (2-p) / (p-1)  #from the gamma distribution
        # beta = b = mu.pow(p-1) / ( disp*(p-1) )

        # Tweedie Exponetial Dispersion Family params
        theta = mu.pow(1-p)/(1-p) #can be negative
        kappa = mu.pow(2-p)/(2-p)


        # LogLikelihood for CompoundPoissonGamma distribution
        # # log(f) = log( a(y,\disp) ) + 1/\disp * [y*theta - kappa] 
        # # log( a(y,\disp) ) = A + B = log( \sum(W_j) ) + log(1/y)
        # # 1/\disp * [y*theta - kappa]  = C * D
        # # C = 1/disp
        # # D = D1 + D2 =y*theta - kappa
        # # log(f) = (A + B ) + C*(D1+D2)

        # Calculating jmax which we create a window around            
        log_jmax = (2-p)*torch.log(y) + -1*torch.log(2-p) + -1*torch.log(disp)  

        # Calculating priors for the activation function to use for mu, y, disp
        # # Ideally want jmax between 1 and 40 w/ starting value of 7.0-> log_jmax between 0 and 3.7 w/starting value of 1.9
        # # if 0<y<2, 1<p<2 then y.pow(2-p) is between 0 and 4
        # # (2-p).pow(-1) is between 1 and \infty
        # # If scaled target is 0<y<2
        # # We want mu to be between 0 and 1, e.g. starting value of 1.0
        # # TO get log_jmax=1.9 w/ y = 1.0 and p = 1.5 then disp = 0.30
        
        jmax = torch.exp(log_jmax)
        half_window_size = self.j_window_size // 2

        jmax_adj = torch.where(jmax<half_window_size+1,  jmax+(half_window_size+1 - torch.floor(jmax) ) , jmax )
        jmax_adj = jmax_adj.clamp(max=70.0)
    
        jmax_adj: Tensor = jmax_adj[:, None].expand( jmax_adj.numel(), (self.j_window_size) ) #expanding
        j_window = torch.arange(-half_window_size, half_window_size, dtype=jmax_adj.dtype, device=jmax_adj.device)
        j = (jmax_adj + j_window).transpose(0,1) # This is a range of j for each index
        j = torch.clamp(j, min=1.0) # ensure no value is less than 0

        
        # TODO: Try to use pytorch lgamma distribution  
        # TODO: Check the significance of the terms of the terms approximation 
        # TODO: Plot the terms which are contributing to the overflow

        #Therefore we used the improved GammaFunc approximation proposed by Gosper
        if self.approx_method == 'gosper':
            #formula: Wj = y.pow(j*-a) * (p-1).pow(j*a) * disp.pow(-j*(1-a)) * (2-p).pow(-j) * (j!).pow(-1) * (\GammaFunc(-j*a)).pow(-1)
                # (\GammaFunc(-j*a)) = gosper(-j*a - 1)

            # Original implmentation
            # Wj_1 = y.pow(j*-a) * (p-1).pow(j*a) * disp.pow(-j*(1-a)) * (2-p).pow(-j) 
            # Wj_2 = self.gosper_gamma(j+1) # Use Gosper approximation for j!)
            # Wj_3 = self.gosper_gamma(-j*a) # Use Gosper approximation for \GammaFunc(-j*a)      
            # A = ( Wj_1 * Wj_2 * Wj_3).sum(dim=0)

            # Stable implementation
            # formula: A= (j * (-ja-1).pow(-a) * y.pow(-a) * disp.pow(1-a) * e.pow(1-a) ).pow(j) x ( (2j+1/3)*pi ).pow(1/2) * ((2(-ja-1)+1/3)*pi).pow(1/2)
            # A = A1 + A2 + A3
            # A1 = (j * (-ja-1).pow(-a) * y.pow(-a) * disp.pow(1-a) * e.pow(1-a) ).pow(j)
            # A2 = ( (2j+1/3)*pi ).pow(1/2)
            # A3 = ((2(-ja-1)+1/3)*pi).pow(1/2)
            # A4 = e/(-ja-1)
            
            # rearranged for stability
            # B formula ordered: B=B1*B2*B3
            # B1.pow(1/j) = (y/disp).pow(-a) 
            # B2.pow(1/j) =  ((-ja-1)/e).pow(-a) 
            # B3.pow(1/j) = j / (disp * e)

            A11 = (y/disp).pow(-a)
            _ = (((-j*a)-1)/self.e)
            A12 = torch.where(_>0, _, _.abs() ).pow(-a)
            
            A13 = j / (disp * self.e)
            A1 = (A11*A12*A13).pow(j)

            A4 = ( (2*j+1/3)*self.pi ).pow(1/2)
            # Ignore negative values for D
            _D = ((2*(-j*a)-1)/3*self.pi)
            D = torch.where(_D>0, _D, _D.abs() ) .pow(1/2)

            E = self.e / ((-j*a)-1)

            A = B * C * D * E   

            A = A.sum(dim=0)

            A = torch.log(A)

            # A = torch.log(B) + torch.log(C) + torch.log(D) + torch.log(E)

        elif self.approx_method == 'jensen_lanczos':
            logWj_1 = (-j*a)*torch.log(y) + (a*j)*torch.log(p-1) + (-j*(1-a))*torch.log(disp) + -j*torch.log(2-p)
            logWj_2 = - torch.lgamma(j+1) 
            logWj_3 = - torch.lgamma(-j*a)

            A = ( logWj_1 + logWj_2 + logWj_3 ).sum(dim=0)
        
        elif self.approx_method == 'jensen_gosper':
            logWj_1 = (-j*a)*torch.log(y) + (a*j)*torch.log(p-1) + (-j*(1-a))*torch.log(disp) + -j*torch.log(2-p)
            logWj_2 = - self.log_gosper_gamma(j+1) 
            logWj_3 = - self.log_gosper_gamma(-j*a)

            A = ( logWj_1 + logWj_2 + logWj_3 ).sum(dim=0)

        
        B = -1*torch.log(y)

        C = 1/disp
        D1 = y*theta
        D2 = -kappa
        
        ll = (A + B) + C*(D1+D2) #NOTE: NLL should be negative

        return -ll 

    def forward(self, rain: Tensor, did_rain:Tensor, mu: Tensor, disp: Tensor, p: Tensor, **kwargs) -> Tensor:
        
        # Ensuring correct dtypes 
        if did_rain.dtype != p.dtype:
            did_rain = did_rain.to(p.dtype)

        if rain.dtype != mu.dtype:
            rain = rain.to(mu.dtype)

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

        # Clamping dispersion and p for stability
        disp = disp.clone()
        p = p.clone()
        with torch.no_grad():
            if disp.numel()>0:
                disp.clamp_(min=self.eps)
                p.clamp_(min=1+self.eps, max=2-self.eps)

        # Gathering indices to seperate days of no rain from days of rain
        count = did_rain.numel()

        # Rainy days juded as 
        indices_rainydays = torch.where(did_rain.view( (1,-1))>0)
        indices_non_rainydays = torch.where(did_rain.view( (1,-1))==0)

        
        # Gathering seperate tensors for days with and without rain
        rain_rainydays = rain.view((1,-1))[indices_rainydays]
        mu_rainydays = mu.view((1,-1))[indices_rainydays]
        disp_rainydays = disp.view((1,-1))[indices_rainydays]
        p_rainydays = p.view((1,-1))[indices_rainydays]

        rain_non_rainydays = rain.view((1,-1))[indices_non_rainydays]
        mu_non_rainydays = mu.view((1,-1))[indices_non_rainydays]
        disp_non_rainydays = disp.view((1,-1))[indices_non_rainydays]
        p_non_rainydays = p.view((1,-1))[indices_non_rainydays]

        # Calculating loss for non rainy days
        loss_norain = self.nll_zero(mu=mu_non_rainydays, disp=disp_non_rainydays, p = p_non_rainydays  )
        
        # Calculating loss for rainy days
        loss_rain = self.nll_positive(obs= rain_rainydays, mu=mu_rainydays, disp=disp_rainydays, p=p_rainydays, global_step=kwargs.get('global_step') )
        

        loss_rain = self.pos_weight * loss_rain
        loss_rain = loss_rain.sum()
        loss_norain = loss_norain.sum()


        if self.reduction == 'mean':
            loss_norain = loss_norain/count
            loss_rain = loss_rain/count

            loss = loss_rain + loss_norain
                    
        return loss, {'loss_norain':loss_norain.detach() , 'loss_rain':loss_rain.detach()}

    def prediction_metrics(self, rain, did_rain, mean, min_rain_value, **kwargs):

        if mean.numel() == 0:
            return {
                'pred_acc':mean.new_tensor(0.0),
                'pred_rec':mean.new_tensor(0.0),
                'pred_mse':mean.new_tensor(0.0),
                'pred_r10mse':mean.new_tensor(0.0)
            }
            
        pred_rain_bool = torch.where( mean>min_rain_value, mean.new_tensor(1.0), mean.new_tensor(0.0))
        pred_acc = torch.mean(  torch.where( pred_rain_bool==did_rain, mean.new_tensor(1.0), mean.new_tensor(0.0)) )
        pred_rec = (did_rain*pred_rain_bool).sum() / did_rain.sum()  if did_rain.sum()>0 else mean.new_tensor(1)
       
        
        indices_rainydays = torch.where(did_rain.view( (1,-1))>0)

        # Note: we unscale the value prior to caluclating mse to allow for comparison between models
        pred_mse = torch.nn.functional.mse_loss(
            mean.view((1,-1))[indices_rainydays],
            rain.view((1,-1))[indices_rainydays]
        ) #MSE on days it did rain

        indices_r10days = torch.where(rain.view( (1,-1))>10)
        pred_r10mse = torch.nn.functional.mse_loss(
            mean.view((1,-1))[indices_r10days],
            rain.view((1,-1))[indices_r10days]
        )

        
        pred_metrics = {'pred_acc': pred_acc.detach(),
                            'pred_rec': pred_rec.detach(),
                            'pred_mse': pred_mse.detach(),
                            'pred_r10mse': pred_r10mse.detach()
                             }
        return pred_metrics
    
# Wasserstein, CRPS,MSE

class VAEGANLoss():
    
    def __init__( self,
        content_loss = True,
        content_loss_name = None, #'ensmeanMSE_phys' #TODO: check which one is used ,
        enemble_size = None,
        kl_weight= None, #TODO: find the kl weight
        cl_weight= None,  #TODO: 
        gp_weight=None
        ):
        
        self.use_content_loss = content_loss
        self.enemble_size = enemble_size
        self.kl_weight = kl_weight
        self.cl_weight = cl_weight
        self.gp_weight = gp_weight
        
        self.content_loss_name = content_loss_name
        self.content_loss_fct = None if self.content_loss_name is None else CL_chooser(self.content_loss_name)
         
    def __call__(self, score, score_pred, z_mean=None, z_logvar=None, mask=None,
                 net=None, constant_field=None ):
        
        raise NotImplementedError
        return None
    
    def wasserstein(self, score, score_pred):
        return torch.mean(score*score_pred) #wasserstein 
    
    def kl_loss(self, z_logvar, z_mean, batch_size, mask=None ):
                
        if mask is None:
            kl_loss = 0.5 * (-1 - z_logvar + z_mean.pow(2) + torch.exp(z_logvar) )
            kl_loss = torch.reshape(kl_loss, [batch_size, -1])
            kl_loss = kl_loss.sum(1).mean()
            
        else:
            #Pooling needs to be aware of the mask
            #Mask areas should be ignored from kl average
            mask_pooled = torch.nn.functional.avg_pool2d(mask.to(torch.float), (5,7), stride=(5,7), count_include_pad=False  )            
            mask_pooled = mask_pooled[:,None]
            mask_pooled = torch.where( mask_pooled>=0.5, torch.ones_like(z_mean), torch.zeros_like(z_mean) )
            masked_elems_per_batch = mask_pooled.sum(dim=(-3, -2, -1))
            
            kl_loss = 0.5 * (-1 - z_logvar + z_mean.pow(2) + torch.exp(z_logvar) )
            kl_loss = kl_loss*mask_pooled      
            
            
            kl_loss = kl_loss.sum((-3,-2,-1)) / masked_elems_per_batch
            kl_loss = kl_loss.mean()
        
        return self.kl_weight*kl_loss
        
    def wasserstein_loss(self, y_true, y_pred):
        return torch.mean(y_true * y_pred, dim=-1)

    def content_loss(self, z_mean, z_logvar, constant_fields, net, mask, score):
        score_pred_ = [ net.decoder( z_mean, z_logvar, constant_fields ) for ii in range(self.ensemble_size) ]
        score_pred_ = torch.stack(score_pred_, axis=-1)  #  batch x W x H x 1 x ens
        score_pred_ = score_pred_.squeeze(-2)  #  batch x W x H x ens
        
        score = torch.masked_select( score, (mask ) )
        score_pred_ = torch.masked_select(score_pred_, (mask ) )
        
        content_loss  = self.content_loss_fct(score, score_pred_)
        content_loss = content_loss.sum()/self.ensemble_size
        return content_loss

    def gradient_penalty(self, discriminator, real_image, fake_image, disc_args, scaler=None):
        b, *_ = real_image.shape
        device = real_image.device
        
        alpha = torch.rand(b, 1, 1, 1).to(real_image.device)
        alpha = alpha.expand_as(real_image)


        interpolated = alpha*real_image + (1-alpha)*fake_image
        interpolated.requires_grad = True

        # Calculate probability of interpolated examples
        prob_interpolated = discriminator(interpolated, *disc_args)
        
         # Calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                               create_graph=True, retain_graph=True)[0]
        
                # Gradients have shape (batch_size, num_channels, img_width, img_height),
        
        gradients = gradients/scaler.get_scale() if scaler is not None else gradients
        
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(b, -1)
               
        
        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        # gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1))
        
        gradients_norm = ((gradients_norm-1)**2).mean()
        
        # Return gradient penalty
        return self.gp_weight * gradients_norm
        
        
        



def denormalise(y_in):    
    return torch.pow(10,y_in) - 1 

def sample_crps(y_true, y_pred):
    raise NotImplementedError

    # mae = torch.abs((y_true.unsqueeze(0)- y_pred.unsqueeze(-1))).mean()
    mae = torch.abs((y_true.unsqueeze(-1)- y_pred)).mean()
    ensemble_size = y_pred.size(-1)
    coef = -1/(2*ensemble_size * (ensemble_size - 1))
    
    # ens_var = coef * tf.reduce_mean(tf.reduce_sum(tf.abs(tf.expand_dims(y_pred, axis=0) - tf.expand_dims(y_pred, axis=1)),
    
    #                                               axis=(0, 1)))

    ens_var =  coef * torch.abs(y_pred.unsqueeze(0)-y_pred.unsqueeze(1)).sum((0,1)).mean()
    return mae + ens_var

def sample_crps_phys(y_true, y_pred):
    y_true = denormalise(y_true)
    y_pred = denormalise(y_pred)
    return sample_crps(y_true, y_pred)

def ensmean_MSE(y_true, y_pred):
    pred_mean = y_pred.mean(0)
    y_true_squ = y_true.squeeze(-1)
    return F.mse_loss(y_true_squ, pred_mean)

def ensmean_MSE_phys(y_true, y_pred):
    y_true = denormalise(y_true)
    y_pred = denormalise(y_pred)
    return ensmean_MSE(y_true, y_pred)

def CL_chooser(CLtype):
    return {"CRPS": sample_crps,
            "CRPS_phys": sample_crps_phys,
            "ensmeanMSE": ensmean_MSE,
            "ensmeanMSE_phys": ensmean_MSE_phys}[CLtype]
