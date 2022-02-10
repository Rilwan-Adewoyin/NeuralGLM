from torch.distributions.distribution import Distribution
from torch.distributions.gamma import Gamma
from torch.distributions.half_normal import HalfNormal
from torch.distributions.log_normal import LogNormal
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.poisson import Poisson

from torch.distributions.transformed_distribution import TransformedDistribution

from torch.distributions import constraints
import torch

LIST_PVar_Model = ["lognormal_hurdle","gamma_hurdle","compound_poisson"] #List of models that use the p variable

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
        rain_prob = self.bernoulli_dist.sample( sample_size )

        sampled_rain = torch.where(rain_prob>=0.5, self.lognormal_dist.sample( (1,) ), 0  )

        return sampled_rain
    

class GammaHurdle():
    r"""
    Creates a Gamma Hurdle distribution parameterized by
    :attr:`loc` and :attr:`scale` where and :attr:`p` :

        p ~ Bernoulli( p )
        Y ~ Gamma( mu, disp) ~

    
    """
    arg_constraints = {'mu': constraints.positive, 'disp': constraints.positive, 'prob':constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(self, mu, disp, prob, validate_args=None):
        
        self.bernoulli_dist = Bernoulli(prob, validate_args=validate_args)

        alpha, beta = self.reparameterize( mu, disp )

        self.gamma_dist = LogNormal(alpha, beta, validate_args=validate_args)

    def sample(self,sample_size=(1,)):
        
        rain_prob = self.bernoulli_dist.sample( sample_size )

        sampled_rain = torch.where(rain_prob>=0.5, self.gamma_dist.sample( (1,) ), 0  )

        return sampled_rain

    def reparameterize(mu, disp):
        # Converts from the \mu, \sigma^2 parameterization to the \alpha , \beta parameterization

        alpha = 1/disp
        beta = alpha/mu

        return alpha, beta

class CompoundPoisson():

    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive}
    support = constraints.positive
    has_rsample = True

    def __init__(self, mu, disp, p, validate_args=None):

        lambda_, alpha, beta = self.reparameterize( mu, disp, p)
        
        self.poisson_dist = Poisson(lambda_, validate_args=validate_args)
        self.gamma_dist = LogNormal(alpha, beta, validate_args=validate_args)

    def sample(self,sample_size=(1,)):

        N = self.poisson_dist.sample( sample_size )


        li_gammas = [ self.gamma_dist.sample( sample_size ) for i in torch.arange(N)]
        rain = torch.stack( li_gammas, dim=0).sum(dim=0)

        return rain
    
    def reparameterize(self, mu, disp, p):
        # Convert from ED form to standard form

        lambda_ = mu.pow(2-p) * ( disp*(2-p) ).pow(-1)
        alpha = disp*(p-1)*mu.pow(p-1)
        beta = (2-p)/(p-1)

        return lambda_, alpha, beta
    
    