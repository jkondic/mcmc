from abc import ABC, abstractmethod
import numpy as np
from jax import grad
import sys
import mh, hmc, mala

class Distribution(ABC):
    'An abstract class initializing a distribution'
    
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def sample(self):
        'sample from the distribution'
        
    @abstractmethod
    def logpdf(self, x):
        'compute the log of the distribution'

        
class Normal(Distribution):
    'Gaussian distribution, f(x) = e^(-(x-mu)^2/2*sigma^2)/sigma*sqrt(2*pi)'

    # sigma is assumed to be a scalar, so that the covariance matrix C = sigma*I
    def __init__(self, mu, sigma=1):
        self.mu = mu
        self.sigma = sigma
        # extract dimension from mu
        self.d = mu.size
        super().__init__()
    
    def sample(self):
        return np.random.normal(self.mu, self.sigma)

    def logpdf(self, x):
        return -0.5 * np.log(2*np.pi)*self.d - np.log(self.sigma) - np.sum((x - self.mu)**2) / (2*self.sigma**2)

class ShiftedNormal(Distribution):
    'A normal distribution with a shifted mean; uses just the mean as arg'

    # mean can be any shape tensor, but sigma is assumed to be a scalar
    def __init__(self, sigma):
        self.sigma = sigma
        super().__init__()
            
    def sample(self, mu):
        # sample from normal distribution (works for any shaped input; shape will match that of mu)
        return mu + np.random.randn(*mu.shape) * self.sigma

    def logpdf(self, x, mu):
        return -0.5 * np.log(2*np.pi) - np.log(self.sigma) - np.sum((x - mu)**2) / (2*self.sigma**2)
    
class SumDistribution(Distribution):
    'A distribution of the sum of distributions/ r.v.s passed to it; can be equivalent to a mixture of Gaussians'
    
    def __init__(self, *distributions):
        self.distributions = distributions
        self.n = len(distributions)
        super().__init__()
        
    def sample(self, *args):
        'randomly choose one of the distributions and sample from it'
        d = np.choice(self.distributions)
        return d.sample(*args)
    
    def logpdf(self, x, *args):
        'compute the log of the sum of the input distributions'
        # no need to normalize for MCMC in general
        return np.log(np.sum(np.exp(d.logpdf(x, *args) - np.log(self.n))
                            for d in self.distributions))

class Boltzmann(Distribution):
    'A standard Boltzmann distribution, p(x) = e^{-beta * cost(x)}'

    # cost is a user-defined function that takes as input a tensor x of shape (shape) and returns a scalar
    def __init__(self, shape, cost, sampler, beta=1):
        self.shape = shape
        self.cost = cost
        self.beta = beta
        self.sampler = sampler
        self.grad_log = grad(self.logpdf)
        super().__init__()
    

    # n is number of samples, and burn_is is the number of samples to ignore at beginning before convergence
    def sample(self, n = 1, burn_in = 100):
        'draw n samples from the Boltzmann distribution using Metropolis-Hastings.'
        
        if self.sampler == 'mh':
            # initialize sampler object
            sampler = mh.MetropolisHastings(target=self, proposal=ShiftedNormal(sigma=1), x0=np.zeros(self.shape), n=n+burn_in)

            # run Metropolis-Hastings
            sampler.run()

            # TESTING: print acceptance rate
            print("ACCEPTANCE RATE: ", sampler.acceptance_rate)

            # return result, ignoring first (burn_in) samples
            return sampler.x[burn_in:]
        
        elif self.sampler == 'hmc':
            # initialize sampler object
            sampler = hmc.HamiltonianMonteCarlo(target=self, x0=np.zeros(self.shape), n=n+burn_in, L=3, p=0.14)

            # run Metropolis-Hastings
            sampler.run()

            # TESTING: print acceptance rate
            print("ACCEPTANCE RATE: ", sampler.acceptance_rate)

            # return result, ignoring first (burn_in) samples
            return sampler.x[burn_in:]

        elif self.sampler == 'mala':
            # initialize sampler object
            sampler = mala.MALA(target=self, x0=np.zeros(self.shape), n=n+burn_in, tau=0.5)

            # run Metropolis-Hastings
            sampler.run()

            # TESTING: print acceptance rate
            print("ACCEPTANCE RATE: ", sampler.acceptance_rate)

            # return result, ignoring first (burn_in) samples
            return sampler.x[burn_in:]
        
        else:
            sys.exit('ERRROR: Invalid sampler specified')
            
    def logpdf(self, x):
        return -self.beta*self.cost(x)