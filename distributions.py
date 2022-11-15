from abc import ABC, abstractmethod
import numpy as np

class Distribution(ABC):
    'An abstract class initializing a distribution'
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def sample(self, mu):
        'sample from the distribution'
        
    @abstractmethod
    def logpdf(self, x, mu):
        'compute the log of the distribution'


class SumDistribution(Distribution):
    'A distribution of the sum of distributions/ r.v.s passed to it'
    'can be equivalent to a mixture of Gaussians'
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
        
class Normal(Distribution):
    'Gaussian distribution, f(x) = e^(-(x-mu)^2/2*sigma^2)/sigma*sqrt(2*pi)'
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        super().__init__()
    
    def sample(self, mu=None):
        return np.random.normal(self.mu, self.sigma)

    def logpdf(self, x, mu=None):
        return -0.5 * np.log(2*np.pi) - np.log(self.sigma) - (x - self.mu)**2 / (2*self.sigma**2)
    
class Uniform(Distribution):
    'Rectangular distribution, f(x) = 1/(b-a) for x in [a,b] and 0 otherwise'
    def sample(self, mu=None):
        return np.random.uniform()

class ShiftedNormal(Distribution):
    'A normal distribution with a shifted mean.'
    'Requires just passing the mean as the argument'
    def __init__(self, sigma):
        self.sigma = sigma
        super().__init__()
            
    def sample(self, mu):
        return np.random.normal(mu, self.sigma)

    def logpdf(self, x, mu):
        return -0.5 * np.log(2*np.pi) - np.log(self.sigma) - (x - mu)**2 / (2*self.sigma**2)