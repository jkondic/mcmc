import numpy as np
import jax.numpy as jnp
from jax import grad

class MALA():
    def __init__(self, target, x0, n, tau): 
        self.target = target # target distribution, p(x), to be approximated
        
        self.n = n # number of samples to draw
        self.x = np.zeros((n, *x0.shape)) # stack of n sample tensors of shape of x0
        self.x[0] = x0
        self.u = np.zeros((n, *x0.shape)) # momentum
        
        self.tau = tau # step size
        
        self.accepted = 0 # number of accepted samples
        self.acceptance_rate = 0.0
        self.acceptance_rate_history = np.zeros(n)
        self.acceptance_rate_history[0] = 0
        
 
    def run(self):
        for i in range(1, self.n):
            
            # NOTE that self.x[i-1] is the same as self.x[i-1,:,:,...,:],
            # for however many tensor components x has
            
            # sample from standard normal distribution
            z0 = np.random.randn(*self.x[0].shape)
            # initialize z
            # store grad calls for reuse
            #f_xm1 = -grad(self.target.logpdf)(self.x[i-1])
            f_xm1 = -self.target.grad_log(self.x[i-1])
            mean = self.x[i-1] - self.tau * f_xm1
            std = np.sqrt(2*self.tau)
            z = mean + std * z0
            #f_z = -grad(self.target.logpdf)(z)
            f_z = -self.target.grad_log(z)
            
            # define acceptance probability
            num = self.target.logpdf(z) - jnp.sum(jnp.power(self.x[i-1] - z + self.tau * f_z, 2))/(4*self.tau)
            den = self.target.logpdf(self.x[i-1]) - jnp.sum(jnp.power((z - self.x[i-1] + self.tau * f_xm1), 2))/(4*self.tau)
            log_A = num - den
            
            # accept or reject
            if np.random.uniform() <= min(1, jnp.exp(log_A)):
                self.x[i] = z
                self.accepted += 1
            else:
                self.x[i] = self.x[i-1]
                
            # update acceptance rate
            self.acceptance_rate = float(self.accepted/i)
            self.acceptance_rate_history[i] = self.acceptance_rate
            
                