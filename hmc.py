import numpy as np
import jax.numpy as jnp
from jax import grad

class HamiltonianMonteCarlo():
    def __init__(self, target, x0, n, L, p): 
        self.target = target # target distribution, p(x), to be approximated
        
        self.n = n # number of samples to draw
        self.x = np.zeros((n, *x0.shape)) # stack of n sample tensors of shape of x0
        self.x[0] = x0
        self.u = np.zeros((n, *x0.shape)) # momentum
        
        self.L = L # number of leapfrog steps
        self.p = p # step size
        
        self.accepted = 0 # number of accepted samples
        self.acceptance_rate = 0.0
        self.acceptance_rate_history = np.zeros(n)
        self.acceptance_rate_history[0] = 0
        
 
    def run(self):
        for i in range(1, self.n):
            
            # NOTE that self.x[i-1] is the same as self.x[i-1,:,:,...,:],
            # for however many tensor components x has
            
            # sample u_proposed from normal distribution
            #u_proposed = np.sqrt(2*self.p)*np.random.randn(*self.x[0].shape)
            u_proposed = np.random.randn(*self.x[0].shape)
            
            # initialize x0 and u0
            x_leapfrog = self.x[i-1]
            u_leapfrog = u_proposed + 0.5* self.p * self.target.grad_log(x_leapfrog)
            
            for l in range(self.L):
                
                if l < self.L-1:
                    p_l = self.p
                else:
                    p_l = 0.5 * self.p
                    
                # update x
                x_leapfrog = x_leapfrog + self.p * u_leapfrog
                # update u
                u_leapfrog = u_leapfrog - p_l * self.target.grad_log(x_leapfrog)
            
            # define acceptance probability
            log_p_xL = self.target.logpdf(x_leapfrog)
            u_L_norm = np.dot(u_leapfrog.flatten(),u_leapfrog.flatten())
            u_p_norm = np.dot(u_proposed.flatten(),u_proposed.flatten())
            u_term = -0.5 * u_L_norm + 0.5 * u_p_norm
            log_p_xi = self.target.logpdf(self.x[i-1])
            log_A = log_p_xL + u_term - log_p_xi
            
            # accept or reject
            if np.random.uniform() < min(1, jnp.exp(log_A)):
                self.x[i] = x_leapfrog
                self.u[i] = u_leapfrog
                self.accepted += 1
            else:
                self.x[i] = self.x[i-1]
                self.u[i] = u_proposed
                
            # update acceptance rate
            self.acceptance_rate = float(self.accepted/i)
            self.acceptance_rate_history[i] = self.acceptance_rate
            
                