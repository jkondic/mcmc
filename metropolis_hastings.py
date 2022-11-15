import numpy as np
from distributions import *

class MetropolisHastings():
    'Metropolis-Hastings algorithm for drawing samples from a distribution to be approximated.'
    
    def __init__(self, target, proposal, x0, n): 
        self.target = target # target distribution, p(x), to be approximated
        self.proposal = proposal # proposal distribution q(x*|x)
        self.n = n # number of samples to draw
        self.x = np.zeros(n) # samples
        self.x0 = x0
        self.accepted = 0 # number of accepted samples
        self.acceptance_rate = float(0)
        self.acceptance_rate_history = np.zeros(n)
        self.acceptance_rate_history[0] = 0
        
    def run(self):  
        for i in range(1, self.n):
            # sample from proposal distribution
            x_proposed = self.proposal.sample(self.x[i-1])
            # define acceptance probability
            log_p_proposed = self.target.logpdf(x_proposed)
            log_p_sampled = self.target.logpdf(self.x[i-1])
            log_q_proposed = self.proposal.logpdf(x_proposed, self.x[i-1])
            log_q_sampled = self.proposal.logpdf(self.x[i-1], x_proposed)
            log_A = log_p_proposed + log_q_sampled - log_p_sampled - log_q_proposed
            # accept or reject
            if np.random.uniform() < min(1, np.exp(log_A)):
            #if np.random.binomial(1, min(1, np.exp(log_A))):
                self.x[i] = x_proposed
                self.accepted += 1
            else:
                self.x[i] = self.x[i-1]
            # update acceptance rate
            self.acceptance_rate = float(self.accepted) / float(i)
            self.acceptance_rate_history[i] = self.acceptance_rate
            
        