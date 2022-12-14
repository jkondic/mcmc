import numpy as np

class MetropolisHastings():
    'Metropolis-Hastings algorithm for drawing samples from a distribution to be approximated.'
    
    def __init__(self, target, proposal, x0, n): 
        self.target = target # target distribution, p(x), to be approximated
        self.proposal = proposal # proposal distribution q(x*|x)
        
        self.n = n # number of samples to draw
        self.x = np.zeros((n, *x0.shape)) # stack of n sample tensors of shape of x0
        self.x[0] = x0
        
        self.accepted = 0 # number of accepted samples
        self.acceptance_rate = 0.0
        self.acceptance_rate_history = np.zeros(n)
        self.acceptance_rate_history[0] = 0
        
    def run(self):  
        for i in range(1, self.n):

            # NOTE that self.x[i-1] is the same as self.x[i-1,:,:,...,:],
            # for however many tensor components x has

            # sample from proposal distribution; assumed to be a ShiftedNormal
            x_proposed = self.proposal.sample(self.x[i-1])

            # define acceptance probability
            log_p_proposed = self.target.logpdf(x_proposed)
            log_p_sampled = self.target.logpdf(self.x[i-1])
            log_q_proposed = self.proposal.logpdf(x_proposed, self.x[i-1])
            log_q_sampled = self.proposal.logpdf(self.x[i-1], x_proposed)
            log_A = log_p_proposed + log_q_sampled - log_p_sampled - log_q_proposed
            
            # accept or reject
            if np.random.uniform() < min(1, np.exp(log_A)):
                self.x[i] = x_proposed
                self.accepted += 1
            else:
                self.x[i] = self.x[i-1]
                
            # update acceptance rate
            self.acceptance_rate = float(self.accepted/i)
            self.acceptance_rate_history[i] = self.acceptance_rate
            
        