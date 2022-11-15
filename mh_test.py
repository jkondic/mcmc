import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import metropolis_hastings as mh
from distributions import *

# define parameters
BURN_IN = 1000

# define target distribution
t1 = Normal(-2, 1)
t2 = Normal(2, 1)
target = SumDistribution(t1, t2)

# define proposal distribution
proposal = ShiftedNormal(3)

# initialize Metropolis-Hastings class
mh_test = mh.MetropolisHastings(target=target, proposal=proposal, x0=1, n=10_000)

# run Metropolis-Hastings algorithm
mh_test.run()

# collect samples
samples = mh_test.x[BURN_IN:]

# print acceptance rate
print("ACCEPTANCE RATE: ", mh_test.acceptance_rate)

# plot samples
x_axis = np.arange(-5, 5, 0.01)
plt.hist(samples, bins=100, density=True)
plt.plot(x_axis, np.exp(target.logpdf(x_axis)))
plt.show()