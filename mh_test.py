import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from distributions import *

### define some example cost functions ###
def sos_cost(x):
    vec = np.ones((2,2))
    return np.sum(np.power(x - vec, 2)) # sum of squares cost

def Branin(x): 
    return np.sum(jnp.power((x[:,1] - (5.1/(4*np.pi**2))*jnp.power(x[:,0], 2) + (5/np.pi)*x[:,0] - 6), 2) + 10*(1-1/(8*jnp.pi))*jnp.cos(x[:,0]) + 10)

def Beale(x):
    return (1.5 - x[:,0] + x[:,0]*x[:,1])**2 + (2.25 - x[:,0] + x[:,0]*x[:,1]**2)**2 + (2.625 - x[0] + x[:,0]*x[:,1]**3)**2

### define the target distribution & obtain samples ###
target = Boltzmann(shape=(2,2), cost=sos_cost, beta=1)
samples = target.sample(n=100000)

### plot the samples ###
x_axis = np.arange(-5, 5, 0.01)
# get the histogram for marginal at index [0,0]
plt.hist(samples[:,0,0], bins=100, density=True)
# plot the analytic marginal (in this case)
plt.plot(x_axis, np.exp(-(x_axis - 1)**2))
plt.show()