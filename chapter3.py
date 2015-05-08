activate_this = "/Users/stefano.romano/DataScience/bin/activate_this.py"
execfile(activate_this, dict(__file__ = activate_this))

import numpy as np
import pymc as pm
from matplotlib import pyplot as plt

data = np.loadtxt("data/mixture_data.csv")
plt.clf()
plt.hist(data, histtype = "stepfilled", color = "g", alpha = 0.4, bins = 20)
plt.ylim(0, 35)

## Data generation model ------------------------------
p = pm.Uniform("p", 0, 1, value = 0.9)
assignment = pm.Categorical("assignment", [p, 1 - p], size = len(data))

sigmas = pm.Uniform("sigmas", 0, 100, size = 2)
taus = 1.0/(sigmas**2)
mus = pm.Normal("mus", [120, 190], [0.01, 0.01])

@pm.deterministic
def tau_i(assignment = assignment, taus = taus):
    return taus[assignment]

@pm.deterministic
def mu_i(assignment = assignment, mus = mus):
    return mus[assignment]

observations = pm.Normal("obs", mu_i, tau_i, value = data, observed = True)

model = pm.Model([p, assignment, sigmas, mus, observations])
map_ = pm.MAP(model)
map_.fit()
mcmc = pm.MCMC(model)
mcmc.sample(50000)

mus_samples = mcmc.trace("mus")[:]
sigmas_samples = mcmc.trace("sigmas")[:]
assignment_samples = mcmc.trace("assignment")[:]

plt.clf()
colors = ["#348ABD", "#A60628"]
for i in range(2):
    plt.plot(mus_samples[:, i], color = colors[i], label = r"Trace of $\mu_%d$" %i)

plt.legend()

plt.clf()
for i in range(2):
    plt.plot(sigmas_samples[:, i], color = colors[i], label = r"Trace of $\sigma_%d$" %i)

plt.legend()
    
for i in range(2):
    plt.hist(mus_samples[:, i], histtype = "stepfilled", color = colors[i],
             label = r"Posterior samples of $\mu_%d$" %i, alpha = 0.7)

plt.legend()
plt.clf()
for i in range(2):
    plt.hist(stds_samples[:, i], histtype = "stepfilled", color = colors[i],
             label = r"Posterior samples of $\sigma_%d$" %i, alpha = 0.7)

plt.legend()

cluster1_freq = assignment_samples.sum(axis = 1)/float(assignment_samples.shape[1])

plt.clf()
plt.plot(cluster1_freq, color = "g", lw = 3)
plt.ylim(0, 1)

# Continue sampling

mcmc.sample(100000)
mus_samples = mcmc.trace("mus", chain = 1)[:]
prev_mus_samples = mcmc.trace("mus", chain = 0)[:]

cluster1_probs = assignment_samples.mean(axis = 0)

from pymc.Matplot import plot as mcplot

mcplot(mcmc.trace("mus"), common_scale = False)
