import random
import numpy as np
import scipy.stats as stats
import pymc as pm

probs = np.arange(0.1, 1, 0.02)
prior = {a:1.0/len(probs) for a in probs}

def flip():
    return random.choice(["heads", "tails"])

def p(x, a):
    if x == "heads":
        return a
    if x == "tails":
        return 1 - a

def posterior(x, prior):
    den = sum(p(x, a) * prior[a] for a in probs)
    return {a:p(x, a) * prior[a]/den for a in probs}

for i in xrange(1000):
    prior = posterior(flip(), prior)
        

mu_true = [10, 15]
counts1 = stats.poisson.rvs(mu_true[0], size = 30)
counts2 = stats.poisson.rvs(mu_true[1], size = 20)
counts = np.concatenate((counts1, counts2), axis = 1)


alpha = 1.0/counts.mean()
lambda_1 = pm.Exponential("lambda_1", alpha)
lambda_2 = pm.Exponential("lambda_2", alpha)

tau = pm.DiscreteUniform("tau", lower = 0, upper = len(counts))

@pm.deterministic
def lambda_(tau = tau, lambda_1 = lambda_1, lambda_2 = lambda_2):
    out = np.zeros(len(counts))
    out[:tau] = lambda_1
    out[tau:] = lambda_2
    return out

observation = pm.Poisson("obs", lambda_, value = counts, observed = True)
model = pm.Model([observation, lambda_1, lambda_2, tau])

mcmc = pm.MCMC(model)
mcmc.sample(40000, 10000, 1)

lambda_1_samples = mcmc.trace("lambda_1")[:]
lambda_2_samples = mcmc.trace("lambda_2")[:]
tau_samples = mcmc.trace("tau")[:]

ax = plt.subplot(311)
ax.set_autoscaley_on(False)

plt.hist(lambda_1_samples, histtype = "stepfilled", bins = 30, alpha = 0.85,
         label = "posterior of $\lambda_1$", color= "#A60628", normed = True)
plt.legend(loc = "upper left")
plt.title("Posterior distribution of the variables $\lambda_1$, $\lambda_2$, $\tau$")
plt.xlim([15, 30])
plt.xlabel("$\lambda_1$ value")

ax = plt.subplot(312)
ax.set_autoscaley_on(False)
plt.hist(lambda_2_samples, histtype='stepfilled', bins=30, alpha=0.85,
         label="posterior of $\lambda_2$", color="#7A68A6", normed=True)
plt.legend(loc="upper left")
plt.xlim([15, 30])
plt.xlabel("$\lambda_2$ value")


plt.subplot(313)
w = 1.0 / tau_samples.shape[0] * np.ones_like(tau_samples)
plt.hist(tau_samples, bins=len(counts), alpha=1,
         label=r"posterior of $\tau$",
         color="#467821", weights=w, rwidth=2.)
plt.xticks(np.arange(len(counts)))

plt.legend(loc="upper left")
plt.ylim([0, .75])
plt.xlim([35, len(counts) - 20])
plt.xlabel(r"$\tau$ (in days)")
plt.ylabel("probability");
