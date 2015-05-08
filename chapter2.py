import pymc as pm
from matplotlib import pyplot as plt

parameter = pm.Exponential("posson_param", 1)
data_generator = pm.Poisson("data_generator", parameter)
data_plus_one = data_generator + 1

@pm.deterministic
def data_plus_two(dg = data_generator):
    return dg + 2

data_generator.children
data_generator.parents
foos = pm.Uniform("foos", 0, 1, size = 20)

parameter.value
data_generator.value
data_plus_one.value
data_plus_two.value
foos.value

samples = [data_generator.random() for _ in xrange(10000)]

plt.hist(samples, bins = 70, normed = True, histtype = "stepfilled")
plt.title("FooBar")
plt.xlim(0, 8)
plt.show()

# AB testing example ---------------
# Preliminary -- just A
p = pm.Uniform("p", lower = 0, upper = 1)
p_true = 0.05
N = 1500

occurrences = pm.rbernoulli(p_true, N)
obs = pm.Bernoulli("obs", p, value = occurrences, observed = True)
model = pm.Model([p, obs])

mcmc = pm.MCMC(model)
mcmc.sample(18000, 1000)

samples = mcmc.trace("p")[:]

# AB

true_p_A = 0.05
true_p_B = 0.04
n_A = 1500
n_B = 750

observations_A = pm.rbernoulli(true_p_A, n_A)
observations_B = pm.rbernoulli(true_p_B, n_B)

p_A = pm.Uniform("p_A", 0, 1)
p_B = pm.Uniform("p_B", 0, 1)
@pm.deterministic
def delta_p(pA = p_A, pB = p_B):
    return pA - pB
obs_A = pm.Bernoulli("obs_A", p_A, value = observations_A, observed = True)
obs_B = pm.Bernoulli("obs_B", p_B, value = observations_B, observed = True)

model = pm.Model([p_A, p_B, delta_p, obs_A, obs_B])
mcmc = pm.MCMC(model)
mcmc.sample(50000, 1000)
samples_A = mcmc.trace("p_A")[:]
samples_B = mcmc.trace("p_B")[:]
samples_delta = mcmc.trace("delta_p")[:]

ax = plt.subplot(311)
plt.xlim(0, 0.1)
plt.hist(samples_A, histtype = "stepfilled", bins = 25, alpha = 0.6,
         label = "posterior of $p_A$", color = "#A60628", normed = True)
plt.vlines(true_p_A, 0, 90, linestyle = "--", label = "true $p_A$", lw = 2)
plt.legend(loc = "upper right")

ax = plt.subplot(312)
plt.hist(samples_B, histtype = "stepfilled", bins = 25, alpha = 0.6,
         label = "posterior of $p_B$", color = "#467821", normed = True)
plt.vlines(true_p_B, 0, 80, linestyle = "--", label = "true $p_B$", lw = 2)
plt.legend(loc = "upper right")

ax = plt.subplot(313)
plt.hist(samples_delta, histtype = "stepfilled", bins = 25, alpha = 0.6,
         label = "posterior of $p_A - p_B$", color = "#7A68A6", normed = True)
plt.vlines(true_p_A - true_p_B, 0, 90, linestyle = "--", lw = 2,
           label = "true $p_A - p_B$")
plt.legend(loc = "upper right")


# Binomial distribution

import scipy.stats as stats
binom = stats.binom

parameters = [(10, 0.4), (10, 0.9)]
colors = ["#348ABD", "#A60628"]
for i in range(2):
    N, p = parameters[i]
    _x = np.arange(N + 1)
    plt.bar(_x, binom.pmf(_x, N, p), color = colors[i],
            edgecolor = colors[i],
            alpha = 0.6,
            label = "$N$: %d, $p$: %.1f" %(N, p),
            linewidth = 3)

plt.legend(loc = "upper left")
plt.xlim(0, 11.5)
plt.xlabel("$k")
plt.ylabel("$p(k)$")

# Students cheating example

N = 1000
p = pm.Uniform("cheating_p", 0, 1)
true_answers = pm.Bernoulli("truths", p, size = N)

first_coin_flip = pm.Bernoulli("first flips", p = 0.5, size = N)
second_coin_flip = pm.Bernoulli("second flips", p = 0.5, size = N)

@pm.deterministic
def observed_proportion(ta = true_answers,
                        fcf = first_coin_flip,
                        scf = second_coin_flip):
    ans = np.arange(N)
    ans[fcf] = ta[fcf]
    ans[~fcf] = scf[~fcf]
    return ans.mean()

X = 350
observations = pm.Binomial("obs", N, observed_proportion, value = X, observed = True)

model = pm.Model([p, observed_proportion, observations])
mcmc = pm.MCMC(model)
mcmc.sample(40000, 15000)

p_samples = mcmc.trace("cheating_p")[:]

plt.hist(p_samples, histtype = "stepfilled", normed = True, alpha = 0.7, bins = 30,
         label = "posterior distribution", color = "#348ABD")
plt.xlim(0, 1)
plt.legend()

# Challenger Space Shuttle Disaster example
np.set_printoptions(precision = 3, suppress = True)

challenger_data = np.genfromtxt("data/challenger_data.csv", skip_header = True,
                                usecols = [1, 2], missing_values = "NA", delimiter = ",")

plt.clf()
plt.scatter(challenger_data[:, 0], challenger_data[:, 1], s = 75, color = "k", alpha = 0.5)
plt.yticks([0, 1])
plt.ylabel("Damage incident?")
plt.xlabel("Outside temperature (Fahrenheit)")
plt.title("Defects of the Space Shuttle O-Rings vs temperature")

# Logistic function
def logistic(x, beta, alpha = 0):
    return 1.0/(1.0 + np.exp(np.dot(beta, x) + alpha))

x = np.linspace(-4, 4, 100)
plt.clf()
plt.plot(x, logistic(x, 1), label = r"$\beta = 1$", ls = "--", lw = 2)
plt.plot(x, logistic(x, 3), label = r"$\beta = 3$", ls = "--", lw = 2)
plt.plot(x, logistic(x, 1, 2), label = r"$\beta = 1,  \alpha = 2$", lw = 2)
plt.plot(x, logistic(x, 2, -3), label = r"$\beta = 2,  \alpha = -3$", lw = 2)
plt.legend()

# Model parameters
temperature = challenger_data[:, 0]
D = challenger_data[:, 1]

beta = pm.Normal("beta", 0, 0.001, value = 0)
alpha = pm.Normal("alpha", 0, 0.001, value = 0)

@pm.deterministic
def p(t = temperature, a = alpha, b = beta):
    return logistic(t, b, a)

observations = pm.Bernoulli("obs", p, value = D, observed = True)
model = pm.Model([alpha, beta, p, observations])

mcmc = pm.MCMC(model)
mcmc.sample(120000, 100000, 2)

alpha_samples = mcmc.trace("alpha")[:, None]
beta_samples = mcmc.trace("beta")[:, None]

plt.clf()
ax = plt.subplot(211)
plt.title(r"Posterior $\alpha$ and $\beta$ distributions")
plt.hist(beta_samples, histtype = "stepfilled", bins = 35, normed = True,
         color = "#7A68A6", alpha = 0.65, label = r"Posterior of $\beta$")
plt.legend(loc = "upper right")
ax = plt.subplot(212)
plt.hist(alpha_samples, histtype = "stepfilled", bins = 35, normed = True,
         color = "#A60628", alpha = 0.65, label = r"Posterior of $\alpha$")
plt.legend(loc = "upper right")


_t = np.linspace(temperature.min() - 5, temperature.max() + 5, 50)[:, None]
p_t = logistic(_t.T, beta_samples, alpha_samples)
_p = p_t.mean(axis = 0)

plt.clf()
plt.plot(_t, _p, color = "r", lw = 3, alpha = 0.6,
         label = "Estimated probability of failure")
plt.legend()
plt.xlabel("temperature")
plt.ylabel("probability")

from scipy.stats.mstats import mquantiles
confidence_intervals = mquantiles(p_t, axis = 0, prob = [0.025, 0.975])

plt.clf()
plt.fill_between(_t[:, 0], *confidence_intervals, alpha = 0.5, color = "#7A68A6")
plt.plot(_t[:, 0], confidence_intervals[0], alpha = 0.5, color = "#7A68A6",
         label = "95% confidence interval")
plt.plot(_t, _p, ls = "--", lw = 2, color = "k",
         label = "average posterior probability of defect")
plt.scatter(temperature, D, color = "k", alpha = 0.4)
plt.xlabel("temperature")
plt.ylabel("probability")
plt.legend(loc = "lower left")
plt.title("Posterior probability estimate given temperature")

# Simulated data and goodness of fit
simulated_data = pm.Bernoulli("simulated", p)
model = pm.Model([alpha, beta, p, observations, simulated_data])
mcmc = pm.MCMC(model)
mcmc.sample(10000)

simulations = mcmc.trace("simulated")[:]
