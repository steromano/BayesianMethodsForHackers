activate_this = "/Users/stefano.romano/DataScience/bin/activate_this.py"
execfile(activate_this, dict(__file__ = activate_this))

import numpy as np
import pymc as pm
from matplotlib import pyplot as plt

from scipy.stats import poisson as pois

sample_size = 500
mus  = [2, 10, 23]
colors = ["r", "b", "g"]
plt.clf()
for i in range(3):
    samples = pois.rvs(mus[i], size = sample_size)
    means = np.array([mean(samples[:k]) for k in xrange(sample_size)])
    plt.plot((means - (mus[i]))/float(mus[i]), color = colors[i],
             label = r"Relative convergence of $\mu_%d$" %i, lw = 2)


plt.axhline(y=0, ls = "--", color = "k", lw = 3)
plt.ylim(-0.2, 0.2)
plt.legend()

## Aggregated geographical data example
from numpy.random import random_integers as dunif
from scipy.stats import norm
import pandas as pd
from pandas import DataFrame
countries = np.concatenate([np.repeat(i, dunif(100, 1500)) for i in xrange(5000)])
height_data = DataFrame({"height": norm.rvs(150, 15, size = len(countries)),
                         "country": countries})

summary = height_data.groupby("country").agg(["mean", "count"]).height.sort("mean")
print summary.head()["count"].mean()
print summary.tail()["count"].mean()
print summary["count"].mean()

## Reddit upvotes example
from IPython.core.display import Image
run top_pic_comments.py 2

def posterior_upvote_ratio(upvotes, downvotes, samples = 20000):
    p = pm.Uniform("p", 0, 1, value = 0.5)
    n = upvotes + downvotes
    obs_upvotes = pm.Binomial("obs", n, p, value = upvotes, observed = True)
    model = pm.Model([p, obs_upvotes])
    mcmc = pm.MCMC(model)
    mcmc.sample(samples)
    return mcmc.trace("p")[:]

# Hack to get a few downvotes, since I get none
from numpy.random import randint
downvotes = randint(0, 10, size = len(contents))
votes[:, 1] = downvotes

posteriors = np.array(map(lambda x: posterior_upvote_ratio(*x), votes))
mean_scores = posteriors.mean(axis = 1)
lowest95_scores = np.percentile(posteriors, q = 5, axis = 1)

summary = DataFrame({"upvotes": votes[:, 0],
                     "downvotes": votes[:, 1],
                     "mean_score": mean_scores,
                     "lowest95_score": lowest95_scores})\
          [["upvotes", "downvotes", "mean_score", "lowest95_score"]].\
          sort("lowest95_score", ascending = False)

# Exercise

import scipy.stats as stats
expon = stats.expon(scale = 4)
N = 1e5
X = expon.rvs(N)
print np.cos(X).mean()
print np.cos(X[X < 1]).mean()
