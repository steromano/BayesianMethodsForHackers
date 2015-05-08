activate_this = "/Users/stefano.romano/DataScience/bin/activate_this.py"
execfile(activate_this, dict(__file__ = activate_this))

import numpy as np
import pymc as pm
from matplotlib import pyplot as plt


## Multi-armed bandit (really good one)

from pymc import rbeta

rand = np.random.rand

class Bandits(object):
    def __init__(self, p_array):
        self.p = p_array
        self.optimal = np.argmax(p_array)

    def __len__(self):
        return len(self.p)
    
    def pull(self, i):
        return rand() < self.p[i]


class BayesianStrategy(object):
    
    def __init__(self, bandits):
        self.bandits = bandits
        self.N = 0
        self.choices = []
        self.bb_score = []
        self.wins = np.zeros(len(bandits))
        self.trials = np.zeros(len(bandits))

    def sample_bandits(self, n = 1):
        bb_score = np.zeros(n)
        choices = np.zeros(n)
        for k in range(n):
            choice = np.argmax(rbeta(self.wins, self.trials - self.wins))
            result = self.bandits.pull(choice)
            self.wins[choice] += result
            self.trials[choice] += 1
            bb_score[k] = result
            choices[k] = choice
            self.N += 1

        self.bb_score = np.r_[self.bb_score, bb_score]
        self.choices = np.r_[self.choices, choices]
        return
            

from IPython.core.pylabtools import figsize            
import scipy.stats as stats
figsize(11.0, 10)
beta = stats.beta
def plot_priors(bs, prob, lw = 3, alpha = 0.2, plt_vlines = True):
    x = np.linspace(0.001, 0.999, 200)
    wins = bs.wins
    trials = bs.trials
    for i in range(len(prob)):
        y = beta(1 + wins[i], 1 + trials[i] - wins[i])
        p = plt.plot(x, y.pdf(x), lw = lw)
        c = p[0].get_markeredgecolor()
        plt.fill_between(x, y.pdf(x), 0, color = c, alpha = alpha,
                         label = "underlying probablity: %.2f" % prob[i])
        if plt_vlines:
            plt.vlines(prob[i], 0, y.pdf(prob[i]), linestyles = "--", colors = c, lw = 2)
        plt.autoscale(tight = True)
        plt.title("Posterior after %d pull" %bs.N + "s" * (bs.N > 1))
        plt.autoscale(tight = True)
    return

prob = np.array([0.85, 0.65, 0.75])
bandits = Bandits(prob)
bs = BayesianStrategy(bandits)
draw_samples = [1, 1, 3, 10, 10, 25, 50, 100, 200, 600]

for j, i in enumerate(draw_samples):
    plt.subplot(5, 2, j + 1)
    bs.sample_bandits(i)
    plot_priors(bs, prob)
    plt.autoscale(tight = True)

plt.tight_layout()



