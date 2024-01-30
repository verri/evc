import numpy as np
import scipy
import matplotlib.pyplot as plt
from cmcrameri import cm

class BayesianEvaluation:


    def __init__(self, baseline_score: np.ndarray, replications: int = 10, rope: float = 0):

        self.baseline_score = baseline_score
        self.n = self.baseline_score.size
        self.rho = 1 / (self.n // replications)
        self.rope = rope


    def calculate_parameters(self, score: np.ndarray):
        """ Calculates the parameters of the Bayesian correlated t-test. """

        x = score - self.baseline_score

        nu = self.n - 1
        mu = x.mean()
        tau2 = (1 / self.n + self.rho / (1 - self.rho)) * x.var(ddof=1)

        return nu, mu, tau2


    def calculate_bounds(self, score: np.ndarray):

        nu, mu, tau2 = self.calculate_parameters(score)
        ppf = lambda x: scipy.stats.t.ppf(x, df=nu, loc=mu, scale=np.sqrt(tau2))

        return ppf(0.05), ppf(0.95)


    def evaluate(self, score):

        nu, mu, tau2 = self.calculate_parameters(score)

        # Calculate the probability that the score is better than the baseline
        # score.  Use the cumulative distribution function of the
        # t-distribution.

        cdf = lambda x: scipy.stats.t.cdf(x, df=nu, loc=mu, scale=np.sqrt(tau2))

        pworse = cdf(-self.rope)
        pequiv = cdf(self.rope) - pworse
        pbetter = 1 - cdf(self.rope)

        return pbetter, pequiv, pworse


    def plot_at(self, score, ax, name, minx=None, maxx=None):

        nu, mu, tau2 = self.calculate_parameters(score)

        pdf = lambda x: scipy.stats.t.pdf(x, df=nu, loc=mu, scale=np.sqrt(tau2))

        # Plot a violin plot of the distribution with PDF `pdf` painting with
        # red the portion that is lower than -self.rope, in yellow the portion
        # between -self.rope and self.rope, and in green the portion that is
        # higher than self.rope.

        if minx is None:
            minx = (score - self.baseline_score).min()

        if maxx is None:
            maxx = (score - self.baseline_score).max()

        x = np.linspace(minx, maxx, 1000)
        y = pdf(x)

        ax.set_title(name)
        ax.plot(x, y, color='black')

        # Load roma pallette from cmcrameri.
        # The first color is red, the second is yellow, the third is green.

        c1, c2, c3 = cm.roma((0.1, 0.4, 0.8))

        ax.fill_between(x, y, where=x < -self.rope, color=c1)
        ax.fill_between(x, y, where=(x >= -self.rope) & (x <= self.rope), color=c2)
        ax.fill_between(x, y, where=x > self.rope, color=c3)

        # Write the probabilities in the plot.
        pbetter, pequiv, pworse = self.evaluate(score)

        ax.text(0.05, 0.95, f'P({name} > baseline) = {100 * pbetter:.0f}%', transform=ax.transAxes)
        ax.text(0.05, 0.90, f'P({name} = baseline) = {100 * pequiv:.0f}%', transform=ax.transAxes)
        ax.text(0.05, 0.85, f'P({name} < baseline) = {100 * pworse:.0f}%', transform=ax.transAxes)

        # Draw a dashed vertical line at mu.

        ax.axvline(mu, color='black', linestyle='dashed')


    def plot(self, scores):

        n = len(scores)
        fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(5, n * 5))

        minx = min([self.calculate_bounds(score)[0] for name, score in scores])
        maxx = max([self.calculate_bounds(score)[1] for name, score in scores])

        for i, (name, score) in enumerate(scores):
            ax = axes.flatten()[i]
            self.plot_at(score, ax, name, minx=minx, maxx=maxx)

        plt.show()
