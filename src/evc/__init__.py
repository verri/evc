import numpy as np
import scipy
import matplotlib.pyplot as plt

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


    def plot(self, score):

        nu, mu, tau2 = self.calculate_parameters(score)

        pdf = lambda x: scipy.stats.t.pdf(x, df=nu, loc=mu, scale=np.sqrt(tau2))

        # Plot a violin plot of the distribution with PDF `pdf` painting with
        # red the portion that is lower than -self.rope, in yellow the portion
        # between -self.rope and self.rope, and in green the portion that is
        # higher than self.rope.

        minx = (score - self.baseline_score).min()
        maxx = (score - self.baseline_score).max()

        x = np.linspace(minx, maxx, 1000)
        y = pdf(x)

        plt.plot(x, y, color='black')

        plt.fill_between(x, y, where=x < -self.rope, color='red')
        plt.fill_between(x, y, where=(x >= -self.rope) & (x <= self.rope), color='yellow')
        plt.fill_between(x, y, where=x > self.rope, color='green')

        plt.show()
