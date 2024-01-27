import numpy as np
import scipy

class BayesianEvaluation:


    def __init__(self, baseline_score: np.ndarray, replications: int = 10, rope: float = 0):

        self.baseline_score = baseline_score
        self.n = self.base_score.size
        self.rho = 1 / (self.n // replications)


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
