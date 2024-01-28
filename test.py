import numpy as np
from evc import BayesianEvaluation

# Create random vector of scores.
baseline = np.random.normal(0, 1, 100)
score = np.random.normal(0.1, 1, 100)

# Create an instance of the BayesianEvaluation class.
be = BayesianEvaluation(baseline)

# Plot the distribution of the score.
be.plot(score)
