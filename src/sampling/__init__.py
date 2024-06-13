"""
Demo implementation of the Sampling class to be used in the project.

At the moment, we only support repeated k-fold cross-validation.
"""

from evc.sampling import RepeatedKFold

class Sampling(RepeatedKFold):

    def __init__(self):
        super().__init__(n_repeats=10, n_splits=10, stratified=True)
