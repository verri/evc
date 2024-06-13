from sklearn.model_selection import KFold, StratifiedKFold

class RepeatedKFold:

    def __init__(self, n_repeats, n_splits, stratified=True):
        self.n_repeats = n_repeats
        self.n_splits = n_splits
        self.cv = StratifiedKFold if stratified else KFold

    def split(self, X, y):
        """Returns the indices of the training and test sets."""
        splits = []
        for seed in range(self.n_repeats):
            cv = self.cv(n_splits=self.n_splits, random_state=seed, shuffle=True)
            for train, test in cv.split(X, y):
                result = { "train": train.tolist(), "test": test.tolist() }
                splits.append(result)
        return splits

    def correction_factor(self):
        """Correlation between test sets.  The same thing as the $\rho$ parameter.
        in the paper."""
        return 1 / self.n_repeats # TODO check vs. 1 / self.n_splits
