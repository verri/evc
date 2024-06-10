from sklearn.model_selection import KFold, StratifiedKFold

class RepeatedKFold:

    def __init__(self, n_repeats, n_splits, stratified=True):
        self.n_repeats = n_repeats
        self.n_splits = n_splits
        self.cv = StratifiedKFold if stratified else KFold

    def split(self, labels):
        """Returns the indices of the training and test sets."""
        splits = []
        for seed in range(self.n_repeats):
            cv = self.cv(n_splits=self.n_splits, random_state=seed, shuffle=True)
            for train, test in cv.split(labels, labels):
                result = { "train": train, "test": test }
                splits.append(result)
