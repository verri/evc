from sklearn.tree import DecisionTreeClassifier

class Model:
    """Model class for Decision Tree."""

    name = "DT"
    version = 0.1

    def __init__(self, max_depth=3):
        self.model = DecisionTreeClassifier(max_depth=max_depth,
                random_state=42)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
