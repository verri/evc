class Evaluation:

    def __init__(self):
        pass

    def evaluate(self, y_true, y_pred):
        return (y_true == y_pred).mean()
