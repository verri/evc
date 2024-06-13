class Evaluation:

    name = 'accuracy'
    version = '0.1'

    def __init__(self):
        pass

    def evaluate(self, y_true, y_pred):
        return (y_true == y_pred).mean()
