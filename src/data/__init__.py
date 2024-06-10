"""
Demo implementation of the Dataset class to be used in the project.
"""

from sklearn.datasets import load_iris

class Dataset:

    def __init__(self):
        pass

    def load(self):
        data = load_iris()
        return data.data, data.target
