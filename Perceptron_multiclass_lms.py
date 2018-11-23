
import numpy as np


class Perceptron_multiclass_lms():
    def __init__(self):
        self.weights = None

    def train(self, X, y):
        y = y - 1 # matrixes in python are 0 indexed minus 1 to fit the labels to the matrix (1-40) to (0-39)
        t = np.zeros((len(X),len(set(y))))
        for i in range(len(y)):
            t[i, y[i]] = 1
        # taking the pseudo inverse
        A = np.linalg.pinv(X)
        w = A @ t
        self.weights = w

    def predict(self, X):
        result = self.weights.T @ X.T
        predictions = np.argmax(result, axis=0)
        predictions = predictions + 1 # matrixes are 0 indexed therefore plus 1 from (0-39) to (1-40)
        return predictions








