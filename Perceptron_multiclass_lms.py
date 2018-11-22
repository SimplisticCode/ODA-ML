
import numpy as np


class Perceptron_multiclass_lms():
    def __init__(self):
        self.weights = None

    def train(self, X, y):
        y = y - 1
        targetvector = np.zeros((len(X),len(set(y))))
        for i in range(len(y)):
            targetvector[i, y[i]] = 1
        # taking the pseudo inverse
        A = np.linalg.pinv(X)
        self.weights = A @ targetvector

    def predict(self, X):
        result = self.weights.T @ X.T
        predictions = np.argmax(result, axis=0)
        return predictions








