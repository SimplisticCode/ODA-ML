
import numpy as np
from sklearn.preprocessing import LabelEncoder


class Perceptron_multiclass_backpropagation():
    def __init__(self):
        self.weights = None
        self.classes = None

    def train(self, X, y, phi=0.1, eta=0.1):

        # Perform label encoding so label indicies start from zero
        le = LabelEncoder()
        encoded_y = le.fit_transform(y)
        self.classes = le.classes_
        n_classes = len(self.classes)

        #initial weights
        self.weights = np.zeros((len(X[0]), len(set(y))))

        #One vs. rest
        for label in range(n_classes):
            ind_pos = [i for i in range(len(y)) if y[i] == label]
            c1 = [X[i] for i in ind_pos]
            not_ind_pos = set(range(len(y))).difference(set(ind_pos))
            c2 = [X[j] for j in not_ind_pos]
            self.update_weights(c1, c2, label, phi, eta)



    def update_weights(self, c1, c2, label, phi, eta):
        


    def predict(self, X):
        result = self.weights.T @ X.T
        predictions = np.argmax(result, axis=0)
        predictions = predictions + 1
        return predictions









