from collections import Counter
import numpy as np


def predict(x_test, centroids):
    # create list for distances and target
    distances = []

    for i in range(len(centroids)):
        # first we compute the euclidean distance
        distance = np.sqrt(np.sum(np.square(x_test - centroids[i, :])))
        # add it to list of distances
        distances.append([distance, i])

    # sort the list
    distances = sorted(distances)

    return


def NearestControid(X_train, y_train, X_test, predictions, k):
    # train on the input data
    train(X_train, y_train)

    # loop over all observations
    for i in range(len(X_test)):
        predictions.append(predict(X_test[i, :], centroids))


def train(X_train, y_train):
    # This method should create the centroids

    return