from collections import defaultdict
import numpy as np
from numpy import sort


class NearestCentroid():
    def __init__(self):
        self.centroids = None

    def fit(self, X, y):
        subrows = defaultdict(list)
        for i in range(len(y)):
            # Collect indices of exemplars for the given class label
            subrows[y[i]].append(i)

        centroids = []
        for index, label in enumerate(subrows.keys()):
            exemplars = X[subrows[label]]
            # compute centroid for exemplars
            centroid = self.centroid(exemplars)
            centroids.append({"centroid": centroid, "label": label})
        self.centroids = centroids
        return self

    def centroid(self, X):
        centroid = X.mean(axis=0)
        return centroid

    def predict(self, X):
        results = []
        for sample in X:
            distances = []
            for centroid in self.centroids:
               distances.append((np.linalg.norm(sample - centroid["centroid"]), centroid["label"]))
            distances = sorted(distances, key=lambda x: x[0])
            results.append(distances[0][1])
                
        return results





