import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
import time

from ORL.loadORL import loadORL
from kNearestNeighbor import kNearestNeighbor

# load in data
filesToLoad = {'images': 'ORL/orl_data.mat', 'labels': 'ORL/orl_lbls.mat'}
x_train, x_test, y_train, y_test = loadORL(filesToLoad)

scaler = StandardScaler()

# Standardizing the features
scaler.fit(x_test)
x_test = scaler.transform(x_test)
x_train = scaler.transform(x_train)

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=10000)

mlp.fit(x_train, y_train)
predictions = mlp.predict(x_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# Make an instance of the Model
pca = PCA(n_components=40)

# apply PCA inorder to get fewer dimensions to work with
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.fit_transform(x_test)

# making our predictions
predictions = []
# plt.scatter(x_train, y_train)

#KNearestNeighbor
kNearestNeighbor(x_train, y_train, x_test, predictions, 3)

# transform the list into an array
predictions = np.asarray(predictions)

# evaluating accuracy
accuracy = accuracy_score(y_test, predictions)
print('The accuracy of our KNearestNeighbor classifier is ' + str(accuracy * 100))


# NearestCentroid
ncc = NearestCentroid()
ncc.fit(x_train, y_train)
# get the model accuracy
modelscore = ncc.score(x_test, y_test)
print('The accuracy of our NearestCentroid classifier is ' + str(modelscore * 100))