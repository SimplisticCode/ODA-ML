import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
import time

from NearestCentroid import NearestCentroid
from NearestSubclassCentroid import NearestSubclassCentroid
from ORL.loadORL import loadORL
from Perceptron_multiclass_backpropagation import Perceptron_multiclass_backpropagation
from Perceptron_multiclass_lms import Perceptron_multiclass_lms
from kNearestNeighbor import kNearestNeighbor

# load in data
filesToLoad = {'images': 'ORL/orl_data.mat', 'labels': 'ORL/orl_lbls.mat'}
x_train, x_test, y_train, y_test = loadORL(filesToLoad)

scaler = StandardScaler()

# Standardizing the features
scaler.fit(x_test)
x_test = scaler.transform(x_test)
x_train = scaler.transform(x_train)

# Make an instance of the Model
pca = PCA(n_components=2)

# apply PCA inorder to get fewer dimensions to work with
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.fit_transform(x_test)

# making our predictions
predictions = []
# plt.scatter(x_train, y_train)

#KNearestNeighbor
#kNearestNeighbor(x_train, y_train, x_test, predictions, 3)

# transform the list into an array
predictions = np.asarray(predictions)

# evaluating accuracy
#accuracy = accuracy_score(y_test, predictions)
#print('The accuracy of our KNearestNeighbor classifier is ' + str(accuracy * 100))

# Perceptron Multiclass Least Mean Square
perc_lms = Perceptron_multiclass_lms();
perc_lms.train(x_train, y_train)
predictions = perc_lms.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
print('The accuracy of our Perceptron Multiclass Least Mean Square classifier is ' + str(accuracy * 100))



# Perceptron Multiclass Least Mean Square
perc_pro = Perceptron_multiclass_backpropagation();
perc_pro.train(x_train, y_train)
predictions = perc_lms.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
print('The accuracy of our Perceptron Multiclass Least Mean Square classifier is ' + str(accuracy * 100))

# NearestCentroid
# Normal data
ncc = NearestCentroid()
ncc.fit(x_train, y_train)
# get the model accuracy
predictions = ncc.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
print('The accuracy of our NearestCentroid classifier is ' + str(accuracy * 100))

# 2d data
ncc1 = NearestCentroid()
ncc1.fit(x_train_pca, y_train)
# get the model accuracy
predictions = ncc1.predict(x_test_pca)
accuracy = accuracy_score(y_test, predictions)
print('The accuracy of our NearestCentroid classifier is ' + str(accuracy * 100))

# Nearest subclass classifier (2 subclasses):
nsc = NearestSubclassCentroid()
nsc.fit(x_train, y_train, 2)
predictions = nsc.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
print('The accuracy of our NearestSubclassCentroid classifier is ' + str(accuracy * 100))

# Nearest subclass classifier (3 subclasses):
nsc3 = NearestSubclassCentroid()
nsc3.fit(x_train, y_train, 3)
predictions = nsc3.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
print('The accuracy of our NearestSubclassCentroid classifier is ' + str(accuracy * 100))

# Nearest subclass classifier (5 subclasses):
nsc5 = NearestSubclassCentroid()
nsc5.fit(x_train, y_train, 5)
predictions = nsc5.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
print('The accuracy of our NearestSubclassCentroid classifier is ' + str(accuracy * 100))