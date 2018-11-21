from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
from MNIST.loadImages import loadMNISTImages, loadMNISTLabels
from NearestCentroid import NearestCentroid
from NearestSubclassCentroid import NearestSubclassCentroid
from kNearestNeighbor import kNearestNeighbor

import matplotlib.pyplot as plt

#load in data
filenameTrain = {'images': 'MNIST/train-images.idx3-ubyte', 'labels': 'MNIST/train-labels.idx1-ubyte'}
filenameTest = {'images': 'MNIST/t10k-images.idx3-ubyte', 'labels': 'MNIST/t10k-labels.idx1-ubyte'}
x_train = loadMNISTImages(filenameTrain['images'])
x_test = loadMNISTImages(filenameTest['images'])
y_train = loadMNISTLabels(filenameTrain['labels'])
y_test = loadMNISTLabels(filenameTest['labels'])

# Reshape the data
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)

#mlp.fit(x_train,y_train)
#predictions = mlp.predict(x_test)

#print(confusion_matrix(y_test,predictions))
#print(classification_report(y_test,predictions))



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

#KNearestNeighbor
# making our predictions
predictions = []

# On original data
#kNearestNeighbor(x_train[:5000], y_train[:5000], x_test[:1000], predictions, 1)

# On 2d data
#kNearestNeighbor(x_train_pca[:5000], y_train[:5000], x_test_pca[:1000], predictions, 1)

# transform the list into an array
predictions = np.asarray(predictions)

# evaluating accuracy
#accuracy = accuracy_score(y_test[:1000], predictions)
#print('\nThe accuracy of our classifier is %d%%' % accuracy * 100)

# train using K-NN

#Nearest Centroid
#Normal data:
ncc = NearestCentroid()
ncc.fit(x_train, y_train)
# get the model accuracy
predictions = ncc.predict(x_test)
predictions = np.asarray(predictions)

accuracy = accuracy_score(y_test, predictions)
print('The accuracy of our NearestCentroid classifier is ' + str(accuracy * 100))


#Visualize data:
plt.scatter(x_test_pca[:,0], x_test_pca[:,1], c=y_test, label=y_test)
plt.legend()
plt.show()


# Nearest subclass classifier (2 subclasses):
nsc = NearestSubclassCentroid()
nsc.fit(x_train, y_train, 2)
predictions = nsc.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
print('The accuracy of our NearestSubclassCentroid (with 2 subclasses) classifier is ' + str(accuracy * 100))

# Nearest subclass classifier (3 subclasses):
nsc3 = NearestSubclassCentroid()
nsc3.fit(x_train, y_train, 3)
predictions = nsc3.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
print('The accuracy of our NearestSubclassCentroid (with 3 subclasses) classifier is ' + str(accuracy * 100))

# Nearest subclass classifier (5 subclasses):
nsc5 = NearestSubclassCentroid()
nsc5.fit(x_train, y_train, 5)
predictions = nsc5.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
print('The accuracy of our NearestSubclassCentroid (with 5 subclasses) classifier is ' + str(accuracy * 100))

