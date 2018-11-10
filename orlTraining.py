import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler

from ORL.loadORL import loadORL
from kNearestNeighbor import kNearestNeighbor

#load in data
filesToLoad = {'images': 'ORL/orl_data.mat', 'labels': 'ORL/orl_lbls.mat'}
data = loadORL(filesToLoad)

x_train = data['train']['X']
x_test = data['test']['X']
y_train = data['train']['y']
y_test = data['test']['y']
scaler = StandardScaler()

# Standardizing the features
#scaler.fit(x_test)
#x_test = scaler.transform(x_test)
#x_train = scaler.transform(x_train)

# Make an instance of the Model
#pca = PCA(.95)

# apply PCA inorder to get fewer dimensions to work with
#x_train_pca = pca.fit_transform(x_train)
#x_test_pca = pca.fit_transform(x_test)

# making our predictions
predictions = []

kNearestNeighbor(x_train, y_train, x_test, predictions, 3)

# transform the list into an array
predictions = np.asarray(predictions)

# evaluating accuracy
accuracy = accuracy_score(y_test, predictions)
print('\nThe accuracy of our classifier is %d%%' % accuracy * 100)

# train using K-NN
ncc = NearestCentroid()
ncc.fit(x_train, y_train)
# get the model accuracy
modelscore = ncc.score(x_test, y_test)
