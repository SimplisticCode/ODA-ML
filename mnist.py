from sklearn import decomposition
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler

from MNIST.loadImages import loadMNISTImages, loadMNISTLabels
from kNearestNeighbor import kNearestNeighbor

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


scaler = StandardScaler()

# Standardizing the features
scaler.fit(x_test)
x_test = scaler.transform(x_test)
x_train = scaler.transform(x_train)

# Make an instance of the Model
pca = decomposition.PCA(.95)

# apply PCA inorder to get fewer dimensions to work with
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.fit_transform(x_test)

# making our predictions
predictions = []

kNearestNeighbor(x_train[:5000], y_train[:5000], x_test[:1000], predictions, 1)

# transform the list into an array
predictions = np.asarray(predictions)

# evaluating accuracy
accuracy = accuracy_score(y_test[:1000], predictions)
print('\nThe accuracy of our classifier is %d%%' % accuracy * 100)

# train using K-NN
ncc = NearestCentroid()
ncc.fit(x_train, y_train)
# get the model accuracy
modelscore = ncc.score(x_test, y_test)
