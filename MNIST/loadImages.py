import struct
import struct as st
from collections import Counter

import numpy as np
from sklearn import decomposition


def loadMNISTImages(filename=None):
    # loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
    # the raw MNIST images

    train_imagesfile = open(filename, 'r+b')
    train_imagesfile.seek(0)
    magic = st.unpack('>4B', train_imagesfile.read(4))
    nImg = st.unpack('>I', train_imagesfile.read(4))[0]  # num of images
    nR = st.unpack('>I', train_imagesfile.read(4))[0]  # num of rows
    nC = st.unpack('>I', train_imagesfile.read(4))[0]  # num of column

    images_array = np.zeros((nImg, nR, nC))
    nBytesTotal = nImg * nR * nC * 1  # since each pixel data is 1 byte
    images_array = 255 - np.asarray(st.unpack('>' + 'B' * nBytesTotal, train_imagesfile.read(nBytesTotal))).reshape(
        (nImg, nR, nC))
    return images_array


def loadMNISTLabels(filename=None):
    # loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
    # the raw MNIST images
    with open(filename, 'rb') as labelfile:
        datastr = labelfile.read()

    index = 0
    mgc_num, label_num = struct.unpack_from('>II', datastr, index)
    index += struct.calcsize('>II')

    label = struct.unpack_from('{}B'.format(label_num), datastr, index)
    index += struct.calcsize('{}B'.format(label_num))

    label_array = np.array(label)
    return label_array


def predict(X_train, y_train, x_test, k):
    # create list for distances and targets
    distances = []
    targets = []

    for i in range(len(X_train)):
        # first we compute the euclidean distance
        distance = np.sqrt(np.sum(np.square(x_test - X_train[i, :])))
        # add it to list of distances
        distances.append([distance, i])

    # sort the list
    distances = sorted(distances)

    # make a list of the k neighbors' targets
    for i in range(k):
        index = distances[i][1]
        targets.append(y_train[index])

    # return most common target
    return Counter(targets).most_common(1)[0][0]


def kNearestNeighbor(X_train, y_train, X_test, predictions, k):
    # train on the input data
    train(X_train, y_train)

    # loop over all observations
    for i in range(len(X_test)):
        predictions.append(predict(X_train, y_train, X_test[i, :], k))


def train(X_train, y_train):
    # do nothing
    return


# the data, split between train and test sets
filenameTrain = {'images': 'train-images.idx3-ubyte', 'labels': 'train-labels.idx1-ubyte'}
filenameTest = {'images': 't10k-images.idx3-ubyte', 'labels': 't10k-labels.idx1-ubyte'}
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

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# apply PCA inorder to get fewer dimensions to work with
x_train_pca = decomposition.PCA(n_components=2).fit_transform(X)
x_test_pca = decomposition.PCA(n_components=2).fit_transform(X)


# making our predictions
predictions = []

kNearestNeighbor(x_train, y_train, x_test, predictions, 7)

# transform the list into an array
predictions = np.asarray(predictions)

# evaluating accuracy
accuracy = accuracy_score(y_test, predictions)
print('\nThe accuracy of our classifier is %d%%' % accuracy * 100)


#train using K-NN
ncc = NearestCentroid()
ncc.fit(x_train, y_train)
#get the model accuracy
modelscore = ncc.score(x_test, y_test)

