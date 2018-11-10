
num_classes = 10


# the data, split between train and test sets
filenameTrain = {'images': 'MNIST/train-images.idx3-ubyte', 'labels': 'MNIST/train-labels.idx1-ubyte'}
filenameTest = {'images': 'MNIST/t10k-images.idx3-ubyte', 'labels': 'MNIST/t10k-labels.idx1-ubyte'}
x_train = loadMNISTImages(filenameTrain)
x_test = loadMNISTImages(filenameTest)
y_train = loadMNISTLabels(filenameTrain)
y_test = loadMNISTLabels(filenameTest)


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



# train using K-NN
# ncc = NearestCentroid()
# ncc.fit(x_train, y_train)
# get the model accuracy
# modelscore = ncc.score(x_test, y_test)

# print(modelscore)

# train using K-NN
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(x_train, y_train)
# get the model accuracy
# model_score = knn.score(x_test, y_test)

# print(model_score)
