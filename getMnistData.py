import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler

from kNearestNeighbor import kNearestNeighbor


def get_data():
    import tensorflow.examples.tutorials.mnist
    mnist = tensorflow.examples.tutorials.mnist.input_data.read_data_sets('MNIST_data', one_hot=True)

    x = mnist.data
    y = mnist.target

    # Scale data to [-1, 1] - This is of mayor importance!!!
    x = x/255.0*2 - 1

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.33,
                                                        random_state=42)
    data = {'train': {'X': x_train,
                      'y': y_train},
            'test': {'X': x_test,
                     'y': y_test}}
    return data


data = get_data()

x_train = data['train']['X']
x_test = data['test']['X']
y_train = data['train']['y']
y_test = data['test']['y']
scaler = StandardScaler()

# Standardizing the features
scaler.fit(x_test)
x_test = scaler.transform(x_test)
x_train = scaler.transform(x_train)

# Make an instance of the Model
pca = PCA(.95)

# apply PCA inorder to get fewer dimensions to work with
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.fit_transform(x_test)

# making our predictions
predictions = []

kNearestNeighbor(x_train_pca, y_train, x_test_pca, predictions, 1)

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