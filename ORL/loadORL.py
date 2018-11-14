import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split

# This function loads in the ORL data and split in a test and training data set
def loadORL(files, randomSeed=None):
    orl_data = scipy.io.loadmat(files['images'])
    orl_label = scipy.io.loadmat(files['labels'])
    nImg = 400  # num of images
    nC = 1200  # num of dimension/pixels
    images_array = np.zeros((nImg, nC))
    images_array = np.asarray(orl_data['data'].transpose().reshape(nImg, nC))
    labels_array = np.asarray(orl_label['lbls'].reshape(400))

    # split the sample to a training set and a test set
    if randomSeed is not None:
        x_train, x_test, y_train, y_test = train_test_split(images_array, labels_array, test_size=0.3, random_state=randomSeed)
    else:
        x_train, x_test, y_train, y_test = train_test_split(images_array, labels_array, test_size=0.3)
    return x_train, x_test, y_train, y_test

