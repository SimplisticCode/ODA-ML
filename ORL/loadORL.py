import numpy as np
import scipy.io


# This function loads in the ORL data and split in a test and training data set
def loadORL(files):
    orl_data = scipy.io.loadmat(files['images'])
    orl_label = scipy.io.loadmat(files['labels'])
    nImg = 400  # num of images
    nC = 1200  # num of dimension/pixels
    images_array = np.zeros((nImg, nC))
    images_array = np.asarray(orl_data['data'].transpose().reshape(nImg, nC))
    labels_array = np.asarray(orl_label['lbls'].reshape(400))

    # split the sample to a training set and a test set
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(images_array, labels_array, test_size=0.33, random_state=42)
    data = {'train': {'X': x_train,
                      'y': y_train},
            'test': {'X': x_test,
                     'y': y_test}}
    return data
