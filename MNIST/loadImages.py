import struct
import struct as st
from collections import Counter

import numpy as np
from sklearn import decomposition
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestCentroid


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




