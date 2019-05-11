from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from keras.datasets import cifar10
import keras
import os
from keras.utils import np_utils
import scipy.io as sio


def data_cifar10(train_start=0, train_end=50000, test_start=0,
                 test_end=10000):
    """
    Load and preprocess CIFAR10 dataset
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :return: tuple of four arrays containing training data, training labels,
             testing data and testing labels.
    """
    assert isinstance(train_start, int)
    assert isinstance(train_end, int)
    assert isinstance(test_start, int)
    assert isinstance(test_end, int)

    cifar10.load_data()
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

    X_train = X_train[train_start:train_end]
    Y_train = Y_train[train_start:train_end]
    X_test = X_test[test_start:test_end]
    Y_test = Y_test[test_start:test_end]

    X_train = X_train.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.

    # Convert class vectors to binary class matrices.
    num_classes = 10
    Y_train = keras.utils.to_categorical(Y_train, num_classes)
    Y_test = keras.utils.to_categorical(Y_test, num_classes)
    print("load cifar10:")
    print("train examples: %d" % (X_train.shape[0]))
    print("test examples: %d" % (X_test.shape[0]))

    return X_train, Y_train, X_test, Y_test


def data_mnist(datadir='./dataset/mnist', train_start=0, train_end=60000, test_start=0,
               test_end=10000):
    """
    Load and preprocess MNIST dataset
    :param datadir: path to folder where data should be stored
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :return: tuple of four arrays containing training data, training labels,
             testing data and testing labels.
    """
    assert isinstance(train_start, int)
    assert isinstance(train_end, int)
    assert isinstance(test_start, int)
    assert isinstance(test_end, int)

    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets(datadir, one_hot=True, reshape=False)
    X_train = np.vstack((mnist.train.images, mnist.validation.images))
    Y_train = np.vstack((mnist.train.labels, mnist.validation.labels))
    X_test = mnist.test.images
    Y_test = mnist.test.labels

    X_train = X_train[train_start:train_end]
    Y_train = Y_train[train_start:train_end]
    X_test = X_test[test_start:test_end]
    Y_test = Y_test[test_start:test_end]

    print("load mnist:")
    print("train examples:" + str(Y_train.shape))
    print("test examples:" + str(X_test.shape))

    return X_train, Y_train, X_test, Y_test


def data_fmnist(datadir='./dataset/fmnist', train_start=0, train_end=60000, test_start=0,
                test_end=10000):
    """
    Load and preprocess FMNIST dataset
    :param datadir: path to folder where data should be stored
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :return: tuple of four arrays containing training data, training labels,
             testing data and testing labels.
    """
    assert isinstance(train_start, int)
    assert isinstance(train_end, int)
    assert isinstance(test_start, int)
    assert isinstance(test_end, int)

    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets(datadir, one_hot=True, reshape=False)
    X_train = np.vstack((mnist.train.images, mnist.validation.images))
    Y_train = np.vstack((mnist.train.labels, mnist.validation.labels))
    X_test = mnist.test.images
    Y_test = mnist.test.labels

    X_train = X_train[train_start:train_end]
    Y_train = Y_train[train_start:train_end]
    X_test = X_test[test_start:test_end]
    Y_test = Y_test[test_start:test_end]

    print("load fmnist:")
    print("train examples: %d" % (X_train.shape[0]))
    print("test examples: %d" % (X_test.shape[0]))

    return X_train, Y_train, X_test, Y_test


def data_svhn(datadir='./dataset/svhn', train_start=0, train_end=60000, test_start=0,
              test_end=10000):
    """
    Load and preprocess SVHN dataset
    :param datadir: path to folder where data should be stored
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :return: tuple of four arrays containing training data, training labels,
             testing data and testing labels.
    """
    assert isinstance(train_start, int)
    assert isinstance(train_end, int)
    assert isinstance(test_start, int)
    assert isinstance(test_end, int)

    train_path = "train_32x32.mat"
    train_path = os.path.join(datadir, train_path)
    train = sio.loadmat(train_path)
    X_train = train['X']
    y_train = train['y']
    y_train[y_train == 10] = 0

    y_train = y_train.ravel()

    X_train = np.transpose(X_train, (3, 0, 1, 2))
    X_train = X_train.astype('float32') / 255
    Y_train = np_utils.to_categorical(y_train)

    test_path = "test_32x32.mat"
    test_path = os.path.join(datadir, test_path)
    test = sio.loadmat(test_path)
    X_test = test['X']
    y_test = test['y']
    y_test[y_test == 10] = 0

    y_test = y_test.ravel()

    X_test = np.transpose(X_test, (3, 0, 1, 2))
    X_test = X_test.astype('float32') / 255
    Y_test = np_utils.to_categorical(y_test)

    print("load fmnist:")
    print("train examples: %d" % (X_train.shape[0]))
    print("test examples: %d" % (X_test.shape[0]))

    return X_train, Y_train, X_test, Y_test
