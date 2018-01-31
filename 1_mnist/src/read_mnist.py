from __future__ import print_function
import cPickle, gzip
import numpy as np
from utils import one_hot, cprint


def read_mnist():
    # wget http://deeplearning.net/data/mnist/mnist.pkl.gz
    cprint('c','Loading MNIST data')
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    (x_train, y_train_ind), (x_dev, y_dev_ind), (x_test, y_test_ind) = cPickle.load(f)
    f.close()

    y_train = one_hot(y_train_ind, 10)
    y_dev = one_hot(y_dev_ind, 10)
    y_test = one_hot(y_test_ind, 10)

    x_train = np.asarray(x_train, dtype=np.float32)
    x_dev = np.asarray(x_dev, dtype=np.float32)
    x_test = np.asarray(x_test, dtype=np.float32)

    y_train = np.asarray(y_train, dtype=np.float32)
    y_dev = np.asarray(y_dev, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32)

    return x_train, y_train, x_dev, y_dev, x_test, y_test



def read_train_dev_mnist():
    # wget http://deeplearning.net/data/mnist/mnist.pkl.gz
    cprint('c','Loading MNIST data')
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    (x_train, y_train_ind), (x_dev, y_dev_ind), (x_test, y_test_ind) = cPickle.load(f)
    f.close()

    y_train = one_hot(y_train_ind, 10)
    y_dev = one_hot(y_dev_ind, 10)
    y_test = one_hot(y_test_ind, 10)

    x_train = np.asarray(x_train, dtype=np.float32)
    x_dev = np.asarray(x_dev, dtype=np.float32)

    y_train = np.asarray(y_train, dtype=np.float32)
    y_dev = np.asarray(y_dev, dtype=np.float32)

    return x_train, y_train, x_dev, y_dev

def read_test_mnist():
    # wget http://deeplearning.net/data/mnist/mnist.pkl.gz
    cprint('c','Loading MNIST data')
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    _, _, (x_test, y_test_ind) = cPickle.load(f)
    f.close()
    y_test = one_hot(y_test_ind, 10)
    x_test = np.asarray(x_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32)
    return x_test, y_test



