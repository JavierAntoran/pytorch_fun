from __future__ import print_function

from utils import *

def read_train_cifar10():
    xtr = []
    ytr = []
    for j in range(5):
        d = load_obj('../data/cifar-10-batches-py/data_batch_%d'  % (j+1) )
        xtr.append(d['data'])
        ytr.append(d['labels'])
    xtr = np.concatenate(xtr)
    ytr = np.concatenate(ytr)
    ytr = np.asarray(ytr, dtype=np.int64)
    return xtr, ytr

def read_test_cifar10():
    d = load_obj('../data/cifar-10-batches-py/test_batch')
    xts = d['data']
    yts = d['labels']
    yts = np.asarray(yts, dtype=np.int64)
    return xts, yts

def read_train_test_cifar10():
    xtr, ytr = read_train_cifar10()
    xts, yts = read_test_cifar10()
    return xtr, ytr, xts, yts

# ----------------------------------------------------------------------------------------------------------------------

def read_train_cifar100():
    d = load_obj('../data/cifar-100-python/train')
    xtr = d['data']
    ytr = d['fine_labels']
    ytr = np.asarray(ytr, dtype=np.int64)
    return xtr, ytr

def read_test_cifar100():
    d = load_obj('../data/cifar-100-python/test')
    xts = d['data']
    yts = d['fine_labels']
    yts = np.asarray(yts, dtype=np.int64)
    return xts, yts

def read_train_test_cifar100():
    xtr, ytr = read_train_cifar100()
    xts, yts = read_test_cifar100()
    return xtr, ytr, xts, yts


