from __future__ import print_function
from __future__ import division
import time, sys
from src.utils import *
from src.read_mnist import *
from src.datafeed import *
import numpy as np

import torch.utils.data as data
import torchvision.transforms as transforms

np.random.seed(1337)  # for reproducibility

## ----------------------------------------------------------------------------------------------------------------
# read data
cprint('c','\nData:')
_, _, x_test, y_test = read_train_test_mnist()

x_test = np.reshape(x_test,(-1,1,28,28))
y_test = np.argmax(y_test, axis=1)

print('  x_test: %d' % len(x_test))

## ---------------------------------------------------------------------------------------------------------------------
# train config
n_epochs = 100
batch_size = 300

## ---------------------------------------------------------------------------------------------------------------------
# net dims
input_dim = x_test.shape[1:]
output_dim = np.max(y_test) + 1

print('  input_dim: %s' % str(input_dim))
print('  output_dim: %d' % output_dim)

# --------------------
from net import Net
net = Net(input_dim, output_dim)
net.load('model/theta_best.dat')

## ---------------------------------------------------------------------------------------------------------------------
# test
cprint('c','\nTest:')

cost_test = 0
err_test = 0

nb_samples_test = len(x_test)

# ----
tic = time.time()
for ind in generate_ind_batch(nb_samples_test, batch_size):
    cost, err =  net.eval(x_test[ind], y_test[ind])

    cost_test += cost / nb_samples_test * len(ind)
    err_test += err / nb_samples_test

toc = time.time()

# ----
print("Jts = %f, err = %f" % (cost_test, err_test), end="")
cprint('r', '   time: %f seconds\n' % (toc - tic))
