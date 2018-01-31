from __future__ import print_function
import time
import numpy as np
from src.utils import *
from src.read_mnist import *

np.random.seed(1337)  # for reproducibility


## ---------------------------------------------------------------------------------------------------------------------
# read data
x_test, y_test = read_test_mnist()

print('  x_test: %s' % str(x_test.shape))

## ---------------------------------------------------------------------------------------------------------------------
# train config
n_epochs = 100
batch_size = 300

## ---------------------------------------------------------------------------------------------------------------------
# net dims
input_dim = x_test.shape[1]
output_dim = y_test.shape[1]
y_test = np.argmax(y_test, axis=1)

## ---------------------------------------------------------------------------------------------------------------------
# tensorflow
from net import Net
net = Net(input_dim, output_dim)
net.load('model/theta_best')

## ---------------------------------------------------------------------------------------------------------------------
# test
cprint('c','\nTest MNIST:')

cost_test = 0
err_test = 0

nb_samples_test = x_test.shape[0]

# ----
tic = time.time()
for ind in generate_ind_batch(nb_samples_test, batch_size):
    cost, err =  net.eval(x_test[ind], y_test[ind])

    cost_test += cost / float(nb_samples_test) * float(len(ind))
    err_test += err / float(nb_samples_test)

toc = time.time()

# ----
print("Jts = %f, err = %f" % (cost_test, err_test), end="")
cprint('r', '   time: %f seconds\n' % (toc - tic))
