from __future__ import print_function
import time, sys
import numpy as np
from src.utils import *
from src.read_mnist import read_train_dev_mnist

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.datasets as datasets

np.random.seed(1337)  # for reproducibility

## ---------------------------------------------------------------------------------------------------------------------
# read data
x_train, y_train, x_dev, y_dev = read_train_dev_mnist()

print('  x_train: %s' % str(x_train.shape))
print('  x_dev: %s' % str(x_dev.shape))

## ---------------------------------------------------------------------------------------------------------------------
# train config
n_epochs = 100
batch_size = 300

## ---------------------------------------------------------------------------------------------------------------------
# net dims
input_dim = x_train.shape[1]
output_dim = y_train.shape[1]

print('  input_dim: %d' % input_dim)
print('  output_dim: %d' % output_dim)
y_train = np.argmax(y_train, axis=1)
y_dev = np.argmax(y_dev, axis=1)

# --------------------

from dataset import Dataset
import torch.utils.data as data
trainset = Dataset(x_train, y_train)
trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=6)


# --------------------
from net import Net
net = Net(input_dim, output_dim)

## ---------------------------------------------------------------------------------------------------------------------
# train
cprint('c','\nTrain MNIST:')
cost_train = np.zeros(n_epochs)
cost_dev = np.zeros(n_epochs)
err_dev = np.zeros(n_epochs)

nb_samples_train = x_train.shape[0]
nb_samples_dev = x_dev.shape[0]

best_cost = np.inf

for i in range(n_epochs):
    net.set_mode_train(True)
    # ---- W
    tic = time.time()
    for x, y in trainloader:
        loss = net.fit(x, y)
        cost_train[i] += loss / float(nb_samples_train) * float(len(x))

    # for ind in generate_ind_batch(nb_samples_train, batch_size):
    #     loss = net.fit(x_train[ind], y_train[ind])
    #     cost_train[i] += loss / float(nb_samples_train) * float(len(ind))

    toc = time.time()

    # ---- print
    print("it %d/%d, Jtr = %f, " % (i, n_epochs, cost_train[i]), end="")
    cprint('r','   time: %f seconds\n' % (toc - tic))
    sys.stdout.flush()

    # ---- dev
    if i % 5 == 0:
        net.set_mode_train(False)
        for ind in generate_ind_batch(nb_samples_dev, batch_size, random=True):
            cost, err =  net.eval(x_dev[ind], y_dev[ind])
            cost_dev[i] += cost / float(nb_samples_dev) * float(len(ind))
            err_dev[i] += err / float(nb_samples_dev)
        cprint('g','    Jdev = %f, err = %f\n' % (cost_dev[i], err_dev[i]))
        if cost_dev[i] < best_cost:
            best_cost = cost_dev[i]
            net.save('model/theta_best')

## ---------------------------------------------------------------------------------------------------------------------
# save model
net.save('model/theta_last')

## ---------------------------------------------------------------------------------------------------------------------
# fig cost vs its
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.figure()
plt.plot(cost_train, 'r')
plt.plot(range(0, n_epochs, 5), cost_dev[::5], 'bo--')
plt.ylabel('J')
plt.xlabel('it')
plt.grid(True)
# plt.show(block=False)
plt.savefig('train_cost.png')
