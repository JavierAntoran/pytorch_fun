from __future__ import print_function
from __future__ import division
import time, sys
from src.utils import *
from src.read_mnist import *
from src.datafeed import *
import numpy as np

import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms

np.random.seed(1337)  # for reproducibility

## ----------------------------------------------------------------------------------------------------------------
# read data
cprint('c','\nData:')
x_train, y_train, x_dev, y_dev = read_train_dev_mnist()

x_train = np.reshape(x_train,(-1,1,28,28))
x_dev = np.reshape(x_dev,(-1,1,28,28))

y_train = np.argmax(y_train, axis=1)
y_dev = np.argmax(y_dev, axis=1) 

print('  x_train: %s' % str(x_train.shape))
print('  x_dev: %s' % str(x_dev.shape))

## ---------------------------------------------------------------------------------------------------------------------
# train config
nb_epochs = 100
batch_size = 300

## ---------------------------------------------------------------------------------------------------------------------
# net dims
input_dim = x_train.shape[1:]
output_dim = np.max(y_train) + 1

print('  input_dim: %s' % str(input_dim))
print('  output_dim: %d' % output_dim)

# --------------------
from net import Net
net = Net(input_dim, output_dim)
net.load('model/theta_best')

## ---------------------------------------------------------------------------------------------------------------------
# train
cprint('c','\nTrain:')

nb_samples_train = x_train.shape[0]
nb_samples_dev = x_dev.shape[0]

net.set_mode_train(False)


e_train = []
for ind in generate_ind_batch(len(x_train), batch_size, random=False):
    x, y = x_train[ind], y_train[ind]
    e  = net.extract(x).cpu().numpy()
    e_train.append(e)
e_train = np.concatenate(e_train)

e_dev = []
for ind in generate_ind_batch(len(x_dev), batch_size, random=False):
    x, y = x_dev[ind], y_dev[ind]
    e  = net.extract(x).cpu().numpy()
    e_dev.append(e)
e_dev = np.concatenate(e_dev)
    


## ---------------------------------------------------------------------------------------------------------------------
# fig cost vs its
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']

plt.figure() 
for i in range(10):
    plt.plot(e_train[y_train == i, 0], e_train[y_train == i, 1], '.', c=c[i])
        
plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc = 'upper right')
plt.grid(True)
plt.savefig('results/e_train.png')



plt.figure() 
for i in range(10):
    plt.plot(e_dev[y_dev == i, 0], e_dev[y_dev == i, 1], '.', c=c[i])
        
plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc = 'upper right')
plt.grid(True)
plt.savefig('results/e_dev.png')
