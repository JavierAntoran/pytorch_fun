from __future__ import print_function
import time, sys
import numpy as np

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

from src.utils import *
from src.read_mnist import read_test_mnist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
nb_units = 500
output_dim = y_test.shape[1]

y_test = np.argmax(y_test, axis=1)

## ---------------------------------------------------------------------------------------------------------------------
# pytorch

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1,  16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.fc1 = nn.Linear(512, 500)
        self.fc1_bn = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 500)
        self.fc2_bn = nn.BatchNorm1d(500)
        self.fc3 = nn.Linear(500, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        # -----
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        # -----
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        # -----
        x = x.view(-1, 512)
        # -----
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = F.relu(x)
        # -----
        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = F.relu(x)
        # -----
        x = self.fc3(x)
        x = F.log_softmax(x)
        return x


# model2 = Net2()
# model2.cuda()

## ---------------------------------------------------------------------------------------------------------------------
# load model
cprint('c','Reading model2.dat')
# model22.load_state_dict(torch.load('model2.dat')) # only weights
model = torch.load('model2.dat') # complete object

## ---------------------------------------------------------------------------------------------------------------------
# train
cprint('c','\nTest MMIST:')
cost_test = 0
err_test = 0

nb_samples_test = x_test.shape[0]

model.eval() # bn not update
# ----
tic = time.time()
for ind in generate_ind_batch(nb_samples_test, batch_size):
    x, y = torch.from_numpy(x_test[ind]), torch.from_numpy(y_test[ind])
    x, y = x.cuda(), y.cuda()
    x, y = Variable(x), Variable(y)

    out = model(x)
    loss = F.nll_loss(out, y)

    pred = out.data.max(1)[1]  # get the index of the max log-probability
    err = pred.ne(y.data).cpu().sum()

    cost_test += loss.data[0]  / float(nb_samples_test) * float(len(ind))
    err_test += err / float(nb_samples_test)

toc = time.time()

# ----
print("Jts = %f, err = %f" % (cost_test, err_test), end="")
cprint('r', '   time: %f seconds\n' % (toc - tic))
