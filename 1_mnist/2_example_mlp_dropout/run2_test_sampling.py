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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 500)
        self.fc4 = nn.Linear(500, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        # -----
        x = self.fc2(x)
        x = F.relu(x)
        # -----
        x = self.fc3(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # -----
        x = self.fc4(x)
        x = F.log_softmax(x)
        return x

model = Net()
model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


## ---------------------------------------------------------------------------------------------------------------------
# load model
cprint('c','Reading model.dat')
model.load_state_dict(torch.load('model.dat'))
# model = torch.load('model.dat') # complete object

## ---------------------------------------------------------------------------------------------------------------------
# train
cprint('c','\nTest MMIST:')
cost_test = 0
err_test = 0

nb_samples_test = x_test.shape[0]

model.train()  # rand dropout
# ----
nb_samples = 100
tic = time.time()
for ind in generate_ind_batch(nb_samples_test, batch_size):
    x, y = torch.from_numpy(x_test[ind]), torch.from_numpy(y_test[ind])
    x, y = x.cuda(), y.cuda()
    x, y = Variable(x), Variable(y)

    out = 0.
    loss = 0.
    for j in range(nb_samples):
        out += model(x).exp() / float(nb_samples)
        loss += F.nll_loss(out, y) / float(nb_samples)

    pred = out.data.max(1)[1]
    err = pred.ne(y.data).cpu().sum()

    cost_test += loss.data[0]  / float(nb_samples_test) * float(len(ind))
    err_test += err / float(nb_samples_test)

toc = time.time()

# ----
print("Jts = %f, err = %f" % (cost_test, err_test), end="")
cprint('r', '   time: %f seconds\n' % (toc - tic))
