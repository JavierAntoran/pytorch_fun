from __future__ import print_function
import time, sys
import numpy as np

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

from src.utils import *
from src.read_mnist import read_train_dev_mnist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
nb_units = 500
output_dim = y_train.shape[1]

y_train = np.argmax(y_train, axis=1)
y_dev = np.argmax(y_dev, axis=1)

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
# train
cprint('c','\nTrain MNIST:')
cost_train = np.zeros(n_epochs)
cost_dev = np.zeros(n_epochs)
err_dev = np.zeros(n_epochs)

nb_samples_train = x_train.shape[0]
nb_samples_dev = x_dev.shape[0]


for i in range(n_epochs):
    # ---- W
    model.train()
    tic = time.time()
    for ind in generate_ind_batch(nb_samples_train, batch_size):
        x, y = torch.from_numpy(x_train[ind]), torch.from_numpy(y_train[ind])
        x, y = x.cuda(), y.cuda()
        x, y = Variable(x), Variable(y)

        optimizer.zero_grad()
        out = model(x)
        loss = F.nll_loss(out, y)
        loss.backward()
        optimizer.step()

        cost_train[i] += loss.data[0] / float(nb_samples_train) * float(len(ind))
    toc = time.time()

    # ---- print
    print("it %d/%d, Jtr = %f, " % (i, n_epochs, cost_train[i]), end="")
    cprint('r','   time: %f seconds\n' % (toc - tic))

    # ---- dev
    if i % 5 == 0:
        model.eval() # deterministic dropout
        for ind in generate_ind_batch(nb_samples_dev, batch_size, random=False):
            x, y = torch.from_numpy(x_dev[ind]), torch.from_numpy(y_dev[ind])
            x, y = x.cuda(), y.cuda()
            x, y = Variable(x), Variable(y)

            out = model(x)
            loss = F.nll_loss(out, y)

            pred = out.data.max(1)[1] # get the index of the max log-probability
            err = pred.ne(y.data).cpu().sum()

            cost_dev[i] += loss.data[0]  / float(nb_samples_dev) * float(len(ind))
            err_dev[i] += err / float(nb_samples_dev)

        cprint('g','    Jdev = %f, err = %f\n' % (cost_dev[i], err_dev[i]))

## ---------------------------------------------------------------------------------------------------------------------
# save model
cprint('c','Writting model.dat')
torch.save(model.state_dict(), 'model.dat') # only weights
# torch.save(model, 'model.dat') # complete object

## ---------------------------------------------------------------------------------------------------------------------
# fig cost vs its
plt.figure()
plt.plot(cost_train, 'r')
plt.plot(range(0, n_epochs, 5), cost_dev[::5], 'bo--')
plt.ylabel('J')
plt.xlabel('it')
plt.grid(True)
# plt.show(block=False)
plt.savefig('train_cost.png')
