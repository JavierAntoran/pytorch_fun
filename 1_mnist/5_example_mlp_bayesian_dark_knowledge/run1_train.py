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
n_epochs = 200
batch_size = 64

## ---------------------------------------------------------------------------------------------------------------------
# net dims
input_dim = x_train.shape[1]
nb_units = 500
output_dim = y_train.shape[1]

y_train = np.argmax(y_train, axis=1)
y_dev = np.argmax(y_dev, axis=1)

## ---------------------------------------------------------------------------------------------------------------------
# pytorch

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1,  16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.fc1 = nn.Linear(512, 500)
        self.fc1_bn = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 500)
        self.fc2_bn = nn.BatchNorm1d(500)
        self.fc3 = nn.Linear(500, 500)
        self.fc3_bn = nn.BatchNorm1d(500)
        self.fc4 = nn.Linear(500, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        # -----
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # -----
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # -----
        x = x.view(-1, 512)
        # -----
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # -----
        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # -----
        x = self.fc3(x)
        x = self.fc3_bn(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        # -----
        x = self.fc4(x)
        x = F.log_softmax(x)
        return x

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1,  16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.fc1 = nn.Linear(512, 500)
        self.fc1_bn = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 500)
        self.fc2_bn = nn.BatchNorm1d(500)
        self.fc3 = nn.Linear(500, 500)
        self.fc3_bn = nn.BatchNorm1d(500)
        self.fc4 = nn.Linear(500, 10)

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
        x = self.fc3_bn(x)
        x = F.relu(x)
        # -----
        x = self.fc4(x)
        x = F.log_softmax(x)
        return x

model1 = Net1()
model1.cuda()
model2 = Net2()
model2.cuda()

optimizer1 = torch.optim.RMSprop(model1.parameters(), lr=0.0001)
optimizer2 = torch.optim.RMSprop(model2.parameters(), lr=0.0001)

## ---------------------------------------------------------------------------------------------------------------------
# train
cprint('c','\nTrain MNIST:')
cost_train1 = np.zeros(n_epochs)
cost_train2 = np.zeros(n_epochs)

cost_dev = np.zeros(n_epochs)
err_dev = np.zeros(n_epochs)

nb_samples_train = x_train.shape[0]
nb_samples_dev = x_dev.shape[0]

for i in range(n_epochs):

    model1.train()
    model2.train()
    # ---- W
    tic = time.time()
    for ind in generate_ind_batch(nb_samples_train, batch_size):
        x, y = torch.from_numpy(x_train[ind]), torch.from_numpy(y_train[ind])
        x, y = x.cuda(), y.cuda()
        x, y = Variable(x), Variable(y)

        optimizer1.zero_grad()
        out1 = model1(x)
        loss1 = F.nll_loss(out1, y)
        loss1.backward()
        optimizer1.step()

        y_pred = out1.exp().detach()

        optimizer2.zero_grad()
        out2 = model2(x)
        loss2 = - (y_pred * out2).sum(1).mean()
        loss2.backward()
        optimizer2.step()

        cost_train1[i] += loss1.data[0] / float(nb_samples_train) * float(len(ind))
        cost_train2[i] += loss2.data[0] / float(nb_samples_train) * float(len(ind))
    toc = time.time()

    # ---- print
    print("it %d/%d, Jtr1 = %f, Jtr2 = %f, " % (i, n_epochs, cost_train1[i],cost_train2[i]), end="")
    cprint('r','   time: %f seconds\n' % (toc - tic))

    # ---- dev
    if i % 5 == 0:
        model2.eval() # bn not update
        for ind in generate_ind_batch(nb_samples_dev, batch_size, random=False):
            x, y = torch.from_numpy(x_dev[ind]), torch.from_numpy(y_dev[ind])
            x, y = x.cuda(), y.cuda()
            x, y = Variable(x), Variable(y)

            out = model2(x)
            loss = F.nll_loss(out, y)

            pred = out.data.max(1)[1] # get the index of the max log-probability
            err = pred.ne(y.data).cpu().sum()

            cost_dev[i] += loss.data[0]  / float(nb_samples_dev) * float(len(ind))
            err_dev[i] += err / float(nb_samples_dev)

        cprint('g','    Jdev = %f, err = %f\n' % (cost_dev[i], err_dev[i]))

## ---------------------------------------------------------------------------------------------------------------------
# save model
cprint('c','Writting model1.dat')
# torch.save(model1.state_dict(), 'model.dat') # only weights
torch.save(model1, 'model1.dat') # complete object

cprint('c','Writting model2.dat')
# torch.save(model2.state_dict(), 'model.dat') # only weights
torch.save(model2, 'model2.dat') # complete object

## ---------------------------------------------------------------------------------------------------------------------
# fig cost vs its
plt.figure()
plt.plot(cost_train1, 'r')
plt.plot(cost_train2, 'k')
plt.plot(range(0, n_epochs, 5), cost_dev[::5], 'bo--')
plt.ylabel('J')
plt.xlabel('it')
plt.grid(True)
# plt.show(block=False)
plt.savefig('train_cost.png')
