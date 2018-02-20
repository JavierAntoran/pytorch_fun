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
x_train, y_train, x_dev, y_dev = read_train_test_mnist()

x_train = np.reshape(x_train,(-1,1,28,28))
x_dev = np.reshape(x_dev,(-1,1,28,28))

y_train = np.argmax(y_train, axis=1)
y_dev = np.argmax(y_dev, axis=1) 

print('  x_train: %d' % len(x_train))
print('  x_dev: %d' % len(x_dev))

## ---------------------------------------------------------------------------------------------------------------------
# train config
nb_epochs = 50
batch_size = 128

## ---------------------------------------------------------------------------------------------------------------------
transform_train = transforms.Compose([
    #transforms.RandomCrop(28, padding=4),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = Datafeed(x_train, y_train, transform=transform_train)
trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3)

testset = Datafeed(x_dev, y_dev, transform=transform_test)
testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=3)

## ---------------------------------------------------------------------------------------------------------------------
# net dims
input_dim = x_train.shape[1:]
output_dim = np.max(y_train) + 1

print('  input_dim: %s' % str(input_dim))
print('  output_dim: %d' % output_dim)

# --------------------
from net import Net
net = Net(input_dim, output_dim)

## ---------------------------------------------------------------------------------------------------------------------
# train
cprint('c','\nTrain:')


err_train = np.zeros(nb_epochs)
cost_train = np.zeros(nb_epochs)
cost_dev = np.zeros(nb_epochs)
err_dev = np.zeros(nb_epochs)

nb_samples_train = len(x_train)
nb_samples_dev = len(x_dev)

best_cost = np.inf
nb_its_dev = 1
tic0 = time.time()
for i in range(nb_epochs):
    net.set_mode_train(True)
    # ---- W
    tic = time.time()
    #for x, y in trainloader:        
    for ind in generate_ind_batch(nb_samples_train, batch_size):
        x, y = x_train[ind], y_train[ind]
        loss, err  = net.fit(x, y)
        cost_train[i] += loss / nb_samples_train * len(x)
        err_train[i] += err / nb_samples_train

    toc = time.time()

    # ---- print
    print("it %d/%d, Jtr = %f, err = %f, " % (i, nb_epochs, cost_train[i],err_train[i]), end="")
    cprint('r','   time: %f seconds\n' % (toc - tic))
    net.update_lr(i)

    # ---- dev
    if i % nb_its_dev == 0:
        net.set_mode_train(False)
        #for x, y in testloader:
        for ind in generate_ind_batch(nb_samples_dev, batch_size, random=False):
            x, y = x_dev[ind], y_dev[ind]
            cost, err =  net.eval(x, y)
            cost_dev[i] += cost / nb_samples_dev * len(x)
            err_dev[i] += err / nb_samples_dev
        cprint('g','    Jdev = %f, err = %f\n' % (cost_dev[i], err_dev[i]))
        if cost_dev[i] < best_cost:
            best_cost = cost_dev[i]
            net.save('model/theta_best.dat')

toc0 = time.time()
runtime_per_it =  (toc0 - tic0)/float(nb_epochs)
cprint('r','   average time: %f seconds\n' % runtime_per_it)

## ---------------------------------------------------------------------------------------------------------------------
# save model
net.save('model/theta_last.dat')

## ---------------------------------------------------------------------------------------------------------------------
# results
cprint('c','\nRESULTS:')
cost_dev_min = cost_dev[::nb_its_dev].min()
err_dev_min = err_dev[::nb_its_dev].min()
cost_train_min = cost_train.min()
nb_parameters = net.get_nb_parameters()
print('  cost_dev: %f (cost_train %f)' % (cost_dev_min, cost_train_min))
print('  err_dev: %f' % (err_dev_min))
print('  nb_parameters: %d (%s)' % (nb_parameters, humansize(nb_parameters)))
print('  time_per_it: %fs\n' % (runtime_per_it))

with open('results/results.txt','w') as f:
    f.write('%f %f %d %s %f\n' % (err_dev_min, cost_dev_min, nb_parameters, humansize(nb_parameters), runtime_per_it))


## ---------------------------------------------------------------------------------------------------------------------
# fig cost vs its
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.figure()
plt.plot(cost_train, 'r')
plt.plot(range(0, nb_epochs, nb_its_dev), cost_dev[::nb_its_dev], 'bo--')
plt.ylabel('J')
plt.xlabel('it')
plt.grid(True)
# plt.show(block=False)
plt.savefig('results/train_cost.png')


