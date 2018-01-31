from __future__ import print_function
import time, sys
from src.utils import *
from src.read_cifar import *
from src.transforms import *
from src.dataset import *
import numpy as np

import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms

np.random.seed(1337)  # for reproducibility

## ----------------------------------------------------------------------------------------------------------------
# read data
x_train, y_train, x_dev, y_dev = read_train_test_cifar10()

x_train = np.reshape(x_train,(-1,3,32,32))
x_dev = np.reshape(x_dev,(-1,3,32,32))

x_train = np.transpose(x_train, (0, 2, 3, 1))
x_dev = np.transpose(x_dev, (0, 2, 3, 1))

print('  x_train: %s' % str(x_train.shape))
print('  x_dev: %s' % str(x_dev.shape))

transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = Datafeed(x_train, y_train, transform=transform_train)
trainloader = data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = Datafeed(x_dev, y_dev, transform=transform_test)
testloader = data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

## ---------------------------------------------------------------------------------------------------------------------
# train config
nb_epochs = 300
batch_size = 128

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
cprint('c','\nTrain CIFAR10:')
cost_train = np.zeros(nb_epochs)
cost_dev = np.zeros(nb_epochs)
err_dev = np.zeros(nb_epochs)

nb_samples_train = x_train.shape[0]
nb_samples_dev = x_dev.shape[0]

best_cost = np.inf
nb_its_dev = 5
tic0 = time.time()
for i in range(nb_epochs):
    net.set_mode_train(True)
    # ---- W
    tic = time.time()
    for x, y in trainloader:
        loss = net.fit(x, y)
        cost_train[i] += loss / float(nb_samples_train) * float(len(x))

    toc = time.time()

    # ---- print
    print("it %d/%d, Jtr = %f, " % (i, nb_epochs, cost_train[i]), end="")
    cprint('r','   time: %f seconds\n' % (toc - tic))
    sys.stdout.flush()
    net.update_lr(i)

    # ---- dev
    if i % nb_its_dev == 0:
        net.set_mode_train(False)
        for x, y in testloader:
            cost, err =  net.eval(x, y)
            cost_dev[i] += cost / float(nb_samples_dev) * float(len(x))
            err_dev[i] += err / float(nb_samples_dev)
        cprint('g','    Jdev = %f, err = %f\n' % (cost_dev[i], err_dev[i]))
        if cost_dev[i] < best_cost:
            best_cost = cost_dev[i]
            net.save('model/theta_best')

toc0 = time.time()
runtime_per_it =  (toc0 - tic0)/float(nb_epochs)
cprint('r','   average time: %f seconds\n' % runtime_per_it)

## ---------------------------------------------------------------------------------------------------------------------
# save model
net.save('model/theta_last')

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
