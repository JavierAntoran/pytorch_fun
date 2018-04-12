from __future__ import print_function
from __future__ import division
import torch, time
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

from src.utils import *

# ------------------------------------------------------------------------------------------------------
# train config

batch_size = 128
nb_epochs = 10
log_interval = 10

# ------------------------------------------------------------------------------------------------------
# dataset
cprint('c','\nData:')

use_cuda = torch.cuda.is_available()

trainset = datasets.MNIST('../data', train=True, download=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=use_cuda, num_workers=1)
     
testset = datasets.MNIST('../data', train=False, transform=transforms.ToTensor())                                            
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, pin_memory=use_cuda, num_workers=1)

input_dim = 784

## ---------------------------------------------------------------------------------------------------------------------
# net dims
cprint('c','\nNetwork:')

from net import Net
net = Net(input_dim)

try:
     epoch = net.load('model/theta_last.dat')
except:
     epoch = 0
     print('failed: starting epoch 0')


## ---------------------------------------------------------------------------------------------------------------------
# train
cprint('c','\nTrain:')

try:
    cost_train, cost_dev, best_cost = load_obj('model/cost.dat')

except:
    print('  init cost variables:')
    cost_train = np.zeros(nb_epochs)
    cost_dev = np.zeros(nb_epochs)
    best_cost = np.inf

nb_its_dev = 1

tic0 = time.time()
for i in range(epoch, nb_epochs):
    net.set_mode_train(True)
    
    tic = time.time()   
    nb_samples = 0
    for x, y in trainloader:
        cost = net.fit(x)

        cost_train[i] += cost * len(x)
        nb_samples += len(x) 
        
    cost_train[i] /= nb_samples
    toc = time.time()

    # ---- print
    print("it %d/%d, Jtr = %f, " % (i, nb_epochs, cost_train[i]), end="")
    cprint('r','   time: %f seconds\n' % (toc - tic))
    net.update_lr(i)

    # ---- dev
    if i % nb_its_dev == 0:
        net.set_mode_train(False)
        nb_samples = 0
        for j, (x, y) in enumerate(testloader):
            cost =  net.eval(x)

            cost_dev[i] += cost * len(x)
            nb_samples += len(x)
            
            # --- reconstruct           
            if j == 0:
                o = net.predict(x).cpu()
                save_image( torch.cat([x[:8], o[:8]]),'results/rec_%d.png' % i, nrow=8 )
                view_image( make_grid(torch.cat([x[:3], o[:3]]), nrow=3).numpy() )
            
        cost_dev[i] /= nb_samples
     
        cprint('g','    Jdev = %f (%f)\n' % (cost_dev[i], best_cost))
        if cost_dev[i] < best_cost:
            best_cost = cost_dev[i]
            net.save('model/theta_best.dat')

    net.save('model/theta_last.dat')
    save_obj([cost_train, cost_dev, best_cost], 'model/cost.dat')
    
    # --- sample
    z = torch.randn(64, 20)
    o = net.decode(z).cpu()
    save_image(o, 'results/sample_%d.png' % i)
    view_image( make_grid( o[:3] ).numpy() )
            

toc0 = time.time()
runtime_per_it =  (toc0 - tic0)/float(nb_epochs)
cprint('r','   average time: %f seconds\n' % runtime_per_it)


## ---------------------------------------------------------------------------------------------------------------------
# results
cprint('c','\nRESULTS:')
nb_parameters = net.get_nb_parameters()
best_cost_dev = np.min( cost_dev )
best_cost_train = np.min( cost_train )

print('  best_cost_dev: %f' %  best_cost_dev)
print('  best_cost_train: %f' %  best_cost_train)
print('  nb_parameters: %d (%s)\n' % (nb_parameters, humansize(nb_parameters)))

with open('results/results.txt','w') as f:
    f.write('%f %f %d %s %f\n' % (best_cost_dev, best_cost_train, nb_parameters, humansize(nb_parameters), runtime_per_it))

## ---------------------------------------------------------------------------------------------------------------------
# fig cost vs its
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.figure()
plt.plot(cost_train, 'r')
plt.plot(cost_dev[::nb_its_dev], 'b')
plt.ylabel('J')
plt.xlabel('it')
plt.grid(True)
plt.savefig('results/train_cost.png')
