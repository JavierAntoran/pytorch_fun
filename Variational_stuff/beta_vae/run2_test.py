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

testset = datasets.MNIST('../data', train=False, transform=transforms.ToTensor())                                            
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1)

input_dim = 784

## ---------------------------------------------------------------------------------------------------------------------
# net dims
cprint('c','\nNetwork:')

from net import Net
net = Net(input_dim)

net.load('model/theta_last.dat')


## ---------------------------------------------------------------------------------------------------------------------
# train
cprint('c','\nTest:')


net.set_mode_train(False)
nb_samples = 0
cost_test = 0
for j, (x, y) in enumerate(testloader):
    cost =  net.eval(x)

    cost_test += cost * len(x)
    nb_samples += len(x)

    # --- reconstruct  
    if j == 0:
        o = net.predict(x).cpu()
        save_image( torch.cat([x[:8], o[:8]]),'results/rec_test.png', nrow=8 )
        view_image( make_grid(torch.cat([x[:3], o[:3]]), nrow=3).numpy() )

cost_test /= nb_samples

cprint('g','    Jtest = %f\n' % cost_test)


# --- sample
z = torch.randn(64, 20)
o = net.decode(z).cpu()
save_image(o, 'results/sample_test.png' )
view_image( make_grid( o[:3] ).numpy() )
