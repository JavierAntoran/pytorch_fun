from __future__ import print_function
from __future__ import division
import torch
import torch.backends.cudnn as cudnn

from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from src.layers_pytorch import *
from src.my_loss import Nloss_GD
from src.utils import *

torch.manual_seed(1)
torch.cuda.manual_seed(1)

nlgd = Nloss_GD(dim=20)

use_cuda = torch.cuda.is_available()

class VAE(nn.Module):
    def __init__(self, input_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(1,   64, kernel_size=4, padding=1, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, padding=1, stride=2)
        self.fc3 = nn.Linear(7 * 7 * 128, 1024)
        self.fc41 = nn.Linear(1024, 20)  # mu
        self.fc42 = nn.Linear(1024, 20)  # log(psi)
        
        self.fc5 = nn.Linear(20, 1024)
        self.fc6 = nn.Linear(1024, 7 * 7 * 128)
        self.conv7 = nn.ConvTranspose2d(128, 64, kernel_size=4, padding=1, stride=2)
        self.conv8 = nn.ConvTranspose2d(64,   1, kernel_size=4, padding=1, stride=2)

    def encode(self, x):
        x = self.conv1(x)
        x = lrelu(x)
        x = self.conv2(x)
        x = lrelu(x)
        x = x.view(-1, 7 * 7 * 128)
        x = self.fc3(x)
        x = lrelu(x)
        h = x
        return self.fc41(h) #, self.fc42(h)

    # def reparameterize(self, mu, logpsi):
    #     if self.training:
    #         std = logpsi.mul(0.5).exp_()
    #         eps = Variable(std.data.new(std.size()).normal_())
    #         return eps.mul(std).add_(mu)
    #     else:
    #         return mu

    def decode(self, z):
        x = self.fc5(z)
        x = lrelu(x)
        x = self.fc6(x)
        x = lrelu(x)
        x = x.view(-1, 128, 7, 7)
        x = self.conv7(x)
        x = lrelu(x)
        h = x
        x = self.conv8(x)
        return sigmoid(x), h

    def forward(self, x):
        z = self.encode(x)
        # z = self.reparameterize(mu, logpsi)
        y, h = self.decode(z)
        return y, z, h


def BCE_KLD_cost(x, z, mu, sq_beta, y):
    BCE = F.binary_cross_entropy(x, y, size_average=False)
    Nloss = 100 * nlgd(z, mu, sq_beta)
    #KLD = -0.5 * torch.sum(1 + logpsi - mu.pow(2) - logpsi.exp())   # https://arxiv.org/abs/1312.6114 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    return BCE + Nloss #+ KLD


class Net(BaseNet):
    def __init__(self,  input_dim, lr=1e-3):
        super(Net, self).__init__()
        cprint('y', '  VAE')
        self.input_dim = input_dim
        self.lr = lr
        self.schedule = None #[] #[50,200,400,600]
        self.create_net()
        self.create_opt()
        self.epoch = 0


    def create_net(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        self.model = VAE(self.input_dim)
        if use_cuda:
            self.model.cuda()
            cudnn.benchmark = True
        
        self.J = BCE_KLD_cost

        print('    Total params: %.2fM' % ( self.get_nb_parameters() / 1000000.0) )

    def create_opt(self):
        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        #self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=5e-2)
        #self.optimizer = SGDW(self.model.parameters(), lr=self.lr, weight_decay=5e-3, momentum=0.9, nesterov=True)

    def fit(self, x):

        if use_cuda:
            mu = Variable(torch.zeros(x.shape[0], 20).cuda(), requires_grad=False)
            sq_betas = Variable(torch.ones(x.shape[0], 20).cuda(), requires_grad=False)
            noise = Variable(torch.randn(x.shape[0], 20).cuda(), requires_grad=False)
        else:
            mu = Variable(torch.zeros(x.shape[0], 20), requires_grad=False)
            sq_betas = Variable(torch.ones(x.shape[0], 20), requires_grad=False)
            noise = Variable(torch.randn(x.shape[0], 20), requires_grad=False)

        x, = to_variable(var=(x,), cuda=use_cuda)
        
        self.optimizer.zero_grad()
        out, z, _ = self.model(x)

        z_noise = z + noise
        loss = self.J(out, z, mu, sq_betas, x)

        loss.backward()
        self.optimizer.step()

        return loss.data[0] / len(x)


    def eval(self, x, train=False):
        x, = to_variable(var=(x,), cuda=use_cuda, volatile=True)

        if use_cuda:
            mu = Variable(torch.zeros(x.shape[0], 20).cuda(), requires_grad=False)
            sq_betas = Variable(torch.ones(x.shape[0], 20).cuda(), requires_grad=False)
        else:
            mu = Variable(torch.zeros(x.shape[0], 20), requires_grad=False)
            sq_betas = Variable(torch.ones(x.shape[0], 20), requires_grad=False)

        out, z, _ = self.model(x)
        loss = self.J(out, z, mu, sq_betas, x)
        return loss.data[0] / len(x)

    def predict(self, x, train=False):
        x, = to_variable(var=(x,), cuda=use_cuda, volatile=True)
        out, _, _ = self.model(x)
        return out.data

    def extract(self, x, train=False):
        x, = to_variable(var=(x,), cuda=use_cuda, volatile=True)
        _, _, out = self.model(x)
        return out.data

    def decode(self, z, train=False):
        z, = to_variable(var=(z,), cuda=use_cuda, volatile=True)
        out, _ = self.model.decode(z)
        return out.data

if __name__ == '__main__':
    # check errors and size
    x = Variable(torch.randn(32, 1, 28, 28))
    
    model = VAE((1, 28, 28))
    y, z, h = model(x)
    print(y)
    print(y.size())
    print(z.size())
    print(h.size())

    net = Net((1, 28, 28))
    c = net.fit(x)
    print(c)

    c = net.eval(x)
    print(c)
    
    c = net.predict(x)
    print(c.size())
    
    c = net.extract(x)
    print(c.size())
