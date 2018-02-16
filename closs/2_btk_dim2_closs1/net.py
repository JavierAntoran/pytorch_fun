from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

from src.utils import *
from src.layers_pytorch import *

class Net_pytorch(nn.Module):
    def __init__(self):
        super(Net_pytorch, self).__init__()
        self.conv1 = nn.Conv2d(  1,  16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d( 16,  32, kernel_size=5, padding=2)        
        self.bn1 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32*7*7, 2)
        self.fc2 = nn.Linear(2, 10)        

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        # -----
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        # -----
        x = self.bn1(x)
        x = x.view(-1, 32*7*7)
        # -----
        x = self.fc1(x)
        e = x
        # -----
        x = self.fc2(x)
        return x, e

class Net:
    def __init__(self, input_dim, output_dim, lr=1e-4):
        cprint('c', '\nNet:')
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.epoch = 0
        self.create_net()
        self.create_opt(lr)

    def create_net(self):
        self.model = Net_pytorch()
        self.model.cuda()
        self.J = nn.CrossEntropyLoss()
        self.C = CenterLoss(self.output_dim, 2, loss_weight=1.0, alpha=0.01).cuda()
        print('    Total params: %.2fM' % (sum(p.numel() for p in self.model.parameters())/1000000.0))

    def create_opt(self, lr):        
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.schedule = [-1] #[50,200,400,600]

    def get_nb_parameters(self):
        return np.sum(p.numel() for p in self.model.parameters())

    def update_lr(self, epoch, gamma=0.99):
        if len(self.schedule) == 0 or epoch in self.schedule:
            self.lr *= gamma
            print('learning rate: %f\n' % self.lr)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
        self.epoch += 1

    def set_mode_train(self, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()

    
    def fit(self, x, y):
        x, y = to_variable_cuda(x, y)
        
        self.optimizer.zero_grad()
        out, e = self.model(x)
        
        loss = self.J(out, y) + self.C(e, y)        
        loss.backward()
        self.optimizer.step()

        pred = out.data.max(1)[1]
        err = pred.ne(y.data).cpu().sum()

        return loss.data[0], err

    def eval(self, x, y):
        x, y = to_variable_cuda(x, y)
        
        out, _ = self.model(x)
        loss = self.J(out, y)

        pred = out.data.max(1)[1]
        err = pred.ne(y.data).cpu().sum()

        return loss.data[0], err

    def predict(self, x):
        x, = to_variable_cuda(x,)
        out, _ = self.model(x)
        return out.data 

    def extract(self, x):
        x, = to_variable_cuda(x,)
        _, e = self.model(x)
        return e.data



    def save(self, filename):
        cprint('c', 'Writting %s\n' % filename)
        torch.save({    
            'epoch': self.epoch,
            'lr': self.lr,
            'model': self.model,
            'optimizer': self.optimizer}, filename)

    def load(self, filename):
        cprint('c', 'Reading %s\n' % filename)
        state_dict = torch.load(filename)
        self.epoch = state_dict['epoch']
        self.lr = state_dict['lr'] 
        self.model = state_dict['model']
        self.optimizer = state_dict['optimizer']
        print('  restoring epoch: %d, lr: %f' % (self.epoch, self.lr))
