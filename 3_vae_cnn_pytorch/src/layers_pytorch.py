import torch
from torch.autograd import Variable
from torch.nn import functional as F

from src.utils import *

# ----------------------------------------------------------------------------------------------------------------------
def to_variable(var=(), cuda=True, volatile=False):
    out = []
    for v in var:
        if isinstance(v, np.ndarray):        
            v = torch.from_numpy(v)
        
        if not v.is_cuda and cuda:
            v = v.cuda()
          
        if not isinstance(v, Variable):            
            v = Variable(v, volatile=volatile)
            
        out.append(v)
    return out
    
# ----------------------------------------------------------------------------------------------------------------------
def tanh(x):
    return F.tanh(x)

def relu(x):
    return F.relu(x)

def lrelu(x):
    return F.leaky_relu(x)

def sigmoid(x):
    return F.sigmoid(x)


# ----------------------------------------------------------------------------------------------------------------------
class BaseNet(object):
    def __init__(self):
        cprint('c', '\nNet:')

    def get_nb_parameters(self):
        return np.sum(p.numel() for p in self.model.parameters())

    def set_mode_train(self, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()

    def update_lr(self, epoch, gamma=0.99):        
        self.epoch += 1
        if self.schedule is not None:
            if len(self.schedule) == 0 or epoch in self.schedule:
                self.lr *= gamma
                print('learning rate: %f  (%d)\n' % self.lr, epoch)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr

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
        return self.epoch
