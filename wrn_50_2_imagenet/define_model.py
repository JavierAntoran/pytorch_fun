import re
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
#from torchviz import make_dot
from torch.utils import model_zoo


class my_wrn_transfer(nn.Module):

    def __init__(self, params, N_out):
        super(my_wrn_transfer, self).__init__()
        self.params = params
        self.N_out = N_out
        # convert numpy arrays to torch Variables
        for k, v in sorted(self.params.items()):
            #print(k, tuple(v.shape))
            self.params[k] = nn.Parameter(v, requires_grad=False)

        # determine network size by parameters
        self.blocks = [sum([re.match('group%d.block\d+.conv0.weight' % j, k) is not None
                       for k in params.keys()]) for j in range(4)]

        print('\nTotal WResnet parameters:', sum(v.numel() for v in self.params.values()))

        self.pretrained_f = self.f
        self.fc1 = nn.Linear(2048, self.N_out)

    def conv2d(self, input, params, base, stride=1, pad=0):
        return F.conv2d(input, params[base + '.weight'],
                        params[base + '.bias'], stride, pad)

    def group(self, input, params, base, stride, n):
        o = input
        for i in range(0, n):
            b_base = ('%s.block%d.conv') % (base, i)
            x = o
            o = self.conv2d(x, params, b_base + '0')
            o = F.relu(o)
            o = self.conv2d(o, params, b_base + '1', stride=i == 0 and stride or 1, pad=1)
            o = F.relu(o)
            o = self.conv2d(o, params, b_base + '2')
            if i == 0:
                o += self.conv2d(x, params, b_base + '_dim', stride=stride)
            else:
                o += x
            o = F.relu(o)
        return o

    def f(self, input, params):
        o = F.conv2d(input, params['conv0.weight'], params['conv0.bias'], 2, 3)
        o = F.relu(o)
        o = F.max_pool2d(o, 3, 2, 1)
        o_g0 = self.group(o, params, 'group0', 1, self.blocks[0])
        o_g1 = self.group(o_g0, params, 'group1', 2, self.blocks[1])
        o_g2 = self.group(o_g1, params, 'group2', 2, self.blocks[2])
        o_g3 = self.group(o_g2, params, 'group3', 2, self.blocks[3])
        o = F.avg_pool2d(o_g3, 7, 1, 0)
        o = o.view(o.size(0), -1)
        # Disabled last linear layer for trasfer learning
        # o = F.linear(o, params['fc.weight'], params['fc.bias'])
        return o

    def forward(self, x):

        # input shape x (batch_size, 3, 224,224)
        x1 = self.pretrained_f(x, self.params)
        x1 = self.fc1(x1)
        return x1




