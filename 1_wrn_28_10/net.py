from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch
from src.utils import *
import math

class wrn_layer(nn.Module):
    def __init__(self, ch_in, ch_out, p_drop=0., stride=1):
        super(wrn_layer, self).__init__()
        self.p_drop = p_drop
        self.conv_shortcut = ch_in != ch_out # conv shortcut (only to adjust nb_ch)
        # -----------------------
        self.bn1 = nn.BatchNorm2d(ch_in)
        if self.p_drop > 0:
            self.drop1 = nn.Dropout(p=p_drop)
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1, bias=False)
        # -----------------------
        self.bn2 = nn.BatchNorm2d(ch_out)
        if self.p_drop > 0:
            self.drop2= nn.Dropout(p=p_drop)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False)
        # -----------------------
        if self.conv_shortcut:
            self.conv3 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride, padding=0, bias=False)

    def forward(self, x):
        x0 = x
        # -----------------------
        x = self.bn1(x)
        x = F.relu(x)
        if self.p_drop > 0:
            x = self.drop1(x)
        x = self.conv1(x)
        # -----------------------
        x = self.bn2(x)
        x = F.relu(x)
        if self.p_drop > 0:
            x = self.drop2(x)
        x = self.conv2(x)
        # -----------------------
        if self.conv_shortcut:
            x += self.conv3(x0)
        else:
            x += x0
        return x

def wrn_block(ch_in, ch_out, nb_blocks, p_drop=0., stride=1):
        layers = []
        for _ in range(nb_blocks):
            layers.append(wrn_layer(ch_in, ch_out, p_drop, stride))
            ch_in = ch_out
            stride = 1
        return nn.Sequential(*layers)

class WideResNet(nn.Module):
    def __init__(self, nb_classes, depth=28, widen_factor=10, p_drop=0.):
        super(WideResNet, self).__init__()
        assert ((depth - 4) % 6 == 0), 'depth != 6n+4'
        n = (depth-4)/6
        nb_ch = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        self.conv1 = nn.Conv2d(3, nb_ch[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = wrn_block(nb_ch[0], nb_ch[1], n, p_drop, stride=1)
        self.block2 = wrn_block(nb_ch[1], nb_ch[2], n, p_drop, stride=2)
        self.block3 = wrn_block(nb_ch[2], nb_ch[3], n, p_drop, stride=2)
        self.bn1 = nn.BatchNorm2d(nb_ch[3]) #, momentum=0.9)
        self.fc2 = nn.Linear(nb_ch[3], nb_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 8)
        x = x.view(x.size(0), -1)
        x = self.fc2(x)
        return x

class Net:
    def __init__(self, input_dim, output_dim, batch_size=None, lr=0.1, p_drop=0., single_gpu=False):
        cprint('c', '\nNet:')
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.lr = 0.1
        self.p_drop = p_drop
        self.single_gpu = single_gpu
        self.schedule = [150, 225]
        self.create_net()

    def create_net(self):
        torch.manual_seed(42)
        self.model = WideResNet(self.output_dim, p_drop=self.p_drop)
        if self.single_gpu:
            self.model.cuda()
        else:
            self.model = torch.nn.DataParallel(self.model).cuda()
        cudnn.benchmark = True

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        self.J = nn.CrossEntropyLoss()
        print('    Total params: %.2fM' % (sum(p.numel() for p in self.model.parameters())/1000000.0))

    def get_nb_parameters(self):
        return np.sum(p.numel() for p in self.model.parameters())

    def set_mode_train(self, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()

    def update_lr(self, epoch, gamma=0.1):
        if epoch in self.schedule:
            self.lr *= gamma
            print('learning rate: %f\n' % self.lr)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr

    def fit(self, x_train, y_train):
        x, y = x_train, y_train
        x, y = x.cuda(), y.cuda()
        x, y = Variable(x), Variable(y)

        self.optimizer.zero_grad()
        out = self.model(x)
        loss =  self.J(out, y)
        loss.backward()
        self.optimizer.step()

        return loss.data[0]

    def eval(self, x_dev, y_dev, train=False):
        x, y = x_dev, y_dev
        x, y = x.cuda(), y.cuda()
        x, y = Variable(x), Variable(y)

        out = self.model(x)
        loss = self.J(out, y)

        pred = out.data.max(1)[1]
        err = pred.ne(y.data).cpu().sum()

        return loss.data[0], err

    def predict(self, x_test, train=False):
        x = torch.from_numpy(x_test)
        x = x.cuda()
        x = Variable(x)
        out = self.model(x)
        pred = out.data
        return pred

    def save(self, filename):
        cprint('c', 'Writting %s\n' % filename)
        torch.save(self.model, filename) # complete object

    def load(self, filename):
        cprint('c', 'Reading %s\n' % filename)
        self.model = torch.load(filename) # complete object

if __name__ == '__main__':
    # check errors and size
    nn = WideResNet(10, 28, 10, 0.0)
    y = nn(Variable(torch.randn(1, 3, 32, 32)))

    print(y.size())
