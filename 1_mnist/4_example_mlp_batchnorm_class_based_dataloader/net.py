from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.backends.cudnn as cudnn

from src.utils import *

class Net_pytorch(nn.Module):
    def __init__(self):
        super(Net_pytorch, self).__init__()
        self.conv1 = nn.Conv2d(1,  16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)

        self.fc1 = nn.Linear(512, 500)
        self.bn1 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 10)

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
        x = self.bn1(x)
        x = F.relu(x)
        # -----
        x = self.fc2(x)
        x = F.log_softmax(x)
        return x

class Net:
    def __init__(self, input_dim, output_dim, batch_size=None):
        cprint('c', '\nNet:')
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.create_net()

    def create_net(self):
        self.model = Net_pytorch()
        self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        #self.J = nn.CrossEntropyLoss()
        cudnn.benchmark = True


    def set_mode_train(self, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()

    def fit(self, x, y):
        # x, y = torch.from_numpy(x_train), torch.from_numpy(y_train)
        x, y = x.cuda(), y.cuda(async=True)
        x, y = Variable(x), Variable(y)


        out = self.model(x)
        loss =  F.nll_loss(out, y) #self.J(out, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data[0]

    def eval(self, x_dev, y_dev, train=False):
        x, y = torch.from_numpy(x_dev), torch.from_numpy(y_dev)
        x, y = x.cuda(), y.cuda()
        x, y = Variable(x), Variable(y)

        out = self.model(x)
        loss = F.nll_loss(out, y) #self.J(out, y)

        pred = out.data.max(1)[1] # get the index of the max log-probability
        err = pred.ne(y.data).cpu().sum()

        return loss.data[0], err

    def predict(self, x_test, train=False):
        x = torch.from_numpy(x_test)
        x = x.cuda()
        x = Variable(x)

        out = self.model(x)
        pred = out.data
        return pred

    # def extract(self, x_test, train=False):
    #     data = {self.X: x_test, self.train: train}
    #     return self.session.run(self.Z, feed_dict=data)

    def save(self, filename):
        cprint('c', 'Writting %s\n' % filename)
        torch.save(self.model, filename) # complete object



    def load(self, filename):
        cprint('c', 'Reading %s\n' % filename)
        self.model = torch.load(filename) # complete object
