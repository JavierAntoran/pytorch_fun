from __future__ import print_function
from __future__ import division

from src.layers_pytorch import *

class Convbn(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Convbn, self).__init__()
        self.conv1 = nn.Conv2d(input_dim[0], 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)

        self.bn1 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32*7*7, 2)
        self.fc2 = nn.Linear(2, 10)        

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = relu(x)
        # -----
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = relu(x)
        # -----
        x = self.bn1(x)
        x = x.view(-1, 32*7*7)
        # -----
        x = self.fc1(x)
        e = x
        # -----
        x = self.fc2(x)
        return x, e



class Net(BaseNet):
    def __init__(self, input_dim, output_dim, lr=1e-4, cuda=True):
        super(Net, self).__init__()
        cprint('c', '\nNet:')
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.cuda = cuda
        self.create_net()
        self.create_opt(lr)

    def create_net(self):
        self.model = Convbn(self.input_dim, self.output_dim)
        self.J = nn.CrossEntropyLoss()
        self.C = CenterLoss2(self.output_dim, 2, loss_weight=1.0, alpha=0.01).cuda()
        if self.cuda:
            self.model.cuda()
            self.J.cuda()
            self.C.cuda()

        print('    Total params: %.2fM' % (sum(p.numel() for p in self.model.parameters()) / 1000000.0))

    def create_opt(self, lr=1e-4):
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.schedule = None  # [-1] #[50,200,400,600]

    def fit(self, x, y):
        x, y = to_variable(var=(x, y), cuda=self.cuda)

        self.optimizer.zero_grad()
        out, e = self.model(x)

        loss = self.J(out, y) + self.C(e, y)
        loss.backward()
        self.optimizer.step()

        pred = out.data.max(1)[1]
        err = pred.ne(y.data).cpu().sum()

        return loss.data[0], err

    def eval(self, x, y, train=False):
        x, y = to_variable(var=(x, y), volatile=True, cuda=self.cuda)

        out, _ = self.model(x)
        loss = self.J(out, y)

        pred = out.data.max(1)[1]
        err = pred.ne(y.data).cpu().sum()

        return loss.data[0], err

    def predict(self, x, train=False):
        x, = to_variable(var=(x,), volatile=True, cuda=self.cuda)

        out, _ = self.model(x)
        return out.data

    def extract(self, x, train=False):
        x, = to_variable(var=(x,), volatile=True, cuda=self.cuda)

        _, e = self.model(x)
        return e.data
