import torch
import torch.nn.functional as F
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
import numpy as np

from src.layers import *

class BatchNorm2d(torch.nn.Module):
    def __init__(self, ch, eps=1e-5, momentum=0.9):
        super(BatchNorm2d, self).__init__()
        self.ch = ch
        self.eps = eps
        self.momentum = momentum

        self.a = Parameter(torch.ones(ch)) #torch.rand(ch))
        self.b = Parameter(torch.zeros(ch))

        self.register_buffer('mean', torch.zeros(ch))
        self.register_buffer('var',  torch.ones(ch))

    def forward(self, input):
        N, C, H, W = input.size()
        if self.training:
            sum_x0 = N * H * W
            sum_x1 = reduce_sum(input, axis=[0, 2, 3] )
            sum_x2 = reduce_sum(input.pow(2), axis=[0, 2, 3])
            mean = sum_x1.div(sum_x0)
            var = sum_x2.sub_(sum_x1.pow(2).div(sum_x0)).div(sum_x0)
            self.mean = self.momentum * self.mean + ( 1 - self.momentum ) * mean.data
            self.var = self.momentum * self.var + ( 1 - self.momentum ) * var.data

        x = input.view(N, C, -1).contiguous()
        # print('m0',self.mean)
        x = F.batch_norm(x, self.mean, self.var, self.a, self.b, eps=self.eps)
        # print('m1',self.mean)
        x = x.view(N, C, H, W)
        return x


class BatchNorm2dFull(torch.nn.Module):
    def __init__(self, ch, eps=1e-5, momentum=0.9):
        super(BatchNorm2dFull, self).__init__()
        self.ch = ch
        self.eps = eps
        self.momentum = momentum

        self.a = Parameter(torch.ones(ch)) #torch.rand(ch))
        self.b = Parameter(torch.zeros(ch))

        self.register_buffer('mean', torch.zeros(ch))
        self.register_buffer('var',  torch.diag(torch.ones(ch)))
        # self.register_buffer('sum_x1', torch.zeros(ch))
        # self.register_buffer('sum_x2', torch.zeros(ch,ch))

        V = 1e-3 * torch.randn(ch, ch)
        Q, R = torch.qr(V)
        self.i = 0
        self.register_buffer('V', V)
        self.register_buffer('Q', Q)
        self.register_buffer('R', R)

        self.register_buffer('s', torch.ones(ch))
        self.register_buffer('Vnt', torch.diag(torch.ones(ch)))

    def forward(self, input):
        N, C, H, W = input.size()

        x = input.permute(1, 0, 2, 3).contiguous()
        x = x.view(C, -1)

        if self.training:
            sum_x0 = N * H * W
            sum_x1 = x.sum(1)
            sum_x2 = torch.mm(x, x.t())

            mean = sum_x1.div(sum_x0)
            sum_x1 = sum_x1.view(C, -1)
            var = sum_x2.sub_( sum_x1.mm(sum_x1.t()).div(sum_x0)).div(sum_x0)

            self.mean = self.momentum * self.mean + ( 1 - self.momentum ) * mean.data
            self.var = self.momentum * self.var + ( 1 - self.momentum ) * var.data

            # self.sum_x1 = self.momentum * self.sum_x1 + ( 1 - self.momentum ) * sum_x1.data
            # self.sum_x2 = self.momentum * self.sum_x2 + ( 1 - self.momentum ) * sum_x2.data
            # mean = self.sum_x1.div(sum_x0)
            # sum_x1 = self.sum_x1.view(C, -1)
            # var = self.sum_x2.sub( sum_x1.mm(sum_x1.t()).div(sum_x0)).div(sum_x0)
            # self.mean = self.momentum * self.mean + (1 - self.momentum) * mean
            # self.var = self.momentum * self.var + ( 1 - self.momentum ) * var


            # --- iteration of block iteration
            self.V = torch.mm(self.var, self.Q)
            self.Q, self.R = torch.qr(self.V)            # cost 2 n r^2 - 2r^3 / 3

            self.i += 1
            # if self.i == 1000:
            #      print('********************************')

            if self.i > 200:
                if self.i % 50 == 0:
                    print('**********')
                    s = torch.norm(self.R, 2, 1)
                    Vn = self.Q * (1 / torch.sqrt(s))
                    Vnt = Vn.t()

                    self.s = 0.9 * self.s + 0.1 * s
                    self.Vnt = 0.9 * self.Vnt + 0.1 * Vnt

        mean = Variable(self.mean, requires_grad=False)
        Vnt = Variable(self.Vnt, requires_grad=False)

        x = x.sub( mean.view(C, -1) )
        x = Vnt.mm( x )

        x = x * self.a.view(C, -1) + self.b.view(C, -1)

        x = x.view(C, N, H, W)
        x = x.permute(1, 0, 2, 3)
        return x


class BatchNorm2dFullr(torch.nn.Module):
    def __init__(self, ch, cho, eps=1e-5, momentum=0.9):
        super(BatchNorm2dFullr, self).__init__()
        self.ch = ch
        self.cho = cho
        self.eps = eps
        self.momentum = momentum

        self.a = Parameter(torch.ones(cho)) #torch.rand(ch))
        self.b = Parameter(torch.zeros(cho))

        self.register_buffer('mean', torch.zeros(ch))
        self.register_buffer('var',  torch.diag(torch.ones(ch)))

        V = 1e-3 * torch.randn(ch, cho)
        Q, R = torch.qr(V)
        self.i = 0
        self.register_buffer('V', V)
        self.register_buffer('Q', Q)
        self.register_buffer('R', R)

        self.register_buffer('s', torch.ones(cho))
        self.register_buffer('Vnt', torch.eye(cho, ch) + 1e-3 * torch.randn(cho, ch))

    def forward(self, input):
        N, C, H, W = input.size()

        x = input.permute(1, 0, 2, 3).contiguous()
        x = x.view(C, -1)

        if self.training:
            sum_x0 = N * H * W
            sum_x1 = x.sum(1)
            sum_x2 = torch.mm(x, x.t())

            mean = sum_x1.div(sum_x0)
            sum_x1 = sum_x1.view(C, -1)
            var = sum_x2.sub_( sum_x1.mm(sum_x1.t()).div(sum_x0)).div(sum_x0)

            # raw_input('pause')
            self.mean = self.momentum * self.mean + ( 1 - self.momentum ) * mean.data
            self.var = self.momentum * self.var + ( 1 - self.momentum ) * var.data

            # --- iteration of block iteration
            self.V = torch.mm(self.var, self.Q)
            self.Q, self.R = torch.qr(self.V)  # cost 2 n r^2 - 2r^3 / 3

            self.i += 1
            # if self.i == 1000:
            #      print('********************************')

            if self.i > 200:
                if self.i % 50 == 0:
                    print('**********')
                    s = torch.norm(self.R, 2, 1)
                    Vn = self.Q * (1 / torch.sqrt(s))
                    Vnt = Vn.t()

                    self.s = 0.9 * self.s + 0.1 * s
                    self.Vnt = 0.9 * self.Vnt + 0.1 * Vnt

        mean = Variable(self.mean, requires_grad=False)
        Vnt = Variable(self.Vnt, requires_grad=False)

        x = x.sub(mean.view(C, -1))
        x = Vnt.mm(x)

        x = x * self.a.view(self.cho, -1) + self.b.view(self.cho, -1)

        x = x.view(self.cho, N, H, W)
        x = x.permute(1, 0, 2, 3)
        return x



if __name__ == '__main__':
    # check errors

    # -------------------------------------------------------------------------------------------------
    # nn = BatchNorm2d(2)
    # nn.train(False)
    # y = nn(Variable(torch.arange(0,12).view(1, 2, 2, 3).repeat(2, 1, 1, 1)))
    # print(y)

    # -------------------------------------------------------------------------------------------------
    # nn = BatchNorm2d(2)
    # y = nn(Variable(torch.arange(0,12).view(1, 2, 2, 3).repeat(2, 1, 1, 1)))
    # print(y)

    # import numpy as np
    # x = np.tile( np.arange(0,12).reshape(1, 2, 2, 3), (2, 1, 1, 1))
    # xx = x.transpose(1, 0, 2, 3).reshape(2,-1)
    # print( np.sum(xx, axis=1) )
    # print( np.sum(xx**2, axis=1) )
    # print( np.sum(xx, axis=1)/xx.shape[1]  )
    # print( (np.sum(xx**2, axis=1) - np.sum(xx, axis=1)**2 / xx.shape[1] )/xx.shape[1] )
    # print( np.var(xx, axis=1))

    # -------------------------------------------------------------------------------------------------
    # nn = BatchNorm2d(2)
    # for _ in range(100):
    #     y = nn(Variable(torch.arange(0, 12).view(1, 2, 2, 3).repeat(2, 1, 1, 1)))
    #     print('m')
    #     print( y.data.numpy().mean(axis=(0, 2, 3)))
    #     print('s')
    #     print( y.data.numpy().std(axis=(0, 2, 3)))
    #
    # print(y)


    # -------------------------------------------------------------------------------------------------
    # nn = BatchNorm2dFull(2)
    # y = nn(Variable(torch.arange(0,12).view(1, 2, 2, 3).repeat(2, 1, 1, 1)))
    # print(y)

    # import numpy as np
    # x = np.tile( np.arange(0,12).reshape(1, 2, 2, 3), (2, 1, 1, 1))
    # xx = x.transpose(1, 0, 2, 3).reshape(2,-1)
    # sum_x2 = np.dot(xx, xx.T)
    # sum_x1 = np.sum(xx, axis=1).reshape(2,-1)
    # print( sum_x1 )
    # print( sum_x2 )
    # print( sum_x1/xx.shape[1]  )
    # print( (sum_x2 - np.dot(sum_x1, sum_x1.T)/ xx.shape[1] )/xx.shape[1] )
    # print( np.cov(xx))

    # s, V = np.linalg.eig(np.cov(xx,bias=True))
    # V, s, Vt = np.linalg.svd(np.cov(xx,bias=True), full_matrices=False)

    # -------------------------------------------------------------------------------------------------
    # np.random.seed(42)
    # X = np.asarray(np.random.randn(2, 2, 2, 3), dtype=np.float32) #X = np.random.randn(9, 4)
    #
    #
    # nn = BatchNorm2dFull(2)
    # for _ in range(1000):
    #     y = nn(Variable(torch.from_numpy(X)))
    #
    #     yy = y.data.numpy()
    #     yy = yy.transpose(1, 0, 2, 3).reshape(2, -1)
    #
    #     print('m')
    #     print( yy.mean(axis=1))
    #     print('s')
    #     print(np.cov(yy))
    #
    # print(y)

    # -------------------------------------------------------------------------------------------------
    np.random.seed(42)
    X = np.asarray(np.random.randn(2, 3, 2, 3), dtype=np.float32) #X = np.random.randn(9, 4)


    nn = BatchNorm2dFullr(3,2)
    for _ in range(1000):
        y = nn(Variable(torch.from_numpy(X)))

        yy = y.data.numpy()
        yy = yy.transpose(1, 0, 2, 3).reshape(2, -1)

        print('m')
        print( yy.mean(axis=1))
        print('s')
        print(np.cov(yy))

    print(y)
