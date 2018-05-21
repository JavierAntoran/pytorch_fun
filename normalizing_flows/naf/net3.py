from __future__ import print_function
from __future__ import division

import torch
import src.flows as flows

from src.layers_pytorch import *
from src.utils import *

class Net(object):

    def __init__(self, input_dim, cuda=False):
        self.cuda = cuda
        self.model = flows.IAF_DSF(input_dim, 128, 1, 3, num_ds_dim=3, num_ds_layers=3)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.const = -0.5 * input_dim * np.log(2 * np.pi)

    def log_p(self, x, lgd=None, context=None):
        x, = to_variable(var=(x,), cuda=self.cuda)

        context = Variable(torch.FloatTensor(len(x), 1).zero_()) + 2.0
        lgd = Variable(torch.FloatTensor(len(x)).zero_())

        z, logdet, _ = self.model((x, lgd, context))
        log_pz = -0.5 *(z ** 2).sum(1) + self.const + logdet

        return log_pz

    def fit(self, x):
        x, = to_variable(var=(x,), cuda=self.cuda)

        self.optim.zero_grad()
        loss = -self.log_p(x).mean()
        loss.backward()
        self.optim.step()

        return loss.data[0]

    def train(self, x, batch_size=64, nb_its=200):
        for i in range(nb_its):
            loss = 0
            for ind in generate_ind_batch(len(x), batch_size, roundup=False):
                loss += self.fit(x[ind]) * len(ind) / len(x)

            cprint('y','%d/%d %f' % (i, nb_its,loss))


