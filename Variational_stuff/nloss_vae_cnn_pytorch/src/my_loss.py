import torch
import torch.nn as nn

import numpy as np


class Nloss_GD(nn.Module):
    def __init__(self, dim):
        super(Nloss_GD, self).__init__()
        self.dim = dim

        torch.manual_seed(0)

    def get_log_likelihoods(self, X, Y, sq_Beta, eps=1e-6):
        # Returns likelihoods of each datapoint for every cluster
        Beta = sq_Beta ** 2
        # batch_size
        log_det_term = 0.5 * (Beta.log().sum(dim=1))
        # print('detterm shape:', log_det_term.shape)
        # 1
        norm_term = -0.5 * np.log(2 * np.pi) * self.dim
        # print('normterm shape:', norm_term.shape)
        # batch_size, dims
        inv_covars = Beta
        # batch_size, dims
        dist = (Y - X).pow(2)
        # batch_size
        exponent = (-0.5 * dist * inv_covars).sum(dim=1)
        # print('exponent shape:', exponent.shape)
        # batch_size
        log_p = (log_det_term + exponent) + norm_term
        # print('log_p shape:', log_p.shape)
        return log_p

    def forward(self, x, y, Beta):
        # Returns -loglike of all data
        # batch_size
        # print(Beta.mean())
        p = self.get_log_likelihoods(x, y, Beta)
        # 1
        E = torch.sum(-p) / x.shape[0]
        return E