import torch.nn as nn
import torch.nn.functional as F
import torch

import math
from torch.optim.optimizer import Optimizer, required

def relu(x):
    return F.relu(x)

def avg_pool2d(x, kernel_size, stride=2, padding='same'):
    if padding == 'same':
        padding = kernel_size // 2
    return F.avg_pool2d(x, kernel_size, stride=stride, padding=padding)

def max_pool2d(x, kernel_size, stride=2, padding='same'):
    if padding == 'same':
        padding = kernel_size // 2
    return F.max_pool2d(x, kernel_size, stride=stride, padding=padding)

def concat(x, dim=0):
    return torch.cat(x, dim)

def concat_(x):
    y = []
    for xi in x:
        n, c, h, w = xi.size()
        yi = xi.view(n, c, h, w, 1)
        y.append(yi)
    return torch.cat(y, 4)

def pad2d(x, pad=(0, 0, 0, 0)):
    x = nn.ConstantPad2d(pad, 0)(x)
    return x

def pad11(x, pad=(0, 1, 0, 1)):
    x = nn.ConstantPad2d((0, 1, 0, 1),0)(x)
    return x[:,:,1:,1:]

def reduce_mean(x, axis=0):
    if isinstance(axis, list):
        aux = x
        for a in reversed(axis):
            aux = aux.mean(a)
        return aux
    else:
        return x.mean(axis)
            

def reduce_sum(x, axis=0):
    if isinstance(axis, list):
        aux = x
        for a in reversed(axis):
            aux = aux.sum(a)
        return aux
    else:
        return x.sum(axis)

def reduce_max(x, axis=0):
    if isinstance(axis, list):
        aux = x
        for a in reversed(axis):
            aux = aux.max(a)
        return aux
    else:
        return x.max(axis)

def dropout(x, keep_prob=0.5):
    return nn.Dropout(1.-keep_prob)(x)

def ch_pool(x):
    x00 = avg_pool2d(pad2d(x, (0, 0, 0, 0))[:,:,:-1,:-1], 1, stride=2, padding=0)
    x01 = avg_pool2d(pad2d(x, (0, 0, 0, 1))[:,:,:-1,1:], 1, stride=2, padding=0)
    x10 = avg_pool2d(pad2d(x, (0, 1, 0, 0))[:,:,1:,:-1], 1, stride=2, padding=0)
    x11 = avg_pool2d(pad2d(x, (0, 1, 0, 1))[:,:,1:,1:], 1, stride=2, padding=0)
    return concat([x00, x01, x10, x11], dim=1)




class SGDW(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in constrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGDW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = d_p.clone()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss

class AdamW(Optimizer):
    """Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # if group['weight_decay'] != 0:
                #     grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1


                p.data.addcdiv_(-step_size, exp_avg, denom)

                if group['weight_decay'] != 0:
                    decay_step_size = -step_size * group['weight_decay']
                    p.data.add_(decay_step_size, p.data)

        return loss

class ShiftLayer(nn.Module):
    def __init__(self, ch_in, kernel_size=3, stride=1):
        super(ShiftLayer, self).__init__()
        self.ch_in = ch_in
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels_per_group = self.ch_in // (self.kernel_size ** 2)

    def forward(self, x):
        x_pad = F.pad(x, (1, 1, 1, 1))
        chg = self.channels_per_group
        s = self.stride
        cat_layers = []
        i = 0
        cat_layers += [x_pad[:, i * chg: (i + 1) * chg, :-2:s, 1:-1:s]]# Bottom shift, grab the Top element
        i = 1
        cat_layers += [x_pad[:, i * chg: (i + 1) * chg, 2::s, 1:-1:s]]# Top shift, grab the Bottom element
        i = 2
        cat_layers += [x_pad[:, i * chg: (i + 1) * chg, 1:-1:s, :-2:s]]# Right shift, grab the left element
        i = 3
        cat_layers += [x_pad[:, i * chg: (i + 1) * chg, 1:-1:s, 2::s]]# Left shift, grab the right element
        i = 4
        cat_layers += [x_pad[:, i * chg: (i + 1) * chg, :-2:s, :-2:s]]# Bottom Right shift, grab the Top left element
        i = 5
        cat_layers += [x_pad[:, i * chg: (i + 1) * chg, :-2:s, 2::s]]# Bottom Left shift, grab the Top right element
        i = 6
        cat_layers += [x_pad[:, i * chg: (i + 1) * chg, 2::s, :-2:s]]# Top Right shift, grab the Bottom Left element
        i = 7
        cat_layers += [x_pad[:, i * chg: (i + 1) * chg, 2::s, 2::s]]# Top Left shift, grab the Bottom Right element
        i = 8
        cat_layers += [x_pad[:, i * chg:, 1:-1:s, 1:-1:s]]
        return torch.cat(cat_layers, 1)

class Shift1dxn(nn.Module):
    def __init__(self, ch_in, stride=3, nb_shifts=3):
        super(Shift2dxn, self).__init__()
        self.ch_in = ch_in
        self.stride = stride
        self.nb_shifts = nb_shifts

    def forward(self, x):
        s = self.stride
        x_ = []
        x_.append(x)
        for n in range(self.nb_shifts):
            m = (n+1) * s
            x_.append( pad2d(x, (0, 0, 0, m))[:,:,m:,:])
            x_.append( pad2d(x, (0, 0, m, 0))[:,:,:-m,:])
        return torch.cat(x_, 1)

class Shift2d4xn(nn.Module):
    def __init__(self, ch_in, stride=3, nb_shifts=3):
        super(Shift2d4xn, self).__init__()
        self.ch_in = ch_in
        self.stride = stride
        self.nb_shifts = nb_shifts

    def forward(self, x):
        s = self.stride
        x_ = []
        x_.append(x)
        for n in range(self.nb_shifts):
            m = (n+1) * s
            x_.append( pad2d(x, (0, 0, 0, m))[:,:,m:,:])
            x_.append( pad2d(x, (0, 0, m, 0))[:,:,:-m,:])
            x_.append( pad2d(x, (0, m, 0, 0))[:,:,:,m:])
            x_.append( pad2d(x, (m, 0, 0, 0))[:,:,:,:-m])
        return torch.cat(x_, 1)

class Shift2d8xn(nn.Module):
    def __init__(self, ch_in, stride=3, nb_shifts=3):
        super(Shift2d8xn, self).__init__()
        self.ch_in = ch_in
        self.stride = stride
        self.nb_shifts = nb_shifts

    def forward(self, x):
        s = self.stride
        x_ = []
        x_.append(x)
        for n in range(self.nb_shifts):
            m = (n+1) * s
            x_.append( pad2d(x, (0, 0, 0, m))[:,:,m:,:])
            x_.append( pad2d(x, (0, 0, m, 0))[:,:,:-m,:])
            x_.append( pad2d(x, (0, m, 0, 0))[:,:,:,m:])
            x_.append( pad2d(x, (m, 0, 0, 0))[:,:,:,:-m])
            x_.append( pad2d(x, (0, m, 0, m))[:,:,m:,m:])
            x_.append( pad2d(x, (m, 0, m, 0))[:,:,:-m,:-m])
            x_.append( pad2d(x, (m, 0, 0, m))[:,:,m:,:-m])
            x_.append( pad2d(x, (0, m, m, 0))[:,:,:-m,m:])
            
        return torch.cat(x_, 1)


from torch.autograd import Variable
if __name__ == '__main__':
    # check errors and size
    """
    x = Variable(torch.arange(0, 49*9)).view(1, 9, 7, 7)
    print(x.size())
    print(x)

    s1 = ShiftLayer(9)
    c1 = nn.Conv2d(9, 3, kernel_size=1, stride=2, padding=0, bias=False)
    print(s1(x))
    print(c1(s1(x)))
    """
    s1 = Shift2d4xn(2,2,2)
    x = Variable(torch.arange(0, 36*2)).view(1, 2, 6, 6)
    print(x)

    print(s1(x))

