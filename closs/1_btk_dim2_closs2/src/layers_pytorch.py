import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer, required
import torch.backends.cudnn as cudnn

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

def sign(x):
    return torch.sign(x)

def step(x):
    return (torch.sign(x) + 1.)/2.


# ----------------------------------------------------------------------------------------------------------------------
class BaseNet(object):
    def __init__(self):
        self.epoch = 0

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
            aux = aux.max(a)[0]
        return aux
    else:
        return x.max(axis)[0]


def dropout(x, keep_prob=0.5):
    return nn.Dropout(1.-keep_prob)(x)

def pad2d(x, pad=(0, 0, 0, 0)):
    x = nn.ConstantPad2d(pad, 0)(x)
    return x




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

class CenterLoss(nn.Module):
    def __init__(self, nb_class, dim, loss_weight=1e-2, alpha=0.05):
        super(CenterLoss, self).__init__()
        self.nb_class = nb_class
        self.dim = dim
        self.loss_weight = loss_weight
        self.alpha =alpha
        self.register_buffer('centers',torch.zeros(self.nb_class, self.dim))

    def forward(self, x, y):
        centers = Variable(self.centers, requires_grad=False)
        c = centers.index_select(0, y)
        diff = c - x
        loss = self.loss_weight * (diff.pow(2).sum(1)).mean()
        batch_size = x.size(0)
        z = Variable(torch.zeros(batch_size, self.nb_class), requires_grad=False)
        if y.is_cuda:
            z = z.cuda()
        z.scatter_(1, y.view(-1, 1), 1)
        n = z.sum(0) + 1
        for i in range(batch_size):
            centers[y[i].data] -= self.alpha * diff[i] / n[y[i]]

        return loss


class CenterLoss2(nn.Module):
    def __init__(self, nb_class, dim, loss_weight=1e-2, alpha=0.05):
        super(CenterLoss2, self).__init__()
        self.nb_class = nb_class
        self.dim = dim
        self.loss_weight = loss_weight
        self.alpha =alpha
        self.register_buffer('centers',torch.zeros(self.nb_class, self.dim))

    def forward(self, x, y):
        batch_size = y.size(0)
        features_dim = x.size(1)
        y_expand = y.view(batch_size,1).expand(batch_size,features_dim)
        centers_var = Variable(self.centers)
        centers_batch = centers_var.gather(0,y_expand).cuda()
        criterion = nn.MSELoss()
        loss = criterion(x,  centers_batch)
        diff = centers_batch - x
        unique_label, unique_reverse, unique_count = np.unique(y.cpu().data.numpy(), return_inverse=True, return_counts=True)
        appear_times = torch.from_numpy(unique_count).gather(0,torch.from_numpy(unique_reverse))
        appear_times_expand = appear_times.view(-1,1).expand(batch_size,features_dim).type(torch.FloatTensor)
        diff_cpu = diff.cpu().data / appear_times_expand.add(1e-6)
        diff_cpu = self.alpha * diff_cpu
        for i in range(batch_size):
            self.centers[y.data[i]] -= diff_cpu[i].type(self.centers.type())
        return self.loss_weight*loss
        
        


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


class Shift2dxn(nn.Module):
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

class Shift1dxn(nn.Module):
    def __init__(self, ch_in, stride=3, nb_shifts=3):
        super(Shift1dxn, self).__init__()
        self.ch_in = ch_in
        self.stride = stride
        self.nb_shifts = nb_shifts

    def forward(self, x):
        x = x.unsqueeze(3)
        s = self.stride
        x_ = []
        x_.append(x)
        for n in range(self.nb_shifts):
            m = (n+1) * s
            x_.append( pad2d(x, (0, 0, 0, m))[:,:,m:,:])
            x_.append( pad2d(x, (0, 0, m, 0))[:,:,:-m,:])

        return torch.cat(x_, 1).squeeze(3)

from torch.autograd import Variable
if __name__ == '__main__':
    # check errors and size
    # x = Variable(torch.arange(0, 49*9)).view(1, 9, 7, 7)
    # print(x.size())
    # print(x)
    #
    # s1 = ShiftLayer(9)
    # c1 = nn.Conv2d(9, 3, kernel_size=1, stride=2, padding=0, bias=False)
    # print(s1(x))
    # print(c1(s1(x)))


    # s1 = Shift2dxn(3,3,3)
    # x = Variable(torch.arange(0, 15*3)).view(1, 3, 15, 1)
    # print(x)
    # print(s1(x))

    s1 = Shift1dxn(3,3,3)
    x = Variable(torch.arange(0, 15*3)).view(1, 3, 15)
    print(x)
    print(s1(x))
