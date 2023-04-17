import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

class Linear_recon(nn.Module):
    def __init__(self, in_features, out_features, n_tasks=1):
        super(Linear_recon, self).__init__()
        self.n_tasks = n_tasks
        self.m_list = nn.ModuleList([nn.Linear(in_features, out_features) for i in range(n_tasks)])

    def set_n_tasks(self, n_tasks=1):
        if n_tasks >= self.n_tasks:
            gap = n_tasks - self.n_tasks
            self.n_tasks = n_tasks
            for i in range(gap):
                module = deepcopy(self.m_list[0])
                self.m_list.append(module)
        else:
            raise ValueError('Can not decrease the number of tasks in fw module')

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]

        out = []
        if len(x) == 1 and self.n_tasks > 1:
            for i, ln in enumerate(self.m_list):
                o = ln(x[0])
                out.append(o)
        elif len(x) > 1 and self.n_tasks == 1:
            for i, x_i in enumerate(x):
                o = self.m_list[0](x_i)
                out.append(o)
        elif len(x) == self.n_tasks:
            for i, ln in enumerate(self.m_list):
                o = ln(x[i])
                out.append(o)
        else:
            raise ValueError('Error')

        return out

class Conv2d_recon(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, groups=1, dilation=1,
                 n_tasks=1):
        super(Conv2d_recon, self).__init__()
        self.n_tasks = n_tasks
        self.m_list = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
                       groups=groups, bias=bias) for i in range(n_tasks)])

    def set_n_tasks(self, n_tasks=1):
        if n_tasks >= self.n_tasks:
            gap = n_tasks - self.n_tasks
            self.n_tasks = n_tasks
            for i in range(gap):
                module = deepcopy(self.m_list[0])
                self.m_list.append(module)
        else:
            raise ValueError('Can not decrease the number of tasks in fw module')

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]

        out = []
        if len(x) == 1 and self.n_tasks > 1:
            for i, conv in enumerate(self.m_list):
                o = conv(x[0])
                out.append(o)
        elif len(x) > 1 and self.n_tasks == 1:
            for i, x_i in enumerate(x):
                o = self.m_list[0](x_i)
                out.append(o)
        elif len(x) == self.n_tasks:
            for i, conv in enumerate(self.m_list):
                o = conv(x[i])
                out.append(o)
        else:
            raise ValueError('Error')
        return out

class BatchNorm2d_recon(nn.Module):
    def __init__(self, num_features, n_tasks=1):
        super(BatchNorm2d_recon, self).__init__()
        self.n_tasks = n_tasks
        self.m_list = nn.ModuleList([nn.BatchNorm2d(num_features) for i in range(n_tasks)])

    def set_n_tasks(self, n_tasks=1):
        if n_tasks >= self.n_tasks:
            gap = n_tasks - self.n_tasks
            self.n_tasks = n_tasks
            for i in range(gap):
                module = deepcopy(self.m_list[0])
                self.m_list.append(module)
        else:
            raise ValueError('Can not decrease the number of tasks in fw module')

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]

        out = []
        if len(x) == 1 and self.n_tasks > 1:
            for i, bn in enumerate(self.m_list):
                o = bn(x[0])
                out.append(o)
        elif len(x) > 1 and self.n_tasks == 1:
            for i, x_i in enumerate(x):
                o = self.m_list[0](x_i)
                out.append(o)
        elif len(x) == self.n_tasks:
            for i, bn in enumerate(self.m_list):
                o = bn(x[i])
                out.append(o)
        else:
            raise ValueError('Error')

        return out

class ReLU_recon(nn.ReLU):
    def __init__(self, inplace=False):
        super(ReLU_recon, self).__init__(inplace)

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]

        out = []
        for x_i in x:
            out.append(super(ReLU_recon, self).forward(x_i))

        return out

class Sigmoid_recon(nn.Sigmoid):
    def __init__(self):
        super(Sigmoid_recon, self).__init__()

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]

        out = []
        for x_i in x:
            out.append(super(Sigmoid_recon, self).forward(x_i))

        return out

class MaxPool2d_recon(nn.MaxPool2d):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False):
        super(MaxPool2d_recon, self).__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]

        if self.return_indices:
            out = []
            indices = []

            for x_i in x:
                o, i = super(MaxPool2d_recon, self).forward(x_i)
                out.append(o)
                indices.append(i)

            return out, indices
        else:
            out = []
            for x_i in x:
                out.append(super(MaxPool2d_recon, self).forward(x_i))

        return out

class MaxUnpool2d_recon(nn.MaxUnpool2d):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(MaxUnpool2d_recon, self).__init__(kernel_size, stride, padding)

    def forward(self, x, indices):
        if not isinstance(x, list):
            x = [x]
        if not isinstance(x, list):
            indices = [indices]

        # assert len(x) == len(indices)

        if len(x) == len(indices):
            out = []
            for i, x_i in enumerate(x):
                out.append(super(MaxUnpool2d_recon, self).forward(x_i, indices[i]))
        elif len(x) > 1 and len(indices) == 1:
            out = []
            for i, x_i in enumerate(x):
                out.append(super(MaxUnpool2d_recon, self).forward(x_i, indices[0]))
        else:
            raise ValueError('Error!')

        return out

class AdaptiveAvgPool2d_recon(nn.AdaptiveAvgPool2d):
    def __init__(self, output_size):
        super(AdaptiveAvgPool2d_recon, self).__init__(output_size)

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]

        out = []
        for x_i in x:
            out.append(super(AdaptiveAvgPool2d_recon, self).forward(x_i))

        return out


def dot_recon(x, y):
    if not isinstance(x, list):
        x = [x]
    if not isinstance(y, list):
        y = [y]

    out = []
    if len(x) == 1 and len(y) > 1:
        for i, y_i in enumerate(y):
            o = x[0] * y_i
            out.append(o)
    elif len(x) > 1 and len(y) == 1:
        for i, x_i in enumerate(x):
            o = x_i * y[0]
            out.append(o)
    elif len(x) == len(y):
        for i, x_i in enumerate(x):
            o = x_i * y[i]
            out.append(o)
    else:
        raise ValueError('Error')

    return out


def cat_recon(x, y, dim=0):
    if not isinstance(x, list):
        x = [x]
    if not isinstance(y, list):
        y = [y]

    out = []
    if len(x) == 1 and len(y) > 1:
        for i, y_i in enumerate(y):
            o = torch.cat((x[0], y_i), dim=dim)
            out.append(o)
    elif len(x) > 1 and len(y) == 1:
        for i, x_i in enumerate(x):
            o = torch.cat((x_i, y[0]), dim=dim)
            out.append(o)
    elif len(x) == len(y):
        for i, x_i in enumerate(x):
            o = torch.cat((x_i, y[i]), dim=dim)
            out.append(o)
    else:
        raise ValueError('Error')

    return out


def interpolate_recon(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None,
                   antialias=False):
    if not isinstance(input, list):
        input = [input]

    out = []
    for x in input:
        o = F.interpolate(x, size, scale_factor, mode, align_corners, recompute_scale_factor, antialias)
        out.append(o)

    return out


