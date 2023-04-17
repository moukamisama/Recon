import torch
import torch.nn as nn
import torch.nn.functional as F

class semantic_loss(nn.Module):
    def __init__(self):
        super(semantic_loss, self).__init__()

    def forward(self, outputs, targets):
        outputs = F.log_softmax(outputs, dim=1)
        loss = F.nll_loss(outputs, targets, ignore_index=-1)

        return loss

class depth_loss(nn.Module):
    def __init__(self):
        super(depth_loss, self).__init__()


    def forward(self, outputs, targets):
        binary_mask = (torch.sum(targets, dim=1) != 0).float().unsqueeze(1).cuda()
        # depth loss: l1 norm
        loss = torch.sum(torch.abs(outputs - targets) * binary_mask) / \
               torch.nonzero(binary_mask, as_tuple=False).size(0)

        return loss

class normal_loss(nn.Module):
    def __init__(self):
        super(normal_loss, self).__init__()

    def forward(self, outputs, targets):
        binary_mask = (torch.sum(targets, dim=1) != 0).float().unsqueeze(1).cuda()
        # normal loss: dot product
        loss = 1 - torch.sum((outputs * targets) * binary_mask) / \
               torch.nonzero(binary_mask, as_tuple=False).size(0)

        return loss