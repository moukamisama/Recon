import itertools
from collections import defaultdict
import torch
import torch.nn as nn


class WeightedSumLoss(nn.Module):
    """
    Overall multi-task loss, consisting of a weighted sum of individual task losses. With
    optional resource loss.
    """

    def __init__(self, tasks_name_to_loss_map, tasks):
        super(WeightedSumLoss, self).__init__()
        self.tasks_name_to_loss_map = tasks_name_to_loss_map
        self.tasks = tasks
        # do not support learnable task weights
        self.task_weights = torch.ones(len(tasks_name_to_loss_map))

    def forward(self, outputs, targets):
        losses = {}
        for k, task in enumerate(self.tasks):
            func = self.tasks_name_to_loss_map[task]
            losses[task] = func(outputs[task], targets[task]) * self.task_weights[k]

        return losses


