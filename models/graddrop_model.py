import torch
import numpy as np
from copy import deepcopy
from scipy.optimize import minimize
from utils.registry import MODEL_REGISTRY
from models.base_model import BaseModel


@MODEL_REGISTRY.register()
class Graddrop(BaseModel):
    def __init__(self, opt, arch_opt=None):
        super(Graddrop, self).__init__(opt=opt, arch_opt=arch_opt)

    def graddrop(self, grads):
        """
            Copy from the implementation of CAGrad: https://github.com/Cranial-XIX/CAGrad
        """
        P = 0.5 * (1. + grads.sum(1) / (grads.abs().sum(1) + 1e-8))
        U = torch.rand_like(grads[:, 0])
        M = P.gt(U).view(-1, 1) * grads.gt(0) + P.lt(U).view(-1, 1) * grads.lt(0)
        g = (grads * M.float()).mean(1)
        return g

    def manipulate_grad(self, losses):
        for i, task in enumerate(self.tasks):
            if i < self.n_tasks:
                losses[task].backward(retain_graph=True)
            else:
                losses[task].backward()

            self._grad2vec(i)
            self.network.zero_grad_shared_modules()

        g = self.graddrop(self.grads)

        # overwrite the gradients
        self.overwrite_grad(g)

        return