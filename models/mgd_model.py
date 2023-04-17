import torch
import numpy as np
from copy import deepcopy
from scipy.optimize import minimize
from utils.registry import MODEL_REGISTRY
from models.base_model import BaseModel
from utils.min_norm_solvers import MinNormSolver


@MODEL_REGISTRY.register()
class MGD(BaseModel):
    def __init__(self, opt, arch_opt=None):
        super(MGD, self).__init__(opt=opt, arch_opt=arch_opt)

    def mgd(self, grads):
        """
            Copy from the implementation of CAGrad: https://github.com/Cranial-XIX/CAGrad
        """
        grads_cpu = grads.t().cpu()
        sol, min_norm = MinNormSolver.find_min_norm_element([grads_cpu[t] for t in range(grads.shape[-1])])
        w = torch.FloatTensor(sol).to(grads.device)
        g = grads.mm(w.view(-1, 1)).view(-1)
        return g

    def manipulate_grad(self, losses):
        for i, task in enumerate(self.tasks):
            if i < self.n_tasks:
                losses[task].backward(retain_graph=True)
            else:
                losses[task].backward()

            self._grad2vec(i)
            self.network.zero_grad_shared_modules()

        g = self.mgd(self.grads)

        # overwrite the gradients
        self.overwrite_grad(g)

        return