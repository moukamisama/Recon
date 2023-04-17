import torch
import numpy as np
from copy import deepcopy
from scipy.optimize import minimize
from utils.registry import MODEL_REGISTRY
from models.base_model import BaseModel

@MODEL_REGISTRY.register()
class PCGrad(BaseModel):
    def __init__(self, opt, arch_opt=None):
        super(PCGrad, self).__init__(opt=opt, arch_opt=arch_opt)
        self.rng = np.random.default_rng()

    def pcgrad(self, grads, rng):
        """
            Copy from the implementation of CAGrad: https://github.com/Cranial-XIX/CAGrad
        """
        grad_vec = grads.t()

        shuffled_task_indices = np.zeros((self.n_tasks, self.n_tasks - 1), dtype=int)
        for i in range(self.n_tasks):
            task_indices = np.arange(self.n_tasks)
            task_indices[i] = task_indices[-1]
            shuffled_task_indices[i] = task_indices[:-1]
            rng.shuffle(shuffled_task_indices[i])
        shuffled_task_indices = shuffled_task_indices.T

        normalized_grad_vec = grad_vec / (
                grad_vec.norm(dim=1, keepdim=True) + 1e-8
        )  # self.n_tasks x dim
        modified_grad_vec = deepcopy(grad_vec)
        for task_indices in shuffled_task_indices:
            normalized_shuffled_grad = normalized_grad_vec[
                task_indices
            ]  # self.n_tasks x dim
            dot = (modified_grad_vec * normalized_shuffled_grad).sum(
                dim=1, keepdim=True
            )  # self.n_tasks x dim
            modified_grad_vec -= torch.clamp_max(dot, 0) * normalized_shuffled_grad
        g = modified_grad_vec.mean(dim=0)
        return g

    def manipulate_grad(self, losses):
        for i, task in enumerate(self.tasks):
            if i < self.n_tasks:
                losses[task].backward(retain_graph=True)
            else:
                losses[task].backward()

            self._grad2vec(i)
            self.network.zero_grad_shared_modules()

        g = self.pcgrad(self.grads, self.rng)

        # overwrite the gradients
        self.overwrite_grad(g)

        return