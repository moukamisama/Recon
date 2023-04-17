import torch
import numpy as np
from scipy.optimize import minimize
from utils.registry import MODEL_REGISTRY
from models.base_model import BaseModel

@MODEL_REGISTRY.register()
class CAGrad(BaseModel):
    def __init__(self, opt, arch_opt=None):
        super(CAGrad, self).__init__(opt=opt, arch_opt=arch_opt)
        self.alpha = opt.alpha

    def cagrad(self, grads, alpha=0.5, rescale=1):
        """
        Copy from the implementation of CAGrad: https://github.com/Cranial-XIX/CAGrad
        """
        GG = grads.t().mm(grads).cpu()  # [num_tasks, num_tasks]
        g0_norm = (GG.mean() + 1e-8).sqrt()  # norm of the average gradient

        n_tasks = self.n_tasks
        x_start = np.ones(n_tasks) / n_tasks
        bnds = tuple((0, 1) for x in x_start)
        cons = ({'type': 'eq', 'fun': lambda x: 1 - sum(x)})
        A = GG.numpy()
        b = x_start.copy()
        c = (alpha * g0_norm + 1e-8).item()

        def objfn(x):
            return (x.reshape(1, n_tasks).dot(A).dot(b.reshape(n_tasks, 1)) + c * np.sqrt(
                x.reshape(1, n_tasks).dot(A).dot(x.reshape(n_tasks, 1)) + 1e-8)).sum()

        res = minimize(objfn, x_start, bounds=bnds, constraints=cons)
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(grads.device)
        gw = (grads * ww.view(1, -1)).sum(1)
        gw_norm = gw.norm()
        lmbda = c / (gw_norm + 1e-8)
        g = grads.mean(1) + lmbda * gw
        if rescale == 0:
            return g
        elif rescale == 1:
            return g / (1 + alpha ** 2)
        else:
            return g / (1 + alpha)

    def manipulate_grad(self, losses):
        for i, task in enumerate(self.tasks):
            if i < self.n_tasks:
                losses[task].backward(retain_graph=True)
            else:
                losses[task].backward()

            self._grad2vec(i)
            self.network.zero_grad_shared_modules()

        g = self.cagrad(self.grads, self.alpha, rescale=1)

        # overwrite the gradients
        self.overwrite_grad(g)

        return