import torch
import os.path as osp
from archs import get_arch_object
from utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class BaseModel():
    """Base Model."""
    def __init__(self, opt, arch_opt=None):
        self.tasks = opt.tasks
        self.n_tasks = len(self.tasks)
        network = get_arch_object(opt.arch)
        self.network = network(**arch_opt)

        self.network = self.network.cuda()
        self.grad_dims = self._get_grad_dims()
        self.initilize_grads()

    # ----------- Common Methods ------------
    def parameters(self):
        return self.network.parameters()

    def initilize_grads(self):
        """
        Initialize the gradients. Need to be called before every training iteration.
        """
        self.grads = torch.zeros(sum(self.grad_dims), self.n_tasks).cuda()

    def train(self):
        self.network.train()

    def eval(self):
        self.network.eval()

    # ----------- Gradient Manipulation Methods ------------
    def _grad2vec(self, task_id):
        # store the gradients for current task
        self.grads[:, task_id].fill_(0.0)
        cnt = 0
        for name, p in self.network.shared_parameters().items():
            grad = p.grad
            if grad is not None:
                grad_cur = grad.data.detach().clone()
                beg = 0 if cnt == 0 else sum(self.grad_dims[:cnt])
                en = sum(self.grad_dims[:cnt + 1])
                self.grads[beg:en, task_id].copy_(grad_cur.data.view(-1))
            cnt += 1

    def _get_grad_dims(self):
        """
        Get the number of parameters in shared layers.
        """
        grad_dims = []
        for key, param in self.network.shared_parameters().items():
            grad_dims.append(param.data.numel())

        return grad_dims

    def overwrite_grad(self, newgrad):
        newgrad = newgrad * self.n_tasks  # to match the sum loss
        cnt = 0
        for name, param in self.network.shared_parameters().items():
            beg = 0 if cnt == 0 else sum(self.grad_dims[:cnt])
            en = sum(self.grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(param.data.size())
            param.grad = this_grad.data.clone()
            cnt += 1

    def manipulate_grad(self, losses):
        # Function that need to be overwritten by subclasses
        pass

    def model_size(self):
        return self.network.model_size()

    # ----------- Train, Eval, Save ------------
    def save(self, path, name, epoch, iterations, seed):
        torch.save(self.network.state_dict(), osp.join(path, f'{seed}_{name}_ep{epoch}_iter{iterations}.pt'))

    def train_loop(self, inputs, targets, loss, optimizer):
        # initialize the tensor that store gradients
        self.initilize_grads()

        output = self.network(inputs)
        losses = loss(output, targets)

        optimizer.zero_grad()
        self.manipulate_grad(losses)
        optimizer.step()

        # clear the gradients
        del self.grads

        return output, losses

    @torch.no_grad()
    def eval_loop(self, inputs, targets, loss):
        output = self.network(inputs)
        losses = loss(output, targets)
        return output, losses