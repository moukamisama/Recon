import torch
import numpy as np

class Averager():
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.data = []

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.data = []

    def add(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.data.append(val)

    def item(self):
        return self.avg

    def obtain_data(self):
        return self.data

    def __len__(self):
        return len(self.data)

class classify_acc_metric():
    def __init__(self, task, mode='train'):
        if mode == 'train':
            self.mode = 'train'
        elif mode == 'test':
            self.mode = 'test'
        else:
            raise NotImplementedError

        self.task_name = task
        self.current_acc = 0
        self.avg_acc = Averager()

    @torch.no_grad()
    def __call__(self, output_info):
        outputs = output_info['outputs'][self.task_name]
        targets = output_info['targets'][self.task_name]

        estimate_labels = torch.argmax(outputs, dim=1)
        acc = (estimate_labels == targets).sum() / outputs.size(0)
        self.current_acc = acc.item()
        self.avg_acc.add(self.current_acc)

        output = {self.obtain_metric_name(): self.current_acc}

        return output

    # should call this function after each epoch
    def reset(self):
        self.current_acc = 0
        self.avg_acc.reset()

    # calculate the average accuracy
    def avg(self):
        return {self.obtain_metric_name(prefix='avg'): self.avg_acc.item()}

    def obtain_metric_name(self, prefix=None):
        if prefix is None:
            return f'{self.mode}_{self.task_name}_acc'
        else:
            return f'{self.mode}_{prefix}_{self.task_name}_acc'

class losses_metric():
    def __init__(self, task, mode='train'):
        if mode == 'train':
            self.mode = 'train'
        elif mode == 'test':
            self.mode = 'test'
        else:
            raise NotImplementedError

        self.task_name = task
        self.current_loss = 0
        self.avg_loss = Averager()

    @torch.no_grad()
    def __call__(self, output_info):
        losses = output_info['losses'][self.task_name]
        self.current_loss = losses.item()
        self.avg_loss.add(self.current_loss)

        output = {self.obtain_metric_name(): self.current_loss}
        return output

    # should call this function after each epoch
    def reset(self):
        self.current_loss = 0
        self.avg_loss.reset()

    # calculate the average loss
    def avg(self):
        return {self.obtain_metric_name(prefix='avg'): self.avg_loss.item()}

    def obtain_metric_name(self, prefix=None):
        if prefix is None:
            return f'{self.mode}_{self.task_name}_loss'
        else:
            return f'{self.mode}_{prefix}_{self.task_name}_loss'

class depth_metric():
    def __init__(self, task, mode='train'):
        if mode == 'train':
            self.mode = 'train'
        elif mode == 'test':
            self.mode = 'test'
        else:
            raise NotImplementedError

        self.task_name = task
        self.current_abs_err = 0
        self.current_rel_err = 0
        self.avg_abs_err = Averager()
        self.avg_rel_err = Averager()

    @torch.no_grad()
    def __call__(self, output_info):
        x_pred = output_info['outputs'][self.task_name]
        x_output = output_info['targets'][self.task_name]

        device = x_pred.device
        binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).to(device)
        x_pred_true = x_pred.masked_select(binary_mask)
        x_output_true = x_output.masked_select(binary_mask)
        abs_err = torch.abs(x_pred_true - x_output_true)
        rel_err = torch.abs(x_pred_true - x_output_true) / torch.abs(x_output_true)

        self.current_abs_err = (torch.sum(abs_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item()
        self.current_rel_err = (torch.sum(rel_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item()

        self.avg_abs_err.add(self.current_abs_err)
        self.avg_rel_err.add(self.current_rel_err)

        # output = {self.obtain_metric_name(postfix='abs_err'): self.current_abs_err,
        #             self.obtain_metric_name(postfix='rel_err'): self.current_rel_err}
        #
        # return output

        # only support avg metric
        return {}

    def reset(self):
        self.current_abs_err = 0
        self.current_rel_err = 0
        self.avg_abs_err.reset()
        self.avg_rel_err.reset()

    def avg(self):
        return {self.obtain_metric_name(prefix='avg', postfix='abs_err'): self.avg_abs_err.item(),
                self.obtain_metric_name(prefix='avg', postfix='rel_err'): self.avg_rel_err.item()}

    def obtain_metric_name(self, prefix=None, postfix='abs_err'):
        if prefix is None:
            return f'{self.mode}_{self.task_name}_{postfix}'
        else:
            return f'{self.mode}_{prefix}_{self.task_name}_{postfix}'

class normal_metric():
    def __init__(self, task, mode='train'):
        if mode == 'train':
            self.mode = 'train'
        elif mode == 'test':
            self.mode = 'test'
        else:
            raise NotImplementedError

        self.task_name = task
        self.current_mean = 0
        self.current_median = 0
        self.current_first_quartile = 0
        self.current_second_quartile = 0
        self.current_third_quartile = 0

        self.avg_mean = Averager()
        self.avg_median = Averager()
        self.avg_first_quartile = Averager()
        self.avg_second_quartile = Averager()
        self.avg_third_quartile = Averager()

    @torch.no_grad()
    def __call__(self, output_info):
        x_pred = output_info['outputs'][self.task_name]
        x_output = output_info['targets'][self.task_name]

        binary_mask = (torch.sum(x_output, dim=1) != 0)
        error = torch.acos(
            torch.clamp(torch.sum(x_pred * x_output, 1).masked_select(binary_mask), -1, 1)).detach().cpu().numpy()

        error = np.degrees(error)
        self.current_mean = np.mean(error)
        self.current_median = np.median(error)
        self.current_first_quartile = np.mean(error < 11.25)
        self.current_second_quartile = np.mean(error < 22.5)
        self.current_third_quartile = np.mean(error < 30)

        self.avg_mean.add(self.current_mean)
        self.avg_median.add(self.current_median)
        self.avg_first_quartile.add(self.current_first_quartile)
        self.avg_second_quartile.add(self.current_second_quartile)
        self.avg_third_quartile.add(self.current_third_quartile)

        # output = {self.obtain_metric_name(postfix='mean'): self.current_mean,
        #             self.obtain_metric_name(postfix='median'): self.current_median,
        #             self.obtain_metric_name(postfix='first_quartile'): self.current_first_quartile,
        #             self.obtain_metric_name(postfix='second_quartile'): self.current_second_quartile,
        #             self.obtain_metric_name(postfix='third_quartile'): self.current_third_quartile}
        #
        # return output

        # only support avg metric
        return {}

    def reset(self):
        self.current_mean = 0
        self.current_median = 0
        self.current_first_quartile = 0
        self.current_second_quartile = 0
        self.current_third_quartile = 0

        self.avg_mean.reset()
        self.avg_median.reset()
        self.avg_first_quartile.reset()
        self.avg_second_quartile.reset()
        self.avg_third_quartile.reset()

    def avg(self):
        return {self.obtain_metric_name(prefix='avg', postfix='mean'): self.avg_mean.item(),
        self.obtain_metric_name(prefix='avg', postfix='median'): self.avg_median.item(),
        self.obtain_metric_name(prefix='avg', postfix='11d25'): self.avg_first_quartile.item(),
        self.obtain_metric_name(prefix='avg', postfix='22d5'): self.avg_second_quartile.item(),
        self.obtain_metric_name(prefix='avg', postfix='30'): self.avg_third_quartile.item()}


    def obtain_metric_name(self, prefix=None, postfix='mean'):
        if prefix is None:
            return f'{self.mode}_{self.task_name}_{postfix}'
        else:
            return f'{self.mode}_{prefix}_{self.task_name}_{postfix}'

class ConfMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, pred, target):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=pred.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + pred[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def get_metrics(self):
        h = self.mat.float()
        acc = torch.diag(h).sum() / h.sum()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return torch.mean(iu), acc

class semantic_metric():
    def __init__(self, n_classes, task, mode='train'):
        if mode == 'train':
            self.mode = 'train'
        elif mode == 'test':
            self.mode = 'test'
        else:
            raise NotImplementedError

        self.task_name = task
        self.conf_matrix = ConfMatrix(n_classes)

    @torch.no_grad()
    def __call__(self, output_info):
        pred = output_info['outputs'][self.task_name]
        target = output_info['targets'][self.task_name]

        pred = pred.argmax(1).flatten().cpu()
        target = target.flatten().cpu()

        self.conf_matrix.update(pred, target)

        # only support avg metric
        return {}

    def reset(self):
        self.conf_matrix.mat = None

    def avg(self):
        iu, acc = self.conf_matrix.get_metrics()
        return {self.obtain_metric_name(prefix='avg', postfix='iu'): iu.item(),
                self.obtain_metric_name(prefix='avg', postfix='acc'): acc.item()}

    def obtain_metric_name(self, prefix=None, postfix='mean'):
        if prefix is None:
            return f'{self.mode}_{self.task_name}_{postfix}'
        else:
            return f'{self.mode}_{prefix}_{self.task_name}_{postfix}'