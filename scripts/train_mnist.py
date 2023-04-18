import argparse
import math

import wandb
import pickle
import logging
import time
import os.path as osp

import torch
import torch.utils.data
import torch.optim as optim
import torch.nn as nn

from utils import make_exp_and_log_dirs, set_random_seed

from utils.logger import get_root_logger, init_wandb_logger, MessageLogger, \
    init_tb_logger, get_env_info, AvgTimer

from models import get_model_object
from losses.weightedsum_loss import WeightedSumLoss
from criterion import losses_metric, classify_acc_metric

def project_args(parser):
    parser.add_argument('--name', default='Baseline', type=str, help='The name of the experiment')
    parser.add_argument('--project', default='multiFashion+MNIST', type=str, help='The name of the project used in wandb')
    parser.add_argument('--path', default='./logs', type=str, help='The path of the experiment')

def training_args(parser):
    parser.add_argument('--dataroot', default='./datasets/multiFashion+MNIST/multi_fashion_and_mnist.pickle', type=str, help='Dataset root')
    parser.add_argument('--optimizer', default='SGD', type=str, help='The type of optimizer')
    parser.add_argument('--loss', default='WeightedSumLoss', type=str, help='The type of loss function')
    parser.add_argument('--lr', default=1e-1, type=float, help='The learning rate')
    parser.add_argument('--n_epoch', default=120, type=int, help='The seed')
    parser.add_argument('--eval_freq', default=5, type=int, help='The freq of evaluation (/epochs)')
    parser.add_argument('--print_freq', default=20, type=int, help='The freq of print (/iterations)')
    parser.add_argument('--bz', default=256, type=int, help='The batch size')
    parser.add_argument('--milestones', default=[60, 90], type=int, nargs='+', help='The milestones of scheduler')
    parser.add_argument('--gamma', default=0.1, type=float, help='The gamma of scheduler')
    parser.add_argument('--seed', default=0, type=int, help='the seed')
    parser.add_argument('--num_threads', default=1, type=int, help='The number of CPU threads')

def task_args(parser):
    parser.add_argument('--tasks', default=['T1', 'T2'], type=str, nargs='+', help='The name list of tasks')

def model_args(parser):
    parser.add_argument('--arch', default='MnistResNet_recon', type=str, help='The architecture')
    parser.add_argument('--method', default='Baseline',
                        choices=['Baseline', 'CAGrad', 'Graddrop', 'PCGrad', 'MGD', 'Recon'],
                        type=str, help='The method we used')

def branch_args(parser):
    parser.add_argument('--topK', default=15, type=int, help='The topK layers will be turn into task-specific')
    parser.add_argument('--branch_type', default='no_branch',
                        choices=['no_branch', 'branched'], type=str, help='The branch type of models')
    parser.add_argument('--sub_method', default='Baseline',
                        choices=['Baseline', 'CAGrad'], type=str,
                        help='The gradient-based methods used in Recon')
    parser.add_argument('--conflict_scores_file', default='', type=str, help='The path of the file that records ranking of layers')

def cagrad_args(parser):
    parser.add_argument('--alpha', default=0.2, type=float, help='the alpha used in CAGrad')

def add_args(parser):
    project_args(parser)
    training_args(parser)
    task_args(parser)
    model_args(parser)
    branch_args(parser)
    cagrad_args(parser)

    opt = parser.parse_args()
    return opt

def create_train_val_dataloader(opt, logger):
    with open(opt.dataroot, 'rb') as f:
        trainX, trainLabel, testX, testLabel = pickle.load(f)

    trainX = torch.from_numpy(trainX.reshape(120000,1,36,36)).float().cuda()
    trainLabel = torch.from_numpy(trainLabel).long().cuda()
    testX = torch.from_numpy(testX.reshape(20000,1,36,36)).float().cuda()
    testLabel = torch.from_numpy(testLabel).long().cuda()

    train_set = torch.utils.data.TensorDataset(trainX, trainLabel)

    test_set  = torch.utils.data.TensorDataset(testX, testLabel)

    train_loader = torch.utils.data.DataLoader(
                     dataset=train_set,
                     batch_size=opt.bz,
                     shuffle=True,
                     drop_last=True)

    test_loader = torch.utils.data.DataLoader(
                    dataset=test_set,
                    batch_size=1,
                    shuffle=False)

    num_iter_per_epcoh = math.floor(len(train_set) / opt.bz)
    total_iters = opt.n_epoch * num_iter_per_epcoh
    total_epochs = opt.n_epoch

    logger.info('Training statistics:'
                f'\n\tNumber of train images: {len(train_set)}'
                f'\n\tBatch size per gpu: {opt.bz}'
                f'\n\tRequire iter number per epoch: {num_iter_per_epcoh}'
                f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.')

    info = {'num_iter_per_epcoh': num_iter_per_epcoh,
            'total_iters': total_iters,
            'total_epochs': total_epochs}

    return train_loader, test_loader, info

def task_name_to_loss_map():
    task_name_to_loss_map = {'T1': nn.CrossEntropyLoss(),
                             'T2': nn.CrossEntropyLoss()}
    return task_name_to_loss_map

def task_name_to_metric_map(mode='train'):
    task_name_to_metric_map = {'T1': [classify_acc_metric(task='T1', mode=mode),
                                      losses_metric(task='T1', mode=mode)],
                               'T2': [classify_acc_metric(task='T2', mode=mode),
                                      losses_metric(task='T2', mode=mode)]}

    return task_name_to_metric_map

def reset_metric_map(metric_map, tasks_list):
    for task in tasks_list:
        for metric in metric_map[task]:
            metric.reset()

def avg_metric(metric_map, tasks_list):
    output_log = {}
    for task in tasks_list:
        for metric in metric_map[task]:
            output_log.update(metric.avg())

    return output_log


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MultiFashion+MNIST')
    opt = add_args(parser)

    # set random seed
    set_random_seed(opt.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # make experiment and log dirs
    exp_path, log_path, tb_log_path = make_exp_and_log_dirs(opt)
    log_file = osp.join(log_path, f"log.log")
    logger = get_root_logger(logger_name='recon', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    # initialize wandb logger
    init_wandb_logger(opt)
    # initialize tensorboard logger
    tb_logger = init_tb_logger(log_dir=osp.join(tb_log_path, f"tb_log.log"))

    # create dataset
    train_loader, test_loader, info = create_train_val_dataloader(opt, logger)
    # do not support resume training
    start_epoch = 0
    current_iter = 0
    total_iters = info['total_iters']

    # create message logger (formatted outputs)
    msg_logger = MessageLogger(opt, total_iters=total_iters, tb_logger=tb_logger)

    # create model
    MODEL = get_model_object(opt.method)
    arch_opt = {'tasks': opt.tasks, 'branch_type': opt.branch_type,
                'topK': opt.topK, 'conflict_scores_file': opt.conflict_scores_file}
    model = MODEL(opt, arch_opt)

    logger.info(f'Created model: {model.__class__.__name__}, total parameters: {model.model_size():4f}MB')


    # create optimizer
    if opt.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
    elif opt.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
    else:
        raise NotImplementedError

    # create loss function
    loss_func = WeightedSumLoss(tasks_name_to_loss_map=task_name_to_loss_map(), tasks=opt.tasks)
    # create criterion function
    tr_metric_map = task_name_to_metric_map('train')
    te_metric_map = task_name_to_metric_map('test')

    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer()
    start_time = time.time()

    for epoch in range(start_epoch, opt.n_epoch):
        model.train()

        for k, data in enumerate(train_loader):
            data_timer.record()
            current_iter += 1

            x = data[0].cuda()
            ts = data[1].cuda()

            ts = {'T1': ts[:, 0], 'T2': ts[:, 1]}

            outputs, losses = model.train_loop(inputs=x, targets=ts, loss=loss_func, optimizer=optimizer)

            output_info = {'outputs': outputs, 'losses': losses, 'targets': ts}

            # update metric
            metric_log = {}
            for task in opt.tasks:
                for metric in tr_metric_map[task]:
                    log = metric(output_info)
                    metric_log.update(log)

            iter_timer.record()
            if current_iter == 1:
                # reset start time in msg_logger for more accurate eta_time
                # not work in resume mode
                msg_logger.reset_start_time()
            if current_iter % opt.print_freq == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                # log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()})
                # update
                log_vars.update(metric_log)
                msg_logger(log_vars)
                # upload log to wandb
                wandb.log(metric_log, step=current_iter)

        scheduler.step()

        if epoch % opt.eval_freq == 0 or epoch == opt.n_epoch - 1:
            model.eval()
            for k, data in enumerate(test_loader):
                x = data[0].cuda()
                ts = data[1].cuda()

                ts = {'T1': ts[:, 0], 'T2': ts[:, 1]}

                output, losses = model.eval_loop(inputs=x, targets=ts, loss=loss_func)

                output_info = {'outputs': output, 'losses': losses, 'targets': ts}

                # update metric
                for task in opt.tasks:
                    for metric in te_metric_map[task]:
                        log = metric(output_info)

            logger.info('Evaluation')
            log_vars = {'epoch': epoch, 'iter': current_iter}
            val_metric_log = avg_metric(te_metric_map, opt.tasks)
            log_vars.update(val_metric_log)
            msg_logger(log_vars)
            wandb.log(val_metric_log, step=current_iter)

        reset_metric_map(tr_metric_map, opt.tasks)
        reset_metric_map(te_metric_map, opt.tasks)

        # only save model or conflict scores after the last iteration
        if current_iter == total_iters:
            logger.info('Saving models and training states.')
            model.save(exp_path, opt.name, epoch, current_iter, opt.seed)

    # logger.info(f'Training finished. Total training time: {time.time() - start_time:.2f}s')