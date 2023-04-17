import numpy as np
import os
import random
import time
import torch
from os import path as osp
import torch.nn.functional as F

def cos(t1, t2):
    t1 = F.normalize(t1, dim=0)
    t2 = F.normalize(t2, dim=0)

    dot = (t1 * t2).sum(dim=0)

    return dot

def pair_cos(pair):
    length = pair.size(0)

    dot_value = []
    for i in range(length - 1):
        for j in range(i + 1, length):
            dot_value.append(cos(pair[i], pair[j]))

    dot_value = torch.stack(dot_value).view(-1)
    return dot_value

def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def merge_dict(dict):
    """Merge a dict of dict into a single dict.

    Args: dict. {Outputs: {T1: O1, T2: O2}, Losses: {T1: L1, T2: L2}}
    Returns: new_dict. {T1: {Outputs: O1, Losses: L1}, T2: {Outputs: O2, Losses: L2}}
    """
    new_dict = {}
    for key, value in dict.items():
        for k, v in value.items():
            if k not in new_dict:
                new_dict[k] = {}
            new_dict[k][key] = v
            
    return new_dict

def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())

def make_exp_and_log_dirs(opt):
    """Make dirs for experiments."""
    path = osp.join(opt.path, f'{opt.project}_{opt.name}')

    mkdir_and_rename(path)

    # log file
    log_path = osp.join(path, 'logger')
    mkdir_and_rename(log_path)

    # tensorboard file
    tb_log_path = osp.join(path, 'tb_logger')
    mkdir_and_rename(tb_log_path)

    return path, log_path, tb_log_path


def mkdir_and_rename(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one.
    Args:
        path (str): Folder path.
    """
    if osp.exists(path):
        new_name = path + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)

def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.
    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.
    Returns:
        A generator for all the interested files with relative paths.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)