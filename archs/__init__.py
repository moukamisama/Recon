import importlib
from copy import deepcopy
from os import path as osp

from utils.registry import ARCH_REGISTRY
from utils import scandir

__all__ = ['get_arch_object']

# automatically scan and import arch modules for registry
# scan all the files under the 'archs' folder and collect files ending with '_arch.py'
arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(arch_folder) if v.endswith('_arch.py')]
# import all the arch modules
_arch_modules = [importlib.import_module(f'archs.{file_name}') for file_name in arch_filenames]

def get_arch_object(object_name):
    arch = ARCH_REGISTRY.get(object_name)
    return arch
