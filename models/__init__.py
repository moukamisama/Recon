import importlib
from copy import deepcopy
from os import path as osp

from utils import get_root_logger, scandir

from utils.registry import MODEL_REGISTRY

__all__ = ['get_model_object']

model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(model_folder) if v.endswith('_model.py')]
# import all the model modules
_model_modules = [importlib.import_module(f'models.{file_name}') for file_name in model_filenames]

def get_model_object(object_name):
    model = MODEL_REGISTRY.get(object_name)
    return model