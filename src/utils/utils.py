"""
General utils
"""

import logging
import logging.config
import os
import random
import pkg_resources as pkg
import json
import urllib
from pathlib import Path

import numpy as np
import torch
import yaml


RANK = int(os.getenv('RANK', -1))

# Settings
VERBOSE = str(os.getenv('YOLOv5_VERBOSE', True)).lower() == 'true'  # global verbose mode
TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'  # tqdm bar format
FONT = 'Arial.ttf'

torch.set_printoptions(linewidth=320, precision=5, profile='long')


def is_ascii(s=''):
    # Is string composed of all ASCII (no UTF) characters?
    # (note str().isascii() introduced in python 3.7)
    s = str(s)  # convert list, tuple, None, etc. to str
    return len(s.encode().decode('ascii', 'ignore')) == len(s)


def yaml_load(file='data.yaml'):
    # Single-line safe yaml loading
    with open(file, errors='ignore') as f:
        return yaml.safe_load(f)


def json_load(file='data.json'):
    # Single-line safe yaml loading
    with open(file, errors='ignore') as f:
        return json.load(f)


def config_load(file_name: str | dict):
    if isinstance(file_name, dict):
        return file_name
    if file_name.endswith(".json"):
        return json_load(file_name)
    elif file_name.endswith((".yaml", ".yml")):
        return yaml_load(file_name)


def is_config(file_name: str | dict):
    if isinstance(file_name, dict):
        return True
    if not isinstance(file_name, str):
        return False
    config_types = (".yml", ".yaml", ".json")
    return file_name.endswith(config_types)


def is_serialized(file_name: str):
    if not isinstance(file_name, str):
        return False
    return file_name.endswith(".pt")


def is_writeable(dir, test=False):
    # Return True if directory has write permissions,
    # test opening a file with write permissions if test=True
    if not test:
        return os.access(dir, os.W_OK)  # possible issues on Windows
    file = Path(dir) / 'tmp.txt'
    try:
        with open(file, 'w'):  # open file with write permissions
            pass
        file.unlink()  # remove file
        return True
    except OSError:
        return False


LOGGING_NAME = "yolov5"


def set_logging(name=LOGGING_NAME, verbose=True):
    # sets up logging for the given name
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            name: {
                "format": "%(message)s"}},
        "handlers": {
            name: {
                "class": "logging.StreamHandler",
                "formatter": name,
                "level": level}},
        "loggers": {
            name: {
                "level": level,
                "handlers": [name],
                "propagate": False,
            }
        }
    })


set_logging(LOGGING_NAME)  # run before defining LOGGER
LOGGER = logging.getLogger(LOGGING_NAME)  # define globally (used by ModelRunner)


def init_seeds(seed=42, deterministic=False):

    # see https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe
    # AutoBatch problem https://github.com/ultralytics/yolov5/issues/9287
    # torch.backends.cudnn.benchmark = True
    # https://github.com/ultralytics/yolov5/pull/8213
    if deterministic and check_version(torch.__version__, '1.12.0'):
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['PYTHONHASHSEED'] = str(seed)


def yaml_save(file='data.yaml', data={}):
    # Single-line safe yaml saving
    with open(file, 'w') as f:
        yaml.safe_dump(
            {k: str(v) if isinstance(v, Path) else v for k, v in data.items()},
            f, sort_keys=False
        )


def check_version(
    current='0.0.0', minimum='0.0.0', name='version ',
    pinned=False, hard=False, verbose=False
):
    # Check version vs. required version
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    s = (
        f"WARNING {name}{minimum} is required by project, "
        "but {name}{current} is currently installed"
    )
    if hard:
        assert result, s  # assert min requirements met
    if verbose and not result:
        LOGGER.warning(s)
    return result


def url2file(url):
    # Convert URL to filename, i.e. https://url.com/file.txt?auth -> file.txt
    url = str(Path(url)).replace(':/', '://')  # Pathlib turns :// -> :/
    return Path(urllib.parse.unquote(url)).name.split('?')[0]
