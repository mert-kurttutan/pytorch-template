"""
Logging utils
"""

import os
import pkg_resources as pkg

from ..utils import LOGGER
from .wandb import WandbLogger

LOGGERS = ('wandb',)
RANK = int(os.getenv('RANK', -1))

try:
    import wandb

    assert hasattr(wandb, '__version__')  # verify package import not local dir
    if pkg.parse_version(wandb.__version__) >= pkg.parse_version('0.12.2') and RANK in {0, -1}:
        try:
            wandb_login_success = wandb.login(timeout=30)
        except wandb.errors.UsageError:  # known non-TTY terminal issue
            wandb_login_success = False
        if not wandb_login_success:
            wandb = None
except (ImportError, AssertionError):
    wandb = None


class Logger():
    # YOLOv5 Loggers class
    def __init__(self, logger="wandb", opt=None, run_config=None, root_logger=None):
        self.opt = opt
        self.logger_type = logger
        self.root_logger = root_logger  # for printing results to console
        if run_config is None:
            run_config = {}

        # W&B
        if logger == "wandb":
            self.logger = WandbLogger(opt, run_config)
        else:
            raise NotImplementedError

    def log_metric(self, metric):
        if self.logger_type == "wandb":
            self.logger.log_metric(metric)
        else:
            raise NotImplementedError

    def log_model(self, file_name, opt, epoch, fitness_score, best_model=False):
        """
        Log the model checkpoint as W&B artifact
        arguments:
        opt (namespace) -- Command line arguments for this run
        epoch (int)  -- Current epoch number
        fitness_score (float) -- fitness score for current epoch
        best_model (boolean) -- Boolean representing if the current checkpoint is the best yet.
        """
        if self.logger_type == "wandb":
            self.logger.log_model(file_name, opt, epoch, fitness_score, best_model)
        else:
            raise NotImplementedError
        LOGGER.info(f"Saving model artifact on epoch {epoch + 1}")

    def end_log(self,):
        self.logger.run.finish()
