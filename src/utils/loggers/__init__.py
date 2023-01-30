"""
Logging utils
"""

import os
import warnings
from pathlib import Path

import pkg_resources as pkg
import torch

from ..general import LOGGER
from .wandb import WandbLogger

LOGGERS = ( 'wandb', )
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
    def __init__(self, save_dir=None, logger="wandb", run_name = "run", run_id=None, opt=None, root_logger=None):
        self.save_dir = save_dir
        self.opt = opt
        self.logger_type = logger
        self.root_logger = root_logger  # for printing results to console
        self.run_id = run_id

        # W&B
        if logger == "wandb":
            wandb.init(
                # Set the project where this run will be logged
                project=opt.project, 
                # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
                name=run_name, 
                # Track hyperparameters and run metadata
                config={
                    "learning_rate": 0.02,
                    "architecture": "CNN",
                    "dataset": "CIFAR-100",
                    "epochs": 10,
                },
                id=self.run_id
            )
            self.run_id = wandb.run.id
        else:
            raise NotImplementedError


    def log_metric(self, metric):
        if self.logger_type == "wandb":
            wandb.log(metric)
        else:
            raise NotImplementedError


    def log_model(self, path, opt, epoch, fitness_score, best_model=False):
        """
        Log the model checkpoint as W&B artifact
        arguments:
        path (Path)   -- Path of directory containing the checkpoints
        opt (namespace) -- Command line arguments for this run
        epoch (int)  -- Current epoch number
        fitness_score (float) -- fitness score for current epoch
        best_model (boolean) -- Boolean representing if the current checkpoint is the best yet.
        """
        model_artifact = wandb.Artifact('run_' + wandb.run.id + '_model',
                                        type='model',
                                        metadata={
                                            'original_url': str(path),
                                            'epochs_trained': epoch + 1,
                                            'save period': opt.save_period,
                                            'project': opt.project,
                                            'total_epochs': opt.epochs,
                                            'fitness_score': fitness_score})
        model_artifact.add_file(f"{path}/last.pt", name='last.pt')
        wandb.log_artifact(model_artifact,
                           aliases=['latest', 'last', 'epoch ' + str(epoch), 'best' if best_model else ''])
        LOGGER.info(f"Saving model artifact on epoch {epoch + 1}")

    def end_log(self,):
        wandb.finish()
