"""Utilities and tools for tracking runs with Weights & Biases."""

import logging
import os
from contextlib import contextmanager
from pathlib import Path


try:
    import wandb

    assert hasattr(wandb, '__version__')  # verify package import not local dir
except (ImportError, AssertionError):
    wandb = None

RANK = int(os.getenv('RANK', -1))
WANDB_ARTIFACT_PREFIX = 'wandb-artifact://'


def remove_prefix(from_string, prefix=WANDB_ARTIFACT_PREFIX):
    return from_string[len(prefix):]


def check_wandb_config_file(data_config_file):
    wandb_config = '_wandb.'.join(data_config_file.rsplit('.', 1))  # updated data.yaml path
    if Path(wandb_config).is_file():
        return wandb_config
    return data_config_file


def get_run_info(run_path):
    run_path = Path(remove_prefix(run_path, WANDB_ARTIFACT_PREFIX))
    run_id = run_path.stem
    project = run_path.parent.stem
    entity = run_path.parent.parent.stem
    model_artifact_name = 'run_' + run_id + '_model'
    return entity, project, run_id, model_artifact_name


class WandbLogger():
    """Log training runs, datasets, models, and predictions to Weights & Biases.

    This logger sends information to W&B at wandb.ai. By default, this information
    includes hyperparameters, system configuration and metrics, model metrics,
    and basic data metrics and analyses.

    By providing additional command line arguments to train.py, datasets,
    models and predictions can also be logged.

    For more on how this logger is used, see the Weights & Biases documentation:
    https://docs.wandb.com/guides/integrations/yolov5
    """

    def __init__(self, opt, run_config, job_type="Training"):

        self.run = wandb.init(
            # Set the project where this run will be logged
            project=opt.project,
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=opt.run_name,
            # Track hyperparameters and run metadata
            config=run_config,
            id=opt.run_id,
            resume="allow",
            job_type=job_type,
        )

        opt.run_name = self.run.name
        opt.run_id = self.run.id

    def log_metric(self, metric):
        self.run.log(metric)

    def log_model(self, file_name, opt, epoch, fitness_score, best_model=False):
        model_artifact = wandb.Artifact(
            f"run_{self.run.id}_model",
            type='model',
            metadata={
                'original_dir': str(opt.bin_dir),
                'epochs_trained': epoch + 1,
                'save period': opt.save_period,
                'project': opt.project,
                'total_epochs': opt.epochs,
                'fitness_score': fitness_score
            }
        )
        model_artifact.add_file(f"{opt.bin_dir}/{file_name}", name=file_name)
        self.run.log_artifact(
            model_artifact,
            aliases=[f"epoch_{epoch}", 'best' if best_model else '']
        )

    def val_one_image(self, pred, path, im):
        """
        Log validation data for one image.
        Updates the result Table if validation dataset is uploaded

        arguments:
        pred (list): class prediction in the format - class
        path (str): local path of the current evaluation image
        """
        # TODO: fill this later
        pass


@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    """ source - https://gist.github.com/simon-weber/7853144
    A context manager that will prevent any logging messages triggered
    during the body from being processed.
    :param highest_level: the maximum logging level in use.
      This would only need to be changed if a custom level greater than CRITICAL is defined.
    """
    previous_level = logging.root.manager.disable
    logging.disable(highest_level)
    try:
        yield
    finally:
        logging.disable(previous_level)
