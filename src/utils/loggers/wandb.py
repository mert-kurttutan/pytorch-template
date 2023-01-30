"""Utilities and tools for tracking runs with Weights & Biases."""

import logging
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict

import yaml
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[3]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from ..general import LOGGER

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

    def __init__(self, opt, run_id=None, job_type='Training'):
        """
        - Initialize WandbLogger instance
        - Upload dataset if opt.upload_dataset is True
        - Setup training processes if job_type is 'Training'

        arguments:
        opt (namespace) -- Commandline arguments for this run
        run_id (str) -- Run ID of W&B run to be resumed
        job_type (str) -- To set the job_type for this run

       """

        # Pre-training routine --
        self.job_type = job_type
        self.wandb, self.wandb_run = wandb, None if not wandb else wandb.run
        self.val_artifact, self.train_artifact = None, None
        self.train_artifact_path, self.val_artifact_path = None, None
        # It's more elegant to stick to 1 wandb.init call,
        #  but useful config data is overwritten in the WandbLogger's wandb.init call
        if isinstance(opt.resume, str):  # checks resume from artifact
            if opt.resume.startswith(WANDB_ARTIFACT_PREFIX):
                entity, project, run_id, model_artifact_name = get_run_info(opt.resume)
                model_artifact_name = WANDB_ARTIFACT_PREFIX + model_artifact_name
                assert wandb, 'install wandb to resume wandb runs'
                # Resume wandb-artifact:// runs here| workaround for not overwriting wandb.config
                self.wandb_run = wandb.init(id=run_id,
                                            project=project,
                                            entity=entity,
                                            resume='allow',
                                            allow_val_change=True)
                opt.resume = model_artifact_name
        elif self.wandb:
            self.wandb_run = wandb.init(config=opt,
                                        resume="allow",
                                        project='YOLOv5' if opt.project == 'runs/train' else Path(opt.project).stem,
                                        entity=opt.entity,
                                        name=opt.name if opt.name != 'exp' else None,
                                        job_type=job_type,
                                        id=run_id,
                                        allow_val_change=True) if not wandb.run else wandb.run
        if self.wandb_run:
            if self.job_type == 'Training':
                if opt.upload_dataset:
                    if not opt.resume:
                        self.wandb_artifact_data_dict = self.check_and_upload_dataset(opt)

                if isinstance(opt.data, dict):
                    # This means another dataset manager has already processed the dataset info (e.g. ClearML)
                    # and they will have stored the already processed dict in opt.data
                    self.data_dict = opt.data
                elif opt.resume:
                    # resume from artifact
                    if isinstance(opt.resume, str) and opt.resume.startswith(WANDB_ARTIFACT_PREFIX):
                        self.data_dict = dict(self.wandb_run.config.data_dict)
                    else:  # local resume
                        self.data_dict = check_wandb_dataset(opt.data)
                else:
                    self.data_dict = check_wandb_dataset(opt.data)
                    self.wandb_artifact_data_dict = self.wandb_artifact_data_dict or self.data_dict

                    # write data_dict to config. useful for resuming from artifacts. Do this only when not resuming.
                    self.wandb_run.config.update({'data_dict': self.wandb_artifact_data_dict}, allow_val_change=True)
                self.setup_training(opt)

            if self.job_type == 'Dataset Creation':
                self.wandb_run.config.update({"upload_dataset": True})
                self.data_dict = self.check_and_upload_dataset(opt)

    def check_and_upload_dataset(self, opt):
        """
        Check if the dataset format is compatible and upload it as W&B artifact

        arguments:
        opt (namespace)-- Commandline arguments for current run

        returns:
        Updated dataset info dictionary where local dataset paths are replaced by WAND_ARFACT_PREFIX links.
        """
        assert wandb, 'Install wandb to upload dataset'
        config_path = self.log_dataset_artifact(opt.data, opt.single_cls,
                                                'YOLOv5' if opt.project == 'runs/train' else Path(opt.project).stem)
        with open(config_path, errors='ignore') as f:
            wandb_data_dict = yaml.safe_load(f)
        return wandb_data_dict

    def setup_training(self, opt):
        """
        Setup the necessary processes for training YOLO models:
          - Attempt to download model checkpoint and dataset artifacts if opt.resume stats with WANDB_ARTIFACT_PREFIX
          - Update data_dict, to contain info of previous run if resumed and the paths of dataset artifact if downloaded
          - Setup log_dict, initialize bbox_interval

        arguments:
        opt (namespace) -- commandline arguments for this run

        """
        self.log_dict, self.current_epoch = {}, 0
        self.bbox_interval = opt.bbox_interval
        if isinstance(opt.resume, str):
            modeldir, _ = self.download_model_artifact(opt)
            if modeldir:
                self.weights = Path(modeldir) / "last.pt"
                config = self.wandb_run.config
                opt.weights, opt.save_period, opt.batch_size, opt.bbox_interval, opt.epochs, opt.hyp, opt.imgsz = str(
                    self.weights), config.save_period, config.batch_size, config.bbox_interval, config.epochs,\
                    config.hyp, config.imgsz
        data_dict = self.data_dict
        if self.val_artifact is None:  # If --upload_dataset is set, use the existing artifact, don't download
            self.train_artifact_path, self.train_artifact = self.download_dataset_artifact(
                data_dict.get('train'), opt.artifact_alias)
            self.val_artifact_path, self.val_artifact = self.download_dataset_artifact(
                data_dict.get('val'), opt.artifact_alias)

        if self.train_artifact_path is not None:
            train_path = Path(self.train_artifact_path) / 'data/images/'
            data_dict['train'] = str(train_path)
        if self.val_artifact_path is not None:
            val_path = Path(self.val_artifact_path) / 'data/images/'
            data_dict['val'] = str(val_path)

        if self.val_artifact is not None:
            self.result_artifact = wandb.Artifact("run_" + wandb.run.id + "_progress", "evaluation")
            columns = ["epoch", "id", "ground truth", "prediction"]
            columns.extend(self.data_dict['names'])
            self.result_table = wandb.Table(columns)
            self.val_table = self.val_artifact.get("val")
            if self.val_table_path_map is None:
                self.map_val_table_path()
        if opt.bbox_interval == -1:
            self.bbox_interval = opt.bbox_interval = (opt.epochs // 10) if opt.epochs > 10 else 1
            if opt.evolve or opt.noplots:
                self.bbox_interval = opt.bbox_interval = opt.epochs + 1  # disable bbox_interval
        train_from_artifact = self.train_artifact_path is not None and self.val_artifact_path is not None
        # Update the the data_dict to point to local artifacts dir
        if train_from_artifact:
            self.data_dict = data_dict


    def map_val_table_path(self):
        """
        Map the validation dataset Table like name of file -> it's id in the W&B Table.
        Useful for - referencing artifacts for evaluation.
        """
        raise NotImplementedError


    def log_training_progress(self, predn, path, names):
        """
        Build evaluation Table. Uses reference from validation dataset table.

        arguments:
        predn (list): list of predictions in the native space in the format - [xmin, ymin, xmax, ymax, confidence, class]
        path (str): local path of the current evaluation image
        names (dict(int, str)): hash map that maps class ids to labels
        """
        class_set = wandb.Classes([{'id': id, 'name': name} for id, name in names.items()])
        box_data = []
        avg_conf_per_class = [0] * len(self.data_dict['names'])
        pred_class_count = {}
        for *xyxy, conf, cls in predn.tolist():
            if conf >= 0.25:
                cls = int(cls)
                box_data.append({
                    "position": {
                        "minX": xyxy[0],
                        "minY": xyxy[1],
                        "maxX": xyxy[2],
                        "maxY": xyxy[3]},
                    "class_id": cls,
                    "box_caption": f"{names[cls]} {conf:.3f}",
                    "scores": {
                        "class_score": conf},
                    "domain": "pixel"})
                avg_conf_per_class[cls] += conf

                if cls in pred_class_count:
                    pred_class_count[cls] += 1
                else:
                    pred_class_count[cls] = 1

        for pred_class in pred_class_count.keys():
            avg_conf_per_class[pred_class] = avg_conf_per_class[pred_class] / pred_class_count[pred_class]

        boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
        id = self.val_table_path_map[Path(path).name]
        self.result_table.add_data(self.current_epoch, id, self.val_table.data[id][1],
                                   wandb.Image(self.val_table.data[id][1], boxes=boxes, classes=class_set),
                                   *avg_conf_per_class)

    def val_one_image(self, pred, predn, path, names, im):
        """
        Log validation data for one image. updates the result Table if validation dataset is uploaded and log bbox media panel

        arguments:
        pred (list): list of scaled predictions in the format - [xmin, ymin, xmax, ymax, confidence, class]
        predn (list): list of predictions in the native space - [xmin, ymin, xmax, ymax, confidence, class]
        path (str): local path of the current evaluation image
        """
        if self.val_table and self.result_table:  # Log Table if Val dataset is uploaded as artifact
            self.log_training_progress(predn, path, names)

        if len(self.bbox_media_panel_images) < self.max_imgs_to_log and self.current_epoch > 0:
            if self.current_epoch % self.bbox_interval == 0:
                box_data = [{
                    "position": {
                        "minX": xyxy[0],
                        "minY": xyxy[1],
                        "maxX": xyxy[2],
                        "maxY": xyxy[3]},
                    "class_id": int(cls),
                    "box_caption": f"{names[int(cls)]} {conf:.3f}",
                    "scores": {
                        "class_score": conf},
                    "domain": "pixel"} for *xyxy, conf, cls in pred.tolist()]
                boxes = {"predictions": {"box_data": box_data, "class_labels": names}}  # inference-space
                self.bbox_media_panel_images.append(wandb.Image(im, boxes=boxes, caption=path.name))

    def log(self, log_dict):
        """
        save the metrics to the logging dictionary

        arguments:
        log_dict (Dict) -- metrics/media to be logged in current step
        """
        if self.wandb_run:
            for key, value in log_dict.items():
                self.log_dict[key] = value

    def end_epoch(self, best_result=False):
        """
        commit the log_dict, model artifacts and Tables to W&B and flush the log_dict.

        arguments:
        best_result (boolean): Boolean representing if the result of this evaluation is best or not
        """
        if self.wandb_run:
            with all_logging_disabled():
                if self.bbox_media_panel_images:
                    self.log_dict["BoundingBoxDebugger"] = self.bbox_media_panel_images
                try:
                    wandb.log(self.log_dict)
                except BaseException as e:
                    LOGGER.info(
                        f"An error occurred in wandb logger. The training will proceed without interruption. More info\n{e}"
                    )
                    self.wandb_run.finish()
                    self.wandb_run = None

                self.log_dict = {}
                self.bbox_media_panel_images = []
            if self.result_artifact:
                self.result_artifact.add(self.result_table, 'result')
                wandb.log_artifact(self.result_artifact,
                                   aliases=[
                                       'latest', 'last', 'epoch ' + str(self.current_epoch),
                                       ('best' if best_result else '')])

                wandb.log({"evaluation": self.result_table})
                columns = ["epoch", "id", "ground truth", "prediction"]
                columns.extend(self.data_dict['names'])
                self.result_table = wandb.Table(columns)
                self.result_artifact = wandb.Artifact("run_" + wandb.run.id + "_progress", "evaluation")

    def finish_run(self):
        """
        Log metrics if any and finish the current W&B run
        """
        if self.wandb_run:
            if self.log_dict:
                with all_logging_disabled():
                    wandb.log(self.log_dict)
            wandb.run.finish()


@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    """ source - https://gist.github.com/simon-weber/7853144
    A context manager that will prevent any logging messages triggered during the body from being processed.
    :param highest_level: the maximum logging level in use.
      This would only need to be changed if a custom level greater than CRITICAL is defined.
    """
    previous_level = logging.root.manager.disable
    logging.disable(highest_level)
    try:
        yield
    finally:
        logging.disable(previous_level)