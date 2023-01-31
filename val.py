import argparse
import os
import sys
from pathlib import Path

from torch import nn
from torch.utils.data import DataLoader

import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from src.data.data import create_simple_dataloader
from src.models.simple_model import MLP

from src.eval.metric import accuracy
from src.utils.utils import (LOGGER, config_load, is_config, is_serialized)

@torch.inference_mode()
def run(
    model: nn.Module | dict | str,
    data: DataLoader | dict | str,
    task='val',  # train, val, test, speed or study
    device="cpu",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    workers=8,  # max dataloader workers (per RANK in DDP mode)
    verbose=False,  # verbose output
    save_dir=Path(''),
    compute_loss=None,
):
    # Initialize/load model and set device
    if is_config(model):
        model_conf = config_load(model)
        model = MLP(**model_conf)

    elif is_serialized(model):
        raise NotImplementedError

    # Configure
    model = model.to(device)
    model.eval()

    if is_config(data):
        data_conf = config_load(data)
        dataloader = create_simple_dataloader(
            **data_conf,
            num_workers=workers,
        )
    elif isinstance(data, DataLoader):
        dataloader = data
    
    # evaluation metrics to compute
    eval_dict = {
        "val/loss": 0,
        "val/accuracy": 0,
    }

    LOGGER.info(f"{task} started")
    for batch_i, (x, y) in enumerate(dataloader):
        batch_weight = x.shape[0] / dataloader.batch_size

        # start of validation step
        y_hat = model(x)

        # update metrics
        eval_dict["val/accuracy"] += accuracy(y_hat, y).item() * batch_weight
        if compute_loss:
            eval_dict["val/loss"] += compute_loss(y_hat, y).item() * batch_weight

    # turn sum -> mean (over batch)
    eval_dict["val/loss"] = eval_dict["val/loss"] / len(dataloader)
    eval_dict["val/accuracy"] = eval_dict["val/accuracy"] / len(dataloader)
    LOGGER.info(eval_dict)

    return eval_dict


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="configs/model.yaml", help='model path(s)')
    parser.add_argument('--data', type=str, default="configs/val.yaml", help='model path(s)')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default="cpu", help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    opt = parser.parse_args()
    LOGGER.info(f"Running validation script: {vars(opt)}")
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)