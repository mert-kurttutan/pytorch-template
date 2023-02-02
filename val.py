import argparse

from torch import nn
from torch.utils.data import DataLoader

import torch

from src.data.data import get_cifar10_dataloader
from src.models.base_cnn import BaseCNN

from src.eval.metric import accuracy
from src.utils.utils import (LOGGER, config_load, is_config, is_serialized)

@torch.inference_mode()
def run(
    model: nn.Module | dict | str,
    data: DataLoader | dict | str,
    data_name='val',  # train, val, test, speed or study
    device="cpu",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    workers=8,  # max dataloader workers (per RANK in DDP mode)
    verbose=False,  # verbose output
    compute_loss=None,
):

    device = "cuda" if device != "cpu" else device
    # Initialize/load model and set device
    if is_config(model):
        model_conf = config_load(model)
        model = BaseCNN(model_conf)

    elif is_serialized(model):
        raise NotImplementedError

    # Configure
    model = model.to(device)
    model.eval()

    if is_config(data):
        data_conf = config_load(data)
        dataloader = get_cifar10_dataloader(
            **data_conf,
            num_workers=workers,
        )
    elif isinstance(data, DataLoader):
        dataloader = data
    
    # evaluation metrics to compute
    eval_dict = {
        f"val/{data_name}_loss": 0,
        f"val/{data_name}_acc": 0,
    }

    LOGGER.info(f"Validation {data_name} started")
    for x, y in dataloader:
        batch_weight = x.shape[0] / dataloader.batch_size
        x, y = x.to(device), y.to(device)

        # start of validation step
        y_hat = model(x)

        # update metrics
        eval_dict[f"val/{data_name}_acc"] += accuracy(y_hat, y).item() * batch_weight
        if compute_loss:
            eval_dict[f"val/{data_name}_loss"] += compute_loss(y_hat, y).item() * batch_weight

    # turn sum -> mean (over batch)
    eval_dict[f"val/{data_name}_loss"] /= len(dataloader)
    eval_dict[f"val/{data_name}_acc"] /= len(dataloader)
    if not compute_loss:
        del eval_dict[f"val/{data_name}_loss"]
    LOGGER.info(eval_dict)

    return eval_dict


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="configs/model.yaml", help='model path(s)')
    parser.add_argument('--data', type=str, default="configs/val.yaml", help='model path(s)')
    parser.add_argument('--data-name', default='val', help='train, val, test, speed or study')
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