import argparse
import os
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn as nn
import yaml
from tqdm import tqdm
from torch import optim

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val as validate  # for end-of-epoch mAP
from src.data.data import create_simple_dataloader
from src.utils.general import (LOGGER, TQDM_BAR_FORMAT, init_seeds,
                           yaml_save, config_load, is_config, is_serialized)
from src.eval.loss import loss_fn
from src.utils.loggers import Logger
from src.models.simple_model import MLP


def train(opt):  # hyp is path/to/hyp.yaml or hyp dictionary
    data, end_epoch, noval, nosave, workers = (
        opt.data, opt.epochs, opt.noval, opt.nosave, opt.workers
    )

    hyp = config_load(opt.hyp)
    model = opt.model
    if is_config(model):
        model_conf = config_load(model)
        model = MLP(**model_conf)

    elif is_serialized(model):
        raise NotImplementedError

    # Configure
    model = model.to(opt.device)

    data = config_load(data)


    train_data, val_data = data["train"], data["val"]

    if is_config(train_data):
        train_loader = create_simple_dataloader(
            **train_data,
            num_workers=workers,
        )
    elif isinstance(train_data, DataLoader):
        train_loader = train_data


    if is_config(val_data):
        val_loader = create_simple_dataloader(
            **val_data,
            num_workers=workers,
        )
    elif isinstance(val_data, DataLoader):
        val_loader = val_data


    # Directories
    LOGGER.info('hyperparameters: ' + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # Loggers
    logger = Logger(
        "root",
        logger="wandb",
        opt=opt,
        root_logger = LOGGER,
        run_id="historical123"
    )

    # Process custom dataset artifact link

    init_seeds(opt.seed, deterministic=True)


    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Resume
    best_metric, start_epoch = -float("inf"), 0

    # Start training
    t0 = time.time()
    LOGGER.info(f'Using {train_loader.num_workers} dataloader workers\n'
                f"Logging results to {'bold'}\n"
                f'Starting training for {end_epoch} epochs...')
    for epoch in range(start_epoch, end_epoch):  # epoch ------------------------------------------------------------------
        # on_train_epoch_start
        model.train()
        loss = 0
        pbar = tqdm(train_loader, bar_format=TQDM_BAR_FORMAT)  # progress bar
        for i, (train_x, train_y) in enumerate(pbar):  # batch -------------------------------------------------------------
            # on_train_batch_start()
            optimizer.zero_grad()
            # Forward
            train_y_hat = model(train_x)  # forward
            loss = loss_fn(train_y_hat, train_y.type(torch.long))  # loss scaled by batch_size

            # Backward
            loss.backward()
            optimizer.step()

            pbar.set_description(f"{epoch}/{end_epoch-1}, loss={loss.item():.4f}")
            logger.log_metric({"train/loss": loss.item(), "train/epoch": epoch})
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers

        # on_train_end
        final_epoch = (epoch + 1 == end_epoch)
        if not noval or final_epoch:  # Calculate mAP
            eval_metric = validate.run(
                data=val_loader,
                model=model,
                compute_loss=loss_fn
            )
            eval_metric["val/epoch"] = epoch

            # Update best mAP
            if eval_metric["val/accuracy"] > best_metric:
                best_metric = eval_metric["val/accuracy"]
            log_vals = [loss] + lr
            # on_validation_end()
            logger.log_metric(eval_metric)

            # Save model
            if (not nosave) or final_epoch:  # if save
                ckpt = {
                    'epoch': epoch,
                    'best_metric': best_metric,
                    'model': deepcopy(model),
                    'optimizer': optimizer.state_dict(),
                    'opt': vars(opt),
                    'date': datetime.now().isoformat()
                }
                bin_dir = "bin"
                last, best = f"{bin_dir}/last.pt", f"{bin_dir}/best.pt"
                # Save last, best and delete
                torch.save(ckpt, last)
                if best_metric == eval_metric["val/accuracy"]:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    is_best = best_metric == eval_metric["val/accuracy"]
                    logger.log_model(bin_dir, opt, epoch, eval_metric["val/accuracy"], is_best)
                del ckpt

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
    # for f in last, best:
    #     if f.exists():
    #         if f is best:
    #             LOGGER.info(f'\nValidating {f}...')
    #             eval_metric = validate.run()  # val best model with plots

    #     logger.log_model(bin_dir, opt, epoch, eval_metric["val/accuracy"])

    logger.end_log()
    print("hi")

    return eval_metric


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="configs/model.yaml", help='initial weights path')
    parser.add_argument('--data', type=str, default="configs/data.yaml", help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default='configs/hyp.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default="pytorch-project", help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt,):

    # DDP mode
    opt.save_dir = "bin"

    # Train
    train(opt,)


def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)