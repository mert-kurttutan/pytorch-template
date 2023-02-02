import argparse
import time
from copy import deepcopy
from datetime import datetime

import torch
import wandb
from tqdm import tqdm


import val as validate
from src.models import get_model
from src.data.data import get_cifar10_dataloader
from src.utils.utils import (
    LOGGER, TQDM_BAR_FORMAT, init_seeds,
    config_load, is_config, is_serialized
)
from src.eval.loss import loss_fn
from src.utils.loggers import Logger
from src.optimizers import get_optimizer


# TODO: Use run_id to resume training

def train(opt):

    init_seeds(opt.seed, deterministic=True)
    device = "cuda" if opt.device != "cpu" else opt.device
    data, end_epoch, no_val, no_save, workers, model = (
        opt.data, opt.epochs, opt.no_val, opt.no_save, opt.workers, opt.model
    )

    if is_config(model):
        model_conf = config_load(model)
        model = get_model(model_conf)
    elif is_serialized(model):
        raise NotImplementedError

    # Configure
    model = model.to(device)

    data = config_load(data)

    train_data, val_data = data["train"], data["val"]
    assert is_config(train_data)

    train_loader = get_cifar10_dataloader(
        **train_data,
        **data["mode"]["train"],
        num_workers=workers,
    )

    val_loader = get_cifar10_dataloader(
        **val_data,
        **data["mode"]["val"],
        num_workers=workers,
    )

    train_loader_eval = get_cifar10_dataloader(
        **train_data,
        **data["mode"]["val"],
        num_workers=workers,
    )

    # Loggers
    logger = Logger(
        logger="wandb",
        opt=opt,
        root_logger = LOGGER,
    )

    # Optimizer
    optimizer = get_optimizer(model, opt.optimizer)

    # Resume
    best_metric, start_epoch = -float("inf"), 0

    # Start training
    t0 = time.time()
    LOGGER.info(
        f'Using {train_loader.num_workers} dataloader workers\n'
        f"Logging results to {'bold'}\n"
        f'Starting training for {end_epoch} epochs...'
    )
    n_steps_per_epoch = len(train_loader)
    for epoch in range(start_epoch, end_epoch):  # epoch ------------------------------------------------------------------
        # on_train_epoch_start
        model.train()
        loss = 0
        pbar = tqdm(train_loader, bar_format=TQDM_BAR_FORMAT)  # progress bar
        metric_dict = {}
        for i, (train_x, train_y) in enumerate(pbar):  # batch -------------------------------------------------------------
            # on_train_batch_start()
            optimizer.zero_grad()
            # Forward
            train_x, train_y = train_x.to(device), train_y.to(device)
            train_y_hat = model(train_x)  # forward
            loss = loss_fn(train_y_hat, train_y.type(torch.long))  # loss scaled by batch_size

            # Backward
            loss.backward()
            optimizer.step()

            metric_dict = {"train/loss": loss.item(), "train/epoch": epoch}
            if i+1 < n_steps_per_epoch:
                logger.log_metric(metric_dict)

            pbar.set_description(f"{epoch}/{end_epoch-1}, loss={loss.item():.4f}")
            # end batch ------------------------------------------------------------------------------------------------

        lr = [x['lr'] for x in optimizer.param_groups]
        metric_dict = {**metric_dict, "lr": wandb.Histogram(lr)}
        # on_train_end
        final_epoch = (epoch + 1 == end_epoch)
        if (not no_val or final_epoch):  # Calculate mAP
            eval_metric = validate.run(
                data=val_loader,
                model=model,
                compute_loss=loss_fn,
                device=opt.device,
                workers=workers,
            )

            train_metric = validate.run(
                data=train_loader_eval,
                model=model,
                device=opt.device,
                data_name="train",
                workers=workers,
            )

            metric_dict = {**metric_dict, **eval_metric, **train_metric}

            # Update best mAP
            if eval_metric["val/val_acc"] > best_metric:
                best_metric = eval_metric["val/val_acc"]

            # Save model
            if not no_save or final_epoch:
                ckpt = {
                    'epoch': epoch,
                    'best_metric': best_metric,
                    'model': deepcopy(model),
                    'optimizer': optimizer.state_dict(),
                    'opt': vars(opt),
                    'date': datetime.now().isoformat()
                }
                last, best = f"{opt.bin_dir}/last.pt", f"{opt.bin_dir}/best.pt"
                is_best = best_metric == eval_metric["val/val_acc"]
                # Save last, best and delete
                torch.save(ckpt, last)
                if is_best:
                    torch.save(ckpt, best)
                if (epoch+1) % opt.save_period == 0 or final_epoch:
                    logger.log_model("last.pt", opt, epoch, eval_metric["val/val_acc"], is_best)

                # save best model if the last model is not already best
                # to prevent model saving duplication
                if final_epoch and not is_best:
                    logger.log_model("best.pt", opt, epoch, eval_metric["val/val_acc"], best_model=True)

                del ckpt

        # Add epoch value for better visuals
        metric_dict["val/epoch"] = epoch

        logger.log_metric(metric_dict)

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')

    logger.end_log()

    return eval_metric


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="configs/model.yaml", help='initial weights path')
    parser.add_argument('--data', type=str, default="configs/data.yaml", help='dataset.yaml path')
    parser.add_argument('--epochs', type=int, default=50, help='total training epochs')
    parser.add_argument('--no-save', action='store_true', help='only save final checkpoint')
    parser.add_argument('--no-val', action='store_true', help='only validate final epoch')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default="pytorch-cifar10", help='save to project/name')
    parser.add_argument('--save-period', type=int, default=float("inf"), help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--bin-dir', type=str, default="bin")
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')

    return parser.parse_known_args()[0] if known else parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    train(opt)