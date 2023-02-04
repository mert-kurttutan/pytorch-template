import time
from copy import deepcopy
from datetime import datetime

import torch

import wandb
from tqdm import tqdm

from src.models import get_model
from src.data.data import get_cifar10_dataloader
from src.optimizers import get_scheduler
from src.utils.utils import (
    LOGGER, TQDM_BAR_FORMAT, init_seeds,
    config_load, is_config, is_serialized
)
from src.eval.loss import loss_fn
from src.utils.loggers import Logger
from src.optimizers import get_optimizer

from torch import nn
from torch.utils.data import DataLoader

import torch

from src.data.data import get_cifar10_dataloader

from src.eval.metric import accuracy
from src.utils.utils import (LOGGER, config_load, is_config, is_serialized)



# TODO: Use run_id to resume training

class ModelRunner():
    def __init__(self, opt) -> None:

        self.opt = opt
        # Loggers
        self.logger = Logger(
            logger="wandb",
            opt=opt,
            root_logger = LOGGER,
        )

    def train(self):
        opt = self.opt
        init_seeds(opt.seed, deterministic=True)
        device = "cuda" if opt.device != "cpu" else opt.device
        data, end_epoch, no_val, no_save, workers, model = (
            opt.data, opt.epochs, opt.no_val, opt.no_save, opt.workers, opt.model
        )

        opt.optimizer = config_load(opt.optimizer)
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

        # Optimizer
        optimizer = get_optimizer(model, opt)
        scheduler = get_scheduler(optimizer, opt)

        # Resume
        best_metric, start_epoch = -float("inf"), 0
        n_steps_per_epoch = len(train_loader)

        # Start training
        t0 = time.time()
        LOGGER.info(
            f'Using {train_loader.num_workers} dataloader workers\n'
            f"Logging results to {'bold'}\n"
            f'Starting training for {end_epoch} epochs...'
        )
        for epoch in range(start_epoch, end_epoch):
            # on_train_epoch_start
            model.train()
            pbar = tqdm(train_loader, bar_format=TQDM_BAR_FORMAT)  # progress bar
            for i, (train_x, train_y) in enumerate(pbar):
                # on_train_batch_start()
                optimizer.zero_grad()
                # Forward
                train_x, train_y = train_x.to(device), train_y.to(device)
                train_y_hat = model(train_x)
                loss = loss_fn(train_y_hat, train_y.type(torch.long))

                # Backward
                loss.backward()
                optimizer.step()

                # Collect Metric
                metric_dict = {"train/loss": loss.item(), "train/epoch": epoch}
                if i+1 < n_steps_per_epoch:
                    self.logger.log_metric(metric_dict)

                pbar.set_description(f"{epoch}/{end_epoch-1}, loss={loss.item():.4f}")
                # end batch ------------------------------------------------------------------------------------------------

            scheduler.step()
            # on_train_end
            final_epoch = (epoch + 1 == end_epoch)
            if (not no_val or final_epoch):
                eval_metric = ModelRunner.validate(
                    data=val_loader,
                    model=model,
                    compute_loss=loss_fn,
                    device=opt.device,
                    workers=workers,
                )

                train_metric = ModelRunner.validate(
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
                        self.logger.log_model("last.pt", opt, epoch, eval_metric["val/val_acc"], is_best)

                    # save best model if the last model is not already best
                    # to prevent model saving duplication
                    if final_epoch and not is_best:
                        self.logger.log_model("best.pt", opt, epoch, eval_metric["val/val_acc"], best_model=True)

                    del ckpt

            # Add epoch value for better visuals
            lr = [x['lr'] for x in optimizer.param_groups]
            metric_dict["lr"] = wandb.Histogram(lr)
            metric_dict["val/epoch"] = epoch

            # Epoch level metrics
            self.logger.log_metric(metric_dict)

        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')

        self.logger.end_log()

        return eval_metric


    @torch.inference_mode()
    def validate(
        model: nn.Module | dict | str = None,
        data: DataLoader | dict | str = None,
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
            model = get_model(model_conf)

        elif is_serialized(model):
            raise NotImplementedError

        # Configure
        model = model.to(device)

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



