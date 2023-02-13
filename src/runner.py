import time
from copy import deepcopy
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader

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

from src.eval.metric import accuracy


class ModelRunner():
    def __init__(self, opt) -> None:

        self.opt = opt

    def init_log(self, opt, run_conf) -> None:
        # Loggers
        self.logger = Logger(
            logger="wandb",
            opt=opt,
            root_logger=LOGGER,
            run_config=run_conf
        )

    def train(self):
        opt = self.opt
        init_seeds(opt.seed, deterministic=True)
        device = "cuda" if opt.device != "cpu" else opt.device
        data, end_epoch, no_val, no_save, model = (
            opt.data, opt.epochs, opt.no_val, opt.no_save, opt.model
        )

        # Resume
        best_metric, start_epoch = -float("inf"), 0

        if opt.resume:
            ckpt = torch.load(opt.model)
            model_conf = ckpt["model_config"]
            model = get_model(model_conf)
            model = model.to(device)
            model.load_state_dict(ckpt["model"])
            opt_conf = ckpt["opt_config"]
            optimizer = get_optimizer(model, ckpt["opt_config"])
            scheduler = get_scheduler(optimizer, ckpt["opt_config"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            best_metric, start_epoch = ckpt["best_metric"], ckpt["epoch"] + 1

            # resume logger as well
            opt.run_id = ckpt["opt"]["run_id"]
            opt.run_name = ckpt["opt"]["run_name"]

        else:
            if is_config(model):
                model_conf = config_load(model)
                model = get_model(model_conf)

            elif is_serialized(model):
                ckpt = torch.load(opt.model)
                model_conf = ckpt["model_config"]
                model = get_model(model_conf)
                model.load_state_dict(ckpt["model"])

            # Configure
            model = model.to(device)
            # Optimizer
            opt_conf = config_load(opt.optimizer)
            opt_conf["epochs"] = opt.epochs
            optimizer = get_optimizer(model, opt_conf)
            scheduler = get_scheduler(optimizer, opt_conf)

        data = config_load(data)
        self.init_log(opt, run_conf={"model": model_conf, "data": data, "optimizer": opt_conf})

        train_data, val_data = data["train"], data["val"]
        assert is_config(train_data)

        train_loader = get_cifar10_dataloader(
            **train_data,
            **data["mode"]["train"],
            num_workers=data["workers"],
        )

        val_loader = get_cifar10_dataloader(
            **val_data,
            **data["mode"]["val"],
            num_workers=data["workers"],
        )

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
                # Forward
                train_x, train_y = train_x.to(device), train_y.to(device)
                train_y_hat = model(train_x)
                loss = loss_fn(train_y_hat, train_y.type(torch.long))

                # Backward
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Collect Metric
                metric_dict = {"train/loss": loss.item(), "train/epoch": epoch}
                if i+1 < n_steps_per_epoch:
                    self.logger.log_metric(metric_dict)

                acc = accuracy(train_y_hat, train_y).item()
                pbar.set_description(
                    f"{epoch}/{end_epoch-1}, loss={loss.item():.4f}, acc={acc:.4f}"
                )
                # end batch ---------------------------------------------------------

            scheduler.step(epoch+1)
            # on_train_end
            final_epoch = (epoch + 1 == end_epoch)
            if (not no_val or final_epoch):
                eval_metric = ModelRunner.validate(
                    data=val_loader,
                    model=model,
                    compute_loss=loss_fn,
                    device=opt.device,
                    workers=data["workers"],
                )

                metric_dict = {**metric_dict, **eval_metric}

                # Update best metric
                if eval_metric["val/val_acc"] > best_metric:
                    best_metric = eval_metric["val/val_acc"]

                # Save model
                if not no_save or final_epoch:
                    ckpt = {
                        'epoch': epoch,
                        'best_metric': best_metric,
                        'model': deepcopy(model).state_dict(),
                        "model_config": model_conf,
                        "opt_config": opt_conf,
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'opt': vars(opt),
                        'date': datetime.now().isoformat()
                    }
                    last, best = (
                        f"{opt.bin_dir}/{opt.run_name}_last.pt",
                        f"{opt.bin_dir}/{opt.run_name}_best.pt"
                    )
                    is_best = best_metric == eval_metric["val/val_acc"]
                    # Save last, best and delete
                    torch.save(ckpt, last)
                    if is_best:
                        torch.save(ckpt, best)
                    if (epoch+1) % opt.save_period == 0 or final_epoch:
                        self.logger.log_model(
                            f"{opt.run_name}_last.pt", opt, epoch,
                            eval_metric["val/val_acc"], is_best
                        )

                    # save best model if the last model is not already best
                    # to prevent model saving duplication
                    if final_epoch and not is_best:
                        self.logger.log_model(
                            f"{opt.run_name}_best.pt", opt, epoch,
                            eval_metric["val/val_acc"], best_model=True
                        )

                    del ckpt

            # Add epoch value for better visuals
            metric_dict["params/lr"] = {
                str(idx): x['lr'] for idx, x in enumerate(optimizer.param_groups)
            }
            metric_dict["val/epoch"] = epoch

            # Epoch level metrics
            self.logger.log_metric(metric_dict)

        LOGGER.info(
            f'\n{epoch-start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.'
        )

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
        dataset_len = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            dataset_len += x.shape[0]

            # start of validation step
            y_hat = model(x)

            # update metrics
            eval_dict[f"val/{data_name}_acc"] += accuracy(y_hat, y).item() * x.shape[0]
            if compute_loss:
                eval_dict[f"val/{data_name}_loss"] += compute_loss(y_hat, y).item() * x.shape[0]

        # turn sum -> mean (over batch)
        eval_dict[f"val/{data_name}_loss"] /= dataset_len
        eval_dict[f"val/{data_name}_acc"] /= dataset_len
        if not compute_loss:
            del eval_dict[f"val/{data_name}_loss"]
        LOGGER.info(eval_dict)

        return eval_dict
