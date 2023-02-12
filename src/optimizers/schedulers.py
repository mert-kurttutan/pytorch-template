import timm.scheduler as scheduler


def get_scheduler(optimizer, opt_config):
    # Scheduler
    return scheduler.CosineLRScheduler(
        optimizer,
        t_initial=opt_config["epochs"],
        lr_min=opt_config.get("lr_min", 1e-6),
        t_in_epochs=True,
        warmup_t=opt_config.get("warmup_t", 0),
        warmup_lr_init=opt_config.get("warmup_lr_init", 1e-4),
    )
