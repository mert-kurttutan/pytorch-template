import math

import timm.scheduler as scheduler


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def get_scheduler(optimizer, opt_config):
    # Scheduler
    return scheduler.CosineLRScheduler(
        optimizer,
        t_initial=opt_config["epochs"],
        cycle_decay=0.5,
        lr_min=1e-6,
        t_in_epochs=True,
        warmup_t=opt_config["epochs"]//10,
        warmup_lr_init=1e-4,
        cycle_limit=1,
    )
