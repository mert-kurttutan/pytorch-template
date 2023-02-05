import math

from torch.optim import lr_scheduler


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def get_scheduler(optimizer, opt_config):
    # Scheduler
    if opt_config["cos_lr"]:
        lf = one_cycle(1, opt_config['lrf'], opt_config["epochs"])
    else:
        lf = lambda x: (
            (1 - x / opt_config["epochs"]) * (1.0 - opt_config['lrf']) + opt_config['lrf']
        )

    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
