
from torch import optim


def get_optimizer(model, opt_config):

    if opt_config["type"] == "sgd":
        return optim.SGD(
            model.parameters(), lr=opt_config["lr0"], momentum=opt_config["momentum"]
        )

    elif opt_config["type"] == "adam":
        return optim.Adam(model.parameters(), lr=opt_config["lr0"])

    elif opt_config["type"] == "adamw":
        return optim.AdamW(
            model.parameters(), lr=opt_config["lr0"],
            weight_decay=opt_config["weight_decay"]
        )
