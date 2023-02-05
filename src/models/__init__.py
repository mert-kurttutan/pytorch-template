from .base_cnn import BaseCNN


def get_model(config):

    if config["model_type"] == "base_cnn":
        return BaseCNN(config)
    else:
        raise NotImplementedError
