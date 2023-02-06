from .base_cnn import BaseCNN
import timm


def get_model(config):

    
    if config["model_name"] == "base_cnn":
        return BaseCNN(config)
    else:
        return timm.create_model(**config)

