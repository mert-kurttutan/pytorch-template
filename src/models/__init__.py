from .base_cnn import BaseCNN
import timm
from torch import nn


def get_classifier_head(in_dim, out_dim):
    return nn.Sequential(
        nn.BatchNorm1d(in_dim),
        nn.Linear(in_features=in_dim, out_features=512, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(0.4),
        nn.Linear(in_features=512, out_features=out_dim, bias=False)
    )


def get_model(config):

    if config["model_name"] == "base_cnn":
        return BaseCNN(config)
    else:
        improve_classifier = config.pop("improve_classifier", True)
        model = timm.create_model(**config)
        config["improve_classifier"] = True
        if improve_classifier:
            in_dim = model.get_classifier().in_features
            out_dim = model.get_classifier().out_features
            model.fc = get_classifier_head(in_dim, out_dim)
        return model
