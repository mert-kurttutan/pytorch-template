
from torch import nn


def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.
    Args:
        outputs: dimension batch_size x class size - output of the model
        labels: dimension batch_size
    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    """
    return nn.CrossEntropyLoss()(outputs, labels)
