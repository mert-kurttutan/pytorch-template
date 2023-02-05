import torch


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.
    Args:
        outputs: (np.ndarray) output of the model
        labels: (np.ndarray) [0, 1, ..., n_class-1]
    Returns: (float) accuracy in [0,1]
    """
    outputs = torch.argmax(outputs, axis=1)
    return torch.mean((outputs == labels).float())
