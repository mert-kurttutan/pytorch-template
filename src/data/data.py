"""
   CIFAR-10 data normalization reference:
   https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
"""

import torch
import torchvision

from torch.utils.data import DataLoader
import timm


def get_dataset_cifar10(
    data_type, img_size, auto_augment_str="rand-m1-mstd0.5-inc1",
) -> torch.utils.data.dataloader.DataLoader:
    """
    creates and return train/dev dataloader
    with hyperparameters (params.subset_percent = 1.)
    """
    # apply co-variant transformation if wanted
    # using random crops and horizontal flip for train set
    cifar10_mean = (0.49139968, 0.48215827 ,0.44653124)
    cifar10_std = (0.24703233, 0.24348505, 0.26158768)
    train_transforms = timm.data.create_transform(
        input_size=img_size,
        is_training=True,
        mean=cifar10_mean,
        std=cifar10_std,
        auto_augment=auto_augment_str,
    )

    # transformer for dev set
    dev_transforms = timm.data.create_transform(
        input_size=img_size, mean=cifar10_mean, std=cifar10_std
    )

    if data_type == 'train':
        dataset = torchvision.datasets.CIFAR10(
            root='./data/cifar10', train=True,
            download=True, transform=train_transforms
        )
    else:
        dataset = torchvision.datasets.CIFAR10(
            root='./data/cifar10', train=False,
            download=True, transform=dev_transforms
        )

    return dataset


def get_cifar10_dataloader(
    dataset_config,
    batch_size,
    shuffle,
    num_workers,
):

    cifar10_dataset = get_dataset_cifar10(**dataset_config)
    return DataLoader(
        dataset=cifar10_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
