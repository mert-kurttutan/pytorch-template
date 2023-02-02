"""
   CIFAR-10 data normalization reference:
   https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
"""

import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader


def get_dataset_cifar10(data_type, augmentation) -> torch.utils.data.dataloader.DataLoader:
    """
    creates and return train/dev dataloader with hyperparameters (params.subset_percent = 1.)
    """
    transform_list = []
    
    # apply co-variant transformation if wanted
    # using random crops and horizontal flip for train set
    if augmentation is not None or len(augmentation) == 0:
      
        for aug_func in augmentation:
            if aug_func == "crop":
                transform_list.append(transforms.RandomCrop(32, padding=4))
        
            elif aug_func == "horizontal_flip":
                transform_list.append(transforms.RandomHorizontalFlip()) # randomly flip image horizontally

    # standard pre-processing
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])


    train_transformer = transforms.Compose(transform_list)

    # transformer for dev set
    dev_transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    
    if data_type == 'train':
        dataset = torchvision.datasets.CIFAR10(
            root='./data/cifar10', train=True,
            download=True, transform=train_transformer
        )
    else:
        dataset = torchvision.datasets.CIFAR10(
            root='./data/cifar10', train=False,
            download=True, transform=dev_transformer
        )

    return dataset


def get_cifar10_dataloader(
    data_type, augmentation, 
    batch_size,
    shuffle,
    num_workers,
):  

    cifar10_dataset = get_dataset_cifar10(data_type, augmentation)
    return DataLoader(
        dataset=cifar10_dataset,
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers,
    )
