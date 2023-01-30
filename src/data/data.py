import torch

from torch.utils.data import Dataset, DataLoader

class SimpleDataset(Dataset):

    def __init__(self, in_dim, out_dim, dataset_len):
        self.dataset_len = dataset_len
        self.in_dim = in_dim
        self.out_dim = out_dim

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        label = idx % self.out_dim
        return torch.ones(self.in_dim)*label, label



def create_simple_dataloader(
    in_dim, out_dim, dataset_len, 
    batch_size,
    shuffle,
    num_workers,
):  

    simple_dataset = SimpleDataset(in_dim, out_dim, dataset_len)
    return DataLoader(
        dataset=simple_dataset,
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers,
    )