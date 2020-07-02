from itertools import accumulate
from typing import Dict, List, Optional, Tuple

import torch
#local_rank = torch.distributed.get_rank()
#torch.distributed.init_process_group(backend="mpi", rank=local_rank, group="main")
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset, DistributedSampler


def load_mnist(
    downsample_pct: float = 0.5,
    train_pct: float = 0.8,
    data_path: str = "~/dataset/mnist",
    batch_size: int = 128,
    num_workers: int=128,
    deterministic_partitions: bool = False,
    downsample_pct_test: Optional[float] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load MNIST dataset (download if necessary) and split data into training,
        validation, and test sets.

    Args:
        downsample_pct: the proportion of the dataset to use for training,
            validation, and test
        train_pct: the proportion of the downsampled data to use for training
        data_path: Root directory of dataset where `MNIST/processed/training.pt`
            and `MNIST/processed/test.pt` exist.
        batch_size: how many samples per batch to load
        num_workers: number of workers (subprocesses) for loading data
        deterministic_partitions: whether to partition data in a deterministic
            fashion
        downsample_pct_test: the proportion of the dataset to use for test, default
            to be equal to downsample_pct

    Returns:
        DataLoader: training data
        DataLoader: validation data
        DataLoader: test data
    """
    # Specify transforms
    transform = transforms.Compose(
        [transforms.Resize((32,32)),
         transforms.ToTensor(),
         #transforms.Normalize((0.1307,), (0.3081,)),
         ]
    )
    # Load training set
    train_valid_set = torchvision.datasets.MNIST(
        root=data_path, train=True, download=True, transform=transform
    )
    # Load test set
    test_set = torchvision.datasets.MNIST(
        root=data_path, train=False, download=True, transform=transform
    )
    return get_partition_data_loaders(
        train_valid_set=train_valid_set,
        test_set=test_set,
        downsample_pct=downsample_pct,
        train_pct=train_pct,
        batch_size=batch_size,
        num_workers=num_workers,
        deterministic_partitions=deterministic_partitions,
        downsample_pct_test=downsample_pct_test,
    )


def get_partition_data_loaders(
    train_valid_set: Dataset,
    test_set: Dataset,
    downsample_pct: float = 0.5,
    train_pct: float = 0.8,
    batch_size: int = 128,
    num_workers: int = 0,
    deterministic_partitions: bool = False,
    downsample_pct_test: Optional[float] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Helper function for partitioning training data into training and validation sets,
        downsampling data, and initializing DataLoaders for each partition.

    Args:
        train_valid_set: torch.dataset
        downsample_pct: the proportion of the dataset to use for training, and
            validation
        train_pct: the proportion of the downsampled data to use for training
        batch_size: how many samples per batch to load
        num_workers: number of workers (subprocesses) for loading data
        deterministic_partitions: whether to partition data in a deterministic
            fashion
        downsample_pct_test: the proportion of the dataset to use for test, default
            to be equal to downsample_pct

    Returns:
        DataLoader: training data
        DataLoader: validation data
        DataLoader: test data
    """
    # Partition into training/validation
    # pyre-ignore [6]
    downsampled_num_examples = int(downsample_pct * len(train_valid_set))
    n_train_examples = int(train_pct * downsampled_num_examples)
    n_valid_examples = downsampled_num_examples - n_train_examples
    train_set, valid_set, _ = split_dataset(
        dataset=train_valid_set,
        lengths=[
            n_train_examples,
            n_valid_examples,
            len(train_valid_set) - downsampled_num_examples,  # pyre-ignore [6]
        ],
        deterministic_partitions=deterministic_partitions,
    )
    if downsample_pct_test is None:
        downsample_pct_test = downsample_pct
    # pyre-ignore [6]
    downsampled_num_test_examples = int(downsample_pct_test * len(test_set))
    test_set, _ = split_dataset(
        test_set,
        lengths=[
            downsampled_num_test_examples,
            len(test_set) - downsampled_num_test_examples,  # pyre-ignore [6]
        ],
        deterministic_partitions=deterministic_partitions,
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    valid_loader = DataLoader(
        valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    return train_loader, valid_loader, test_loader


def split_dataset(
    dataset: Dataset, lengths: List[int], deterministic_partitions: bool = False
) -> List[Dataset]:
    """
    Split a dataset either randomly or deterministically.

    Args:
        dataset: the dataset to split
        lengths: the lengths of each partition
        deterministic_partitions: deterministic_partitions: whether to partition
            data in a deterministic fashion

    Returns:
        List[Dataset]: split datasets
    """
    if deterministic_partitions:
        indices = list(range(sum(lengths)))
    else:
        indices = torch.randperm(sum(lengths)).tolist()
    return [
        Subset(dataset, indices[offset - length : offset])
        for offset, length in zip(accumulate(lengths), lengths)
    ]

class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.c1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=(5,5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(16, 120, kernel_size=(5, 5)),
            nn.ReLU()
        )
        self.f4 = nn.Sequential(nn.Linear(120, 84), nn.ReLU())
        self.f5 = nn.Sequential(nn.Linear(84, 10))
    def forward(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = x.view(x.size(0), -1)
        x = self.f4(x)
        x = self.f5(x)
        return x