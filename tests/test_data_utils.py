import os

import pytest
import torch

from lafo.utils.data_utils import get_loader_in, get_loader_out

is_github_actions = os.getenv("GITHUB_ACTIONS") == "true"


def test_get_loader_in_cifar10():
    root_dir = "./data"
    in_dataset = "cifar10"
    arch_base = "resnet"
    batch_size = 32
    num_workers = 2

    loaders = get_loader_in(
        root_dir=root_dir, in_dataset=in_dataset, arch_base=arch_base, batch_size=batch_size, num_workers=num_workers
    )

    assert isinstance(loaders.train_loader, torch.utils.data.DataLoader), "Train loader is not a DataLoader instance"
    assert isinstance(loaders.val_loader, torch.utils.data.DataLoader), "Validation loader is not a DataLoader instance"
    assert len(loaders.train_loader.dataset) == 50000, "Train dataset size is incorrect"
    assert len(loaders.val_loader.dataset) == 10000, "Validation dataset size is incorrect"


@pytest.mark.skipif(is_github_actions, reason="Skipping test that requires data folder in GitHub Actions")
def test_get_loader_in_imagenet():
    root_dir = "./data"
    in_dataset = "imagenet"
    arch_base = "resnet"
    batch_size = 32
    num_workers = 2

    loaders = get_loader_in(
        root_dir=root_dir, in_dataset=in_dataset, arch_base=arch_base, batch_size=batch_size, num_workers=num_workers
    )
    assert isinstance(loaders.train_loader, torch.utils.data.DataLoader), "Train loader is not a DataLoader instance"
    assert isinstance(loaders.val_loader, torch.utils.data.DataLoader), "Validation loader is not a DataLoader instance"
    # Note: Length checks for ImageNet datasets are omitted due to variability in dataset size


def test_get_loader_out_svhn():
    root_dir = "./data"
    in_dataset = "cifar10"
    arch_base = "resnet"
    batch_size = 32
    num_workers = 2
    val_dataset_name = "SVHN"

    loaders = get_loader_out(
        root_dir=root_dir,
        val_dataset_name=val_dataset_name,
        in_dataset=in_dataset,
        arch_base=arch_base,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    assert isinstance(
        loaders.val_ood_loader, torch.utils.data.DataLoader
    ), "OOD validation loader is not a DataLoader instance"
    assert len(loaders.val_ood_loader.dataset) == 26032, "OOD validation dataset size is incorrect"


@pytest.mark.skipif(is_github_actions, reason="Skipping test that requires data folder in GitHub Actions")
def test_get_loader_out_dtd():
    root_dir = "./data"
    in_dataset = "cifar10"
    arch_base = "resnet"
    batch_size = 32
    num_workers = 2
    val_dataset_name = "dtd"

    loaders = get_loader_out(
        root_dir=root_dir,
        val_dataset_name=val_dataset_name,
        in_dataset=in_dataset,
        arch_base=arch_base,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    assert isinstance(
        loaders.val_ood_loader, torch.utils.data.DataLoader
    ), "OOD validation loader is not a DataLoader instance"
    # Note: Length checks for DTD datasets are omitted due to variability in dataset size
