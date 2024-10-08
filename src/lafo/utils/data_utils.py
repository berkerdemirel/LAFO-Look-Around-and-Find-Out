import os

import hydra
import torch
import torchvision
from easydict import EasyDict
from omegaconf import DictConfig
from torchvision import transforms

transform_dict = {
    "imagenet": {
        "resnet": {
            "train": transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.CenterCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            ),
            "test": transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            ),
        },
        "vit": {
            "train": transforms.Compose(
                [
                    transforms.Resize((384, 384)),
                    transforms.RandomResizedCrop(384),
                    transforms.CenterCrop(384),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            ),
            "test": transforms.Compose(
                [
                    transforms.Resize((384, 384)),
                    transforms.CenterCrop(384),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            ),
        },
    },
    "cifar10": {
        "resnet": {
            "train": transforms.Compose(
                [
                    transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            ),
            "test": transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.CenterCrop(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            ),
        },
    },
}


def get_loader_in(
    root_dir: str,
    cfg: DictConfig,
) -> EasyDict:
    if cfg.in_dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(
            root=root_dir,
            train=True,
            download=True,
            transform=transform_dict[cfg.in_dataset][cfg.arch_base]["train"],
        )
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        valset = torchvision.datasets.CIFAR10(
            root=root_dir,
            train=False,
            download=True,
            transform=transform_dict[cfg.in_dataset][cfg.arch_base]["test"],
        )
        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
    elif cfg.in_dataset == "imagenet":
        train_dir = os.path.join(root_dir, cfg.in_dataset, "train")
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(train_dir, transform_dict[cfg.in_dataset][cfg.arch_base]["train"]),
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            shuffle=False,
            pin_memory=True,
        )
        val_dir = os.path.join(root_dir, cfg.in_dataset, "val")
        val_loader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(val_dir, transform_dict[cfg.in_dataset][cfg.arch_base]["test"]),
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            shuffle=False,
            pin_memory=True,
        )
    else:
        raise NotImplementedError

    return EasyDict(
        {
            "train_loader": train_loader,
            "val_loader": val_loader,
        }
    )


def get_loader_out(
    root_dir: str,
    cfg: DictConfig,
    val_dataset_name: str,
) -> EasyDict:
    if val_dataset_name == "SVHN":
        val_ood_loader = torch.utils.data.DataLoader(
            torchvision.datasets.SVHN(
                root=root_dir,
                split="test",
                download=True,
                transform=transform_dict[cfg.in_dataset][cfg.arch_base]["test"],
            ),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=2,
        )
    else:
        val_path = (
            os.path.join(root_dir, val_dataset_name, "images")
            if "dtd" == val_dataset_name
            else os.path.join(root_dir, val_dataset_name)
        )
        val_ood_loader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(
                root=val_path, transform=transform_dict[cfg.in_dataset][cfg.arch_base]["test"]
            ),
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    return EasyDict(
        {
            "val_ood_loader": val_ood_loader,
        }
    )


@hydra.main(config_path="../cfgs", config_name="config.yaml", version_base="1.2")
def main(cfg: DictConfig):
    print(cfg)
    benchmark = cfg.benchmark + "_cfgs"
    for ood_data_name in cfg[benchmark].out_datasets:
        print(ood_data_name)
        val_loader = get_loader_out("../../../data", cfg[benchmark], ood_data_name)
        print(val_loader)

    loader_in_dict = get_loader_in("../../../data", cfg[benchmark])
    print(loader_in_dict.train_loader)
    print(loader_in_dict.val_loader)


if __name__ == "__main__":
    main()
