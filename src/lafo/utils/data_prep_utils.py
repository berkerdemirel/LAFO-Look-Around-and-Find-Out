import numpy as np
import torch

from lafo.utils.data_utils import get_loader_in, get_loader_out
from lafo.utils.model_utils import get_model
from lafo.utils.score_functions import get_class_means


def data_prep_cache(
    data_dir: str,
    in_dataset: str,
    num_classes: int,
    out_datasets: list[str],
    model_path: str,
    model_name: str,
    arch_base: str,
    batch_size: int,
    num_workers: int,
    cache_root: str,
    device: str,
) -> tuple[torch.nn.Module, torch.utils.data.DataLoader, dict[str, torch.utils.data.DataLoader], int, torch.Tensor]:
    in_loader_dict = get_loader_in(
        root_dir=data_dir, in_dataset=in_dataset, arch_base=arch_base, batch_size=batch_size, num_workers=num_workers
    )
    id_train_size = len(in_loader_dict.train_loader.dataset)
    id_val_size = len(in_loader_dict.val_loader.dataset)

    cache_dir = f"{cache_root}{in_dataset}_train_{model_name}_in"
    feat_log = torch.from_numpy(
        np.copy(np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode="r", shape=(id_train_size, 512)))
    )
    # score_log = torch.from_numpy(
    #     np.copy(np.memmap(f"{cache_dir}/score.mmap", dtype=float, mode="r", shape=(id_train_size, num_classes)))
    # )
    label_log = torch.from_numpy(
        np.copy(np.memmap(f"{cache_dir}/label.mmap", dtype=float, mode="r", shape=(id_train_size,)))
    )

    cache_dir = f"{cache_root}{in_dataset}_val_{model_name}_in"
    feat_log_val = torch.from_numpy(
        np.copy(np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode="r", shape=(id_val_size, 512)))
    )
    score_log_val = torch.from_numpy(
        np.copy(np.memmap(f"{cache_dir}/score.mmap", dtype=float, mode="r", shape=(id_val_size, num_classes)))
    )
    # label_log_val = torch.from_numpy(
    #     np.memmap(f"{cache_dir}/label.mmap", dtype=float, mode="r", shape=(id_val_size,))
    # )

    ood_loaders = {}
    for ood_dataset_name in out_datasets:
        ood_dataset_size = len(
            get_loader_out(
                root_dir=data_dir,
                val_dataset_name=ood_dataset_name,
                in_dataset=in_dataset,
                arch_base=arch_base,
                batch_size=batch_size,
                num_workers=num_workers,
            )["val_ood_loader"].dataset
        )

        ood_feat_log = torch.from_numpy(
            np.copy(
                np.memmap(
                    f"{cache_root}{ood_dataset_name}vs{in_dataset}_{model_name}_out/feat.mmap",
                    dtype=float,
                    mode="r",
                    shape=(ood_dataset_size, 512),
                )
            )
        ).to(device)
        ood_score_log = torch.from_numpy(
            np.copy(
                np.memmap(
                    f"{cache_root}{ood_dataset_name}vs{in_dataset}_{model_name}_out/score.mmap",
                    dtype=float,
                    mode="r",
                    shape=(ood_dataset_size, num_classes),
                )
            )
        ).to(device)
        ood_dataset = torch.utils.data.TensorDataset(ood_feat_log.cpu(), ood_score_log.cpu())
        ood_loader = torch.utils.data.DataLoader(
            ood_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        ood_loaders[ood_dataset_name] = ood_loader

    model = get_model(in_dataset, model_name, model_path, num_classes)

    in_dataset = torch.utils.data.TensorDataset(feat_log_val.cpu(), score_log_val.cpu())
    in_loader = torch.utils.data.DataLoader(in_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    class_means = torch.zeros(num_classes, feat_log.size(1)).to(device)
    for i in range(num_classes):
        class_means[i] = torch.mean(feat_log[label_log == i], dim=0).to(device)

    return model, in_loader, ood_loaders, num_classes, class_means


def data_prep_no_cache(
    data_dir,
    in_dataset,
    num_classes,
    out_datasets,
    model_path,
    model_name,
    arch_base,
    batch_size,
    num_workers,
    device,
) -> tuple[torch.nn.Module, torch.utils.data.DataLoader, dict[str, torch.utils.data.DataLoader], int, torch.Tensor]:
    in_loader_dict = get_loader_in(
        root_dir=data_dir, in_dataset=in_dataset, arch_base=arch_base, batch_size=batch_size, num_workers=num_workers
    )
    model = get_model(in_dataset, model_name, model_path, num_classes)
    class_means = get_class_means(in_loader_dict.train_loader, model, device)

    ood_loaders = {}

    for ood_dataset_name in out_datasets:
        ood_loader = get_loader_out(
            root_dir=data_dir,
            val_dataset_name=ood_dataset_name,
            in_dataset=in_dataset,
            arch_base=arch_base,
            batch_size=batch_size,
            num_workers=num_workers,
        )["val_ood_loader"]
        ood_loaders[ood_dataset_name] = ood_loader
    return model, in_loader_dict.val_loader, ood_loaders, num_classes, class_means
