import os

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from lafo.utils.data_utils import get_loader_in, get_loader_out
from lafo.utils.model_utils import get_model

feat_dim_dict = {"resnet50": 2048, "resnet50-supcon": 2048, "vit": 768, "resnet18": 512, "resnet18-supcon": 512}


def vit_features(model, x):
    b, c, fh, fw = x.shape
    x = model.patch_embedding(x)  # b,d,gh,gw
    x = x.flatten(2).transpose(1, 2)  # b,gh*gw,d
    if hasattr(model, "class_token"):
        x = torch.cat((model.class_token.expand(b, -1, -1), x), dim=1)  # b,gh*gw+1,d
    if hasattr(model, "positional_embedding"):
        x = model.positional_embedding(x)  # b,gh*gw+1,d
    x = model.transformer(x)  # b,gh*gw+1,d
    x = model.norm(x)[:, 0]  # b,d
    return x


def feature_extract_helper(
    model, dataloader, feat_dim, in_dataset, model_name, num_classes, device, ood_dataset, split
):
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)

    os.environ["CUDA_VISIBLE_DEVICES"] = device
    model.to(device)
    model.eval()
    if ood_dataset:
        cache_dir = f"./cache/{ood_dataset}vs{in_dataset}_{model_name}_out"
    else:
        cache_dir = (
            f"./cache/{in_dataset}_{split}_{model_name}_in" if split else f"./cache/{in_dataset}_{model_name}_in"
        )
    if True:  # not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        feat_log = np.memmap(
            f"{cache_dir}/feat.mmap", dtype=float, mode="w+", shape=(len(dataloader.dataset), feat_dim)
        )
        score_log = np.memmap(
            f"{cache_dir}/score.mmap", dtype=float, mode="w+", shape=(len(dataloader.dataset), num_classes)
        )
        label_log = np.memmap(f"{cache_dir}/label.mmap", dtype=float, mode="w+", shape=(len(dataloader.dataset),))

        with torch.no_grad():
            start_ind = 0
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                B, C, H, W = inputs.size()
                inputs, targets = inputs.to(device), targets.to(device)
                # start_ind = batch_idx * B
                # end_ind = min((batch_idx + 1) * B, len(dataloader.dataset))
                end_ind = start_ind + B
                if model_name == "resnet50-supcon":
                    out = model.encoder(inputs)
                elif model_name == "resnet50":
                    out = model.features(inputs)
                elif model_name == "vit":
                    out = vit_features(model, inputs)
                elif model_name == "resnet18":
                    out = model.penult_feature(inputs)
                elif model_name == "resnet18-supcon":
                    out = model.penult_feature(inputs)
                if len(out.shape) > 2:
                    out = F.adaptive_avg_pool2d(out, 1)
                    out = out.view(out.size(0), -1)
                score = model.fc(out)
                feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()
                label_log[start_ind:end_ind] = targets.data.cpu().numpy()
                score_log[start_ind:end_ind] = score.data.cpu().numpy()
                start_ind = end_ind
                if batch_idx % 100 == 0:
                    print(f"{batch_idx}/{len(dataloader)}")
    else:
        feat_log = np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode="r", shape=(len(dataloader.dataset), feat_dim))
        score_log = np.memmap(
            f"{cache_dir}/score.mmap", dtype=float, mode="r", shape=(len(dataloader.dataset), num_classes)
        )
        label_log = np.memmap(f"{cache_dir}/label.mmap", dtype=float, mode="r", shape=(len(dataloader.dataset),))


def feat_extract(cfg, data_dir, device):
    model = get_model(cfg.in_dataset, cfg.model_name, cfg.model_path, cfg.num_classes)
    feat_dim = feat_dim_dict[cfg.model_name]

    for ood_data_name in cfg.out_datasets:
        val_loader = get_loader_out(
            root_dir=data_dir,
            val_dataset_name=ood_data_name,
            in_dataset=cfg.in_dataset,
            arch_base=cfg.arch_base,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
        )
        feature_extract_helper(
            model,
            val_loader["val_ood_loader"],
            feat_dim,
            cfg.in_dataset,
            cfg.model_name,
            cfg.num_classes,
            device,
            ood_data_name,
            None,
        )

    in_loader_dict = get_loader_in(
        root_dir=data_dir,
        in_dataset=cfg.in_dataset,
        arch_base=cfg.arch_base,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    in_loader_train = in_loader_dict.train_loader
    in_loader_val = in_loader_dict.val_loader

    feature_extract_helper(
        model, in_loader_train, feat_dim, cfg.in_dataset, cfg.model_name, cfg.num_classes, device, None, "train"
    )
    feature_extract_helper(
        model, in_loader_val, feat_dim, cfg.in_dataset, cfg.model_name, cfg.num_classes, device, None, "val"
    )
    print("Feature extraction done!")


@hydra.main(config_path="../cfgs", config_name="config.yaml", version_base="1.2")
def main(cfg: DictConfig):
    benchmark = cfg.benchmark + "_cfgs"

    feat_extract(cfg[benchmark], cfg.data_dir, cfg.device)


if __name__ == "__main__":
    main()
