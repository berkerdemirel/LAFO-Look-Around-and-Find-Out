import logging
from typing import List, Optional

import hydra
import lightning.pytorch as pl
import omegaconf
import torch
from lightning.pytorch import Callback
from omegaconf import DictConfig, ListConfig
import numpy as np
from lafo.utils import metrics
from lafo.utils.load_model import get_model
from lafo.utils.score_functions import LAFO, knn_score, fDBD
import matplotlib.pyplot as plt

# seed_everything
from pytorch_lightning import seed_everything

# Force the execution of __init__.py if this file is executed directly.
import lafo  # noqa

torch.set_float32_matmul_precision("high")

def cache_run(cfg, device):
    id_train_size = 50000
    id_val_size = 10000
    # benchmark = cfg.benchmark + "_cfgs"
    num_classes = cfg.num_classes

    cache_dir = f"cache/{cfg.in_dataset}_train_{cfg.model_name}_in"
    feat_log = torch.from_numpy(np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode='r', shape=(id_train_size, 512))).to(device)
    score_log = torch.from_numpy(np.memmap(f"{cache_dir}/score.mmap", dtype=float, mode='r', shape=(id_train_size, num_classes))).to(device)
    label_log = torch.from_numpy(np.memmap(f"{cache_dir}/label.mmap", dtype=float, mode='r', shape=(id_train_size,))).to(device)


    cache_dir = f"cache/{cfg.in_dataset}_val_{cfg.model_name}_in"
    feat_log_val = torch.from_numpy(np.memmap(f"{cache_dir}/feat.mmap", dtype=float, mode='r', shape=(id_val_size, 512))).to(device)
    score_log_val = torch.from_numpy(np.memmap(f"{cache_dir}/score.mmap", dtype=float, mode='r', shape=(id_val_size, num_classes))).to(device)
    label_log_val = torch.from_numpy(np.memmap(f"{cache_dir}/label.mmap", dtype=float, mode='r', shape=(id_val_size,))).to(device)


    ood_feat_score_log = {}
    ood_dataset_size = {
        'SVHN':26032,
        'iSUN': 8925,
        'places365': 10000,
        'dtd': 5640
    }

    for ood_dataset in cfg.out_datasets:
        ood_feat_log = torch.from_numpy(np.memmap(f"cache/{ood_dataset}vs{cfg.in_dataset}_{cfg.model_name}_out/feat.mmap", dtype=float, mode='r', shape=(ood_dataset_size[ood_dataset], 512))).to(device)
        ood_score_log = torch.from_numpy(np.memmap(f"cache/{ood_dataset}vs{cfg.in_dataset}_{cfg.model_name}_out/score.mmap", dtype=float, mode='r', shape=(ood_dataset_size[ood_dataset], num_classes))).to(device)
        ood_feat_score_log[ood_dataset] = ood_feat_log, ood_score_log 

    model = get_model(cfg=cfg)

    model.to(cfg.device)

    in_dataset = torch.utils.data.TensorDataset(feat_log_val.cpu(), score_log_val.cpu())
    in_loader = torch.utils.data.DataLoader(in_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=2)


    # class_means = torch.load("/nfs/scistore19/locatgrp/bdemirel/fDBD-OOD/class_means_CIFAR-10.pt")
    class_means = torch.zeros(num_classes, feat_log.size(1)).to(device)
    for i in range(num_classes):
        class_means[i] = torch.mean(feat_log[label_log == i], dim=0).to(device)


    score_in = LAFO(model, in_loader, num_classes, class_means)
    # score_in = knn_score(model, in_loader, num_classes, train_feats, k=50)
    # score_in = fDBD(model, in_loader, num_classes, class_means)

    all_results = []
    all_score_out = []


    for ood_dataset_name, (feat_log, score_log) in ood_feat_score_log.items():
        ood_dataset = torch.utils.data.TensorDataset(feat_log.cpu(), score_log.cpu())
        # dataloader
        ood_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

        scores_out_test = LAFO(model, ood_loader, num_classes, class_means)
        # scores_out_test = knn_score(model, ood_loader, num_classes, train_feats, k=50)
        # scores_out_test = fDBD(model, ood_loader, num_classes, class_means)

        all_score_out.extend(scores_out_test)
        results = metrics.cal_metric(score_in, scores_out_test)
        all_results.append(results)
        # create histogram plot for score_in scores_out_test
        plt.hist(score_in, bins=70, alpha=0.5, label='in', density=True)
        plt.hist(scores_out_test, bins=70, alpha=0.5, label='out', density=True)
        plt.legend(loc='upper right', fontsize=9)
        if ood_dataset_name == 'dtd':
            ood_dataset_name = 'Texture'
        print(ood_dataset_name)
        # plt.xlabel('LAFO(x)', fontsize=16)
        plt.xlabel('LAFO(x)', fontsize=16)
        plt.ylabel('Density', fontsize=16)
        plt.title(f"{cfg.in_dataset} vs {ood_dataset_name}", fontsize=16)
        plt.savefig(f"./{cfg.in_dataset}_{ood_dataset_name}_hist_LAFO.png", dpi=600)
        plt.close()
    metrics.print_all_results(all_results, cfg.out_datasets, 'LAFO')


def pipeline(cfg):
    pass


@hydra.main(config_path="./cfgs", config_name="config.yaml", version_base="1.2")
def main(cfg: omegaconf.DictConfig):
    seed_everything(cfg.seed)
    if cfg.use_cache:
        logging.info("Using cached features")
        benchmark = cfg.benchmark + "_cfgs"
        cache_run(cfg[benchmark], cfg.device)
    else:
        logging.info("Running the model through the pipeline")
    



if __name__ == "__main__":
    main()
