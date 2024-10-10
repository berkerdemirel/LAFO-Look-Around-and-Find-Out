import logging

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

from lafo.utils import metrics
from lafo.utils.data_prep_utils import data_prep_cache, data_prep_no_cache
from lafo.utils.plots_utils import plot_helper
from lafo.utils.score_functions import LAFO

torch.set_float32_matmul_precision("high")


def cache_run(cfg, data_dir, cache_root, use_cache, device):
    data_for_cache = data_prep_cache(
        data_dir,
        cfg.in_dataset,
        cfg.num_classes,
        cfg.out_datasets,
        cfg.model_path,
        cfg.model_name,
        cfg.arch_base,
        cfg.batch_size,
        cfg.num_workers,
        cache_root,
        device,
    )

    model, in_loader, ood_loaders, num_classes, class_means = data_for_cache
    score_in = LAFO(
        model=model,
        test_loader=in_loader,
        num_classes=num_classes,
        class_means=class_means,
        use_cache=use_cache,
        device=device,
    )
    # score_in = knn_score(model, in_loader, num_classes, train_feats, k=50)
    # score_in = fDBD(model, in_loader, num_classes, class_means)

    all_results = []
    all_score_out = []
    for ood_dataset_name in cfg.out_datasets:
        ood_loader = ood_loaders[ood_dataset_name]

        scores_out_test = LAFO(
            model=model,
            test_loader=ood_loader,
            num_classes=num_classes,
            class_means=class_means,
            use_cache=use_cache,
            device=device,
        )
        # scores_out_test = knn_score(model, ood_loader, num_classes, train_feats, k=50)
        # scores_out_test = fDBD(model, ood_loader, num_classes, class_means)

        all_score_out.extend(scores_out_test)
        results = metrics.cal_metric(score_in, scores_out_test)
        all_results.append(results)
        plot_helper(score_in, scores_out_test, cfg.in_dataset, ood_dataset_name)
        # create histogram plot for score_in scores_out_test
    metrics.print_all_results(all_results, cfg.out_datasets, "LAFO")


def pipeline(cfg, data_dir, use_cache, device):
    data_for_no_cache = data_prep_no_cache(
        data_dir,
        cfg.in_dataset,
        cfg.num_classes,
        cfg.out_datasets,
        cfg.model_path,
        cfg.model_name,
        cfg.arch_base,
        cfg.batch_size,
        cfg.num_workers,
        device,
    )
    model, in_loader, ood_loaders, num_classes, class_means = data_for_no_cache

    score_in = LAFO(
        model=model,
        test_loader=in_loader,
        num_classes=num_classes,
        class_means=class_means,
        use_cache=use_cache,
        device=device,
    )
    # score_in = knn_score(model, in_loader, num_classes, train_feats, k=50)
    # score_in = fDBD(model, in_loader, num_classes, class_means)

    all_results = []
    all_score_out = []

    for ood_dataset_name in cfg.out_datasets:
        ood_loader = ood_loaders[ood_dataset_name]
        scores_out_test = LAFO(
            model=model,
            test_loader=ood_loader,
            num_classes=num_classes,
            class_means=class_means,
            device=device,
            use_cache=use_cache,
        )
        # scores_out_test = knn_score(model, ood_loader, num_classes, train_feats, k=50)
        # scores_out_test = fDBD(model, ood_loader, num_classes, class_means)

        all_score_out.extend(scores_out_test)
        results = metrics.cal_metric(score_in, scores_out_test)
        all_results.append(results)
        plot_helper(score_in, scores_out_test, cfg.in_dataset, ood_dataset_name)
    metrics.print_all_results(all_results, cfg.out_datasets, "LAFO")


@hydra.main(config_path="./cfgs", config_name="config.yaml", version_base="1.2")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    if cfg.use_cache:
        logging.info("Using cached features")
        benchmark = cfg.benchmark + "_cfgs"
        cache_run(cfg[benchmark], cfg.data_dir, cfg.cache_dir, cfg.use_cache, cfg.device)
    else:
        logging.info("Running the model through the pipeline")
        benchmark = cfg.benchmark + "_cfgs"
        pipeline(cfg[benchmark], cfg.data_dir, cfg.use_cache, cfg.device)


if __name__ == "__main__":
    main()
