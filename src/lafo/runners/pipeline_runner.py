from lafo.utils import metrics
from lafo.utils.data_prep_utils import data_prep_no_cache
from lafo.utils.plots_utils import plot_helper
from lafo.utils.score_functions import LAFO


def pipeline_run(cfg, data_dir, use_cache, device):
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

        all_score_out.extend(scores_out_test)
        results = metrics.cal_metric(score_in, scores_out_test)
        all_results.append(results)
        plot_helper(score_in, scores_out_test, cfg.in_dataset, ood_dataset_name)
    metrics.print_all_results(all_results, cfg.out_datasets, "LAFO")
