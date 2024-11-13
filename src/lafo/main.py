import logging

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

from lafo.runners.cache_runner import cache_run
from lafo.runners.pipeline_runner import pipeline_run

torch.set_float32_matmul_precision("high")


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
        pipeline_run(cfg[benchmark], cfg.data_dir, cfg.use_cache, cfg.device)


if __name__ == "__main__":
    main()
