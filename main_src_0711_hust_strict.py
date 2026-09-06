#!/usr/bin/env python3
"""SSP-lite plus SDE robust source route for the strict HUST protocol."""

import hydra
import omegaconf

from Lib.hust_source_training import prepare_hust_source_config, train_hust_source


@hydra.main(version_base=None, config_path="./Configs", config_name="defaults")
def run(cfg: omegaconf.DictConfig):
    source = prepare_hust_source_config(cfg)
    return train_hust_source(cfg, source=source, variant="robust")


if __name__ == "__main__":
    run()
