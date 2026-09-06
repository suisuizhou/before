#!/usr/bin/env python3
"""0711 SSP-lite/SDE robust source training for strict WTPG."""

import hydra
import omegaconf

from Lib.wtpg_source_training import prepare_wtpg_source_config, train_wtpg_source


@hydra.main(version_base=None, config_path="./Configs", config_name="defaults")
def run(cfg: omegaconf.DictConfig):
    source = prepare_wtpg_source_config(cfg)
    return train_wtpg_source(cfg, source, "robust")


if __name__ == "__main__":
    run()

