#!/usr/bin/env python3
"""Ordinary source training for strict WTPG."""

import hydra
import omegaconf

from Lib.wtpg_source_training import prepare_wtpg_source_config, train_wtpg_source


@hydra.main(version_base=None, config_path="./Configs", config_name="defaults")
def run(cfg: omegaconf.DictConfig):
    source = prepare_wtpg_source_config(cfg)
    return train_wtpg_source(cfg, source, "ordinary")


if __name__ == "__main__":
    run()

