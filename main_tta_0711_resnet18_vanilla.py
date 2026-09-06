#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict 0711 PU4D TTA from the exact same vanilla ResNet18 source checkpoints as DtCC."""
from __future__ import annotations

from pathlib import Path

import hydra
import omegaconf

import main_tta_0711_strict_randomstream as base0711
from Lib.pu4d_vanilla_protocol import resolve_vanilla_checkpoint, vanilla_checkpoint_dir


class Vanilla0711Trainer(base0711.Strict0711ResNetTrainer):
    def _checkpoint_path(self):
        root = Path(str(getattr(self.cfg, "vanilla_source_root", "TTA_Model_VANILLA")))
        source = int(self.cfg.Dataset.TL_Task[0])
        return resolve_vanilla_checkpoint(root, source, int(self.cfg.seed_run), self.cfg.model_name)


@hydra.main(version_base=None, config_path="./Configs", config_name="defaults")
def run(cfg: omegaconf.DictConfig):
    if not hasattr(cfg, "vanilla_source_root"):
        from omegaconf import open_dict
        with open_dict(cfg):
            cfg.vanilla_source_root = "TTA_Model_VANILLA"
    print("[0711 VANILLA SOURCE ROOT]", cfg.vanilla_source_root)
    original_cls = base0711.Strict0711ResNetTrainer
    base0711.Strict0711ResNetTrainer = Vanilla0711Trainer
    try:
        base0711.run.__wrapped__(cfg)
    finally:
        base0711.Strict0711ResNetTrainer = original_cls


if __name__ == "__main__":
    run()
