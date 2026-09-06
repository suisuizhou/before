#!/usr/bin/env python3
"""UO strict 0711 adapter-only TTA with an explicit source checkpoint root."""
import hydra
import omegaconf
import os
from omegaconf import open_dict
from pathlib import Path

import main_tta_0711_uo_adapter_only as adapter_base
import main_tta_0711 as base


class RootedUOAdapterOnlyTrainer(adapter_base.UOAdapterOnlyTrainer):
    def __init__(self, cfg, run_obj=None):
        super().__init__(cfg, run_obj)
        with open_dict(self.cfg):
            root = str(getattr(self.cfg, "source_checkpoint_root", "./TTA_Model"))
            self.cfg.save_model_path = root
            self.cfg.model_path = (
                Path(root) / (str(self.cfg.Dataset.data_name) + str(self.cfg.Opt.lr_src))
                / (str(self.cfg.Dataset.TL_Task) + "_Task")
            )
            self.cfg.model_path.mkdir(parents=True, exist_ok=True)


@hydra.main(version_base=None, config_path="./Configs", config_name="defaults")
def run(cfg: omegaconf.DictConfig):
    adapter_base.prepare_uo_config(cfg)
    with open_dict(cfg):
        cfg.source_checkpoint_root = os.environ.get(
            "UO_SOURCE_ROOT", "./TTA_Model_UO_ROBUST_TUNED"
        )
    original = base.EVMT0711Trainer
    base.EVMT0711Trainer = RootedUOAdapterOnlyTrainer
    try:
        return base.run.__wrapped__(cfg)
    finally:
        base.EVMT0711Trainer = original


if __name__ == "__main__":
    run()
