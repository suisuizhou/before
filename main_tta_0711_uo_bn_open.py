#!/usr/bin/env python3
"""UO 0711 comparison with BN affine/statistics updates enabled.

This runner intentionally lives beside the strict adapter-only runner and uses
an independent log/checkpoint namespace.  It reuses the current UO robust
Source checkpoints and the current best target hyperparameters; only the BN
policy is changed to the historical generic 0711 behavior.
"""

import os
from pathlib import Path

import hydra
import omegaconf
from omegaconf import open_dict

import main_tta_0711 as base
import main_tta_0711_uo_adapter_only as uo_base

# Keep an unpatched reference: run() temporarily replaces
# base.EVMT0711Trainer with the rooted subclass below.
GENERIC_TRAINER = base.EVMT0711Trainer


class UOBNOpenTrainer(uo_base.UOAdapterOnlyTrainer):
    """0711 target adaptation with BN affine and running statistics enabled."""

    def configure_trainable_parameters(self):
        # The generic implementation adds BN affine parameters together with
        # the spectral adapter and F-Warp carriers when update_bn_affine=True.
        return GENERIC_TRAINER.configure_trainable_parameters(self)

    def _set_student_adaptation_mode(self):
        # train() is required for BN running mean/variance updates; the shared
        # implementation immediately disables Dropout to keep the comparison
        # deterministic apart from the fixed random stream.
        return GENERIC_TRAINER._set_student_adaptation_mode(self)


class RootedUOBNOpenTrainer(UOBNOpenTrainer):
    """Resolve the current robust Source checkpoints without overwriting them."""

    def __init__(self, cfg, run_obj=None):
        super().__init__(cfg, run_obj)
        with open_dict(self.cfg):
            root = str(getattr(self.cfg, "source_checkpoint_root", "./TTA_Model_UO_ROBUST_TUNED"))
            self.cfg.save_model_path = root
            self.cfg.model_path = (
                Path(root)
                / (str(self.cfg.Dataset.data_name) + str(self.cfg.Opt.lr_src))
                / (str(self.cfg.Dataset.TL_Task) + "_Task")
            )
            self.cfg.model_path.mkdir(parents=True, exist_ok=True)


def prepare_config(cfg):
    uo_base.prepare_uo_config(cfg)
    with open_dict(cfg):
        cfg.source_checkpoint_root = os.environ.get(
            "UO_SOURCE_ROOT", "./TTA_Model_UO_ROBUST_TUNED"
        )
        # Current best UO target hyperparameters; only BN policy differs.
        cfg.Opt.lr_tar = 0.20
        cfg.Opt.weight_decay_tar = 1e-4
        cfg.TTA0711.lambda_mt = 0.20
        cfg.TTA0711.lambda_pcl = 0.05
        cfg.TTA0711.lambda_ncl = 0.05
        cfg.TTA0711.ema_beta = 0.98
        cfg.TTA0711.adapter_lr_scale = 2.0
        cfg.TTA0711.warp_lr_scale = 0.40
        cfg.TTA0711.mode = "full"
        cfg.TTA0711.passes = 1
        cfg.TTA0711.warmup_passes = 0
        cfg.TTA0711.aux_ramp_passes = 1
        cfg.TTA0711.update_bn_affine = True
        cfg.TTA0711.use_frequency_warp = True
        cfg.TTA0711.min_reliability = 0.0
        cfg.stream_seed = 2025
        cfg.process_wandb = False
        cfg.bn_open_experiment = True


@hydra.main(version_base=None, config_path="./Configs", config_name="defaults")
def run(cfg: omegaconf.DictConfig):
    prepare_config(cfg)
    original = base.EVMT0711Trainer
    base.EVMT0711Trainer = RootedUOBNOpenTrainer
    try:
        return base.run.__wrapped__(cfg)
    finally:
        base.EVMT0711Trainer = original


if __name__ == "__main__":
    run()
