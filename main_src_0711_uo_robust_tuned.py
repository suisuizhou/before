#!/usr/bin/env python3
"""Isolated UO 0711 source training with tunable robust spectral augmentation.

The original UO source checkpoints are never overwritten: this runner redirects
all checkpoints to ``TTA_Model_UO_ROBUST_TUNED``.
"""
import hydra
import omegaconf
import os
import torch
from omegaconf import open_dict

import main_Src_SDE_STABLE as base


class UORobustSourceTrainer(base.SDE_SourceTrainer):
    def train(self):
        # The shared source entry hard-codes ./TTA_Model. Redirect only this
        # isolated experiment to a separate checkpoint family.
        with open_dict(self.cfg):
            self.cfg.save_model_path = os.environ.get(
                "UO_SOURCE_ROOT", "./TTA_Model_UO_ROBUST_TUNED"
            )
        return super().train()


def prepare_uo_source_config(cfg):
    with open_dict(cfg):
        cfg.Dataset.data_name = "UO"
        cfg.Dataset.data_path = "Dataset/UO_EXPERIMENT_20260903/UO_CACHE_CH1_256"
        cfg.Dataset.TL_list = [0, 1, 2, 3]
        cfg.Dataset.input_kind = "fft"
        cfg.Model.model_name = "ResNet18_1D_SDE"
        cfg.Model.use_spectral_adapter = False
        cfg.Model.band_num = 256
        cfg.Model.input_len = 512
        cfg.Model.bottleneck = True
        cfg.Model.bottleneck_num = 128
        cfg.Model.model_type = "linear"
        cfg.Opt.lr_src = 0.0008
        cfg.Opt.weight_decay_src = 0.0001
        cfg.src_epoch = 15
        cfg.batch_size = 256
        cfg.num_workers = 4
        cfg.label_smoothing = 0.05
        cfg.mixup_alpha = 0.2
        cfg.mixup_prob = 0.5
        cfg.prompt_len_src = 3

        # Moderate, class-preserving spectral robustness profile.
        cfg.use_ssp_lite = True
        cfg.ssp_style_prob = 0.8
        cfg.ssp_style_strength = 0.15
        cfg.ssp_style_knots = 8
        cfg.ssp_lambda_style = 0.35
        cfg.ssp_lambda_cons = 0.03
        cfg.use_sde_lite = True
        cfg.sde_warp_prob = 0.8
        cfg.sde_warp_knots = 16
        cfg.sde_warp_max = 1.75
        cfg.sde_lambda_warp = 0.35
        cfg.sde_lambda_style_warp = 0.20
        cfg.sde_lambda_cons = 0.03
        cfg.sde_lambda_feat = 0.02
        cfg.sde_use_style_warp = True
        cfg.save_every = 0
        cfg.PR = 0
        cfg.process_wandb = False

        # Optional profile overrides are supplied through environment variables
        # so each experiment can be launched without changing this runner.
        for key, cast in (
            ("UO_STYLE_PROB", float), ("UO_STYLE_STRENGTH", float),
            ("UO_WARP_PROB", float), ("UO_WARP_MAX", float),
            ("UO_MIXUP_PROB", float), ("UO_LABEL_SMOOTHING", float),
            ("UO_AMP_MIN", float), ("UO_AMP_MAX", float),
            ("UO_NOISE_STD", float),
            ("UO_LAMBDA_STYLE", float), ("UO_LAMBDA_WARP", float),
            ("UO_LAMBDA_STYLE_WARP", float), ("UO_LAMBDA_CONS", float),
            ("UO_LAMBDA_FEAT", float),
        ):
            if key in os.environ:
                field = {
                    "UO_STYLE_PROB": "ssp_style_prob",
                    "UO_STYLE_STRENGTH": "ssp_style_strength",
                    "UO_WARP_PROB": "sde_warp_prob",
                    "UO_WARP_MAX": "sde_warp_max",
                    "UO_MIXUP_PROB": "mixup_prob",
                    "UO_LABEL_SMOOTHING": "label_smoothing",
                    "UO_AMP_MIN": "source_amp_min",
                    "UO_AMP_MAX": "source_amp_max",
                    "UO_NOISE_STD": "source_noise_std",
                    "UO_LAMBDA_STYLE": "ssp_lambda_style",
                    "UO_LAMBDA_WARP": "sde_lambda_warp",
                    "UO_LAMBDA_STYLE_WARP": "sde_lambda_style_warp",
                    "UO_LAMBDA_CONS": "sde_lambda_cons",
                    "UO_LAMBDA_FEAT": "sde_lambda_feat",
                }[key]
                setattr(cfg, field, cast(os.environ[key]))


@hydra.main(version_base=None, config_path="./Configs", config_name="defaults")
def run(cfg: omegaconf.DictConfig):
    prepare_uo_source_config(cfg)
    original_style = base.base.spectral_style_augment
    amp_min = float(getattr(cfg, "source_amp_min", 1.0))
    amp_max = float(getattr(cfg, "source_amp_max", 1.0))
    noise_std = float(getattr(cfg, "source_noise_std", 0.0))

    def profile_style_augment(x, strength=0.15, knots=8, prob=0.7):
        out = original_style(x, strength=strength, knots=knots, prob=prob)
        if amp_min != 1.0 or amp_max != 1.0:
            scale = out.new_empty(out.shape[0], 1, 1).uniform_(amp_min, amp_max)
            if out.ndim == 2:
                out = out * scale.squeeze(-1)
            elif out.ndim == 3 and out.shape[-1] == 1:
                out = out * scale.transpose(1, 2)
            else:
                out = out * scale
        if noise_std > 0:
            out = out + torch.randn_like(out) * noise_std
        return out

    base.base.spectral_style_augment = profile_style_augment
    # main_Src_SDE_STABLE.run delegates to the imported stronger trainer
    # module through its ``base`` alias.
    original = base.base.SourceTrainer
    base.base.SourceTrainer = UORobustSourceTrainer
    try:
        return base.run.__wrapped__(cfg)
    finally:
        base.base.SourceTrainer = original
        base.base.spectral_style_augment = original_style


if __name__ == "__main__":
    run()
