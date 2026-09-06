#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""DtCC-ViT-Prompt source training wrapper for Protocol A.

Reuses the project's stable source trainer but forces the DtCC source recipe:
ViT1D+Prompt, label smoothing, 60 epochs, no mixup/SSP/SDE augmentation.
"""
from __future__ import annotations

import hydra
import omegaconf
from omegaconf import open_dict

try:
    import main_Src_stronger_SSP_lite_STABLE as base_src
except ImportError as exc:
    raise ImportError(
        "main_Src_stronger_SSP_lite_STABLE.py is required in the project root."
    ) from exc

from Lib.vit_protocol_a import apply_vit_cfg


def force_protocol_a_source(cfg):
    with open_dict(cfg):
        cfg.Dataset.input_kind = "fft"
        cfg.Dataset.norm_kind = "mean-std"
        cfg.src_epoch = 60
        cfg.batch_size = 64
        cfg.num_workers = int(getattr(cfg, "num_workers", 4))
        cfg.process_wandb = False

        apply_vit_cfg(cfg, use_spectral_adapter=False)
        cfg.prompt_len_src = 3

        cfg.label_smoothing = 0.1
        cfg.mixup_alpha = 0.0
        cfg.mixup_prob = 0.0
        cfg.use_ssp_lite = False
        cfg.ssp_style_prob = 0.0
        cfg.ssp_lambda_style = 0.0
        cfg.ssp_lambda_cons = 0.0

        cfg.Opt.lr_src = 1e-3
        cfg.Opt.weight_decay_src = 1e-4
        cfg.Opt.lr_scheduler = getattr(cfg.Opt, "lr_scheduler", "fix")
        cfg.PR = 0

    # Explicit audit constants retained in source for experiment inspection.
    src_epoch = 60
    prompt_len_src = 3
    label_smoothing = 0.1
    mixup_prob = 0.0
    use_ssp_lite = False
    assert src_epoch == cfg.src_epoch
    assert prompt_len_src == cfg.prompt_len_src
    assert label_smoothing == cfg.label_smoothing
    assert mixup_prob == cfg.mixup_prob
    assert use_ssp_lite == cfg.use_ssp_lite


@hydra.main(version_base=None, config_path="./Configs", config_name="defaults")
def run(cfg: omegaconf.DictConfig):
    force_protocol_a_source(cfg)
    base_src.run.__wrapped__(cfg)


if __name__ == "__main__":
    run()
