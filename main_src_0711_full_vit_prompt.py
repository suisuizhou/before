#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""0711-Full-ViT-Prompt source training wrapper for Protocol A.

Reuses the validated SDE source trainer and keeps its SSP/SDE/mixup recipe,
while fixing the agreed ViT1D architecture, prompt length and 60 epochs.
Target Adapter/F-Warp parameters are not trained here; they start from identity
at test time, matching the established 0711-Full workflow.
"""
from __future__ import annotations

import hydra
import omegaconf
from omegaconf import open_dict

try:
    import main_Src_SDE_STABLE as base_src
except ImportError as exc:
    raise ImportError("main_Src_SDE_STABLE.py is required in the project root.") from exc

from Lib.vit_protocol_a import apply_vit_cfg


def force_protocol_a_source(cfg):
    with open_dict(cfg):
        cfg.Dataset.input_kind = "fft"
        cfg.Dataset.norm_kind = "mean-std"
        cfg.src_epoch = 60
        cfg.batch_size = 64
        cfg.num_workers = int(getattr(cfg, "num_workers", 4))
        cfg.process_wandb = False

        # Source learns the ViT+Prompt representation. Spectral Adapter/F-Warp
        # remain target-time identity parameters and are enabled by the TTA runner.
        apply_vit_cfg(cfg, use_spectral_adapter=False)
        cfg.prompt_len_src = 3

        cfg.label_smoothing = 0.1
        cfg.mixup_alpha = 0.2
        cfg.mixup_prob = 0.5

        cfg.use_ssp_lite = True
        cfg.ssp_style_prob = 0.7
        cfg.ssp_style_strength = 0.15
        cfg.ssp_style_knots = 8
        cfg.ssp_lambda_style = 0.5
        cfg.ssp_lambda_cons = 0.05

        cfg.use_sde_lite = True
        cfg.sde_warp_prob = 0.7
        cfg.sde_warp_knots = 16
        cfg.sde_warp_max = 2.0
        cfg.sde_lambda_warp = 0.5
        cfg.sde_lambda_style_warp = 0.25
        cfg.sde_lambda_cons = 0.03
        cfg.sde_lambda_feat = 0.02
        cfg.sde_use_style_warp = True

        cfg.Opt.lr_src = 1e-3
        cfg.Opt.weight_decay_src = 1e-4
        cfg.PR = 0

    # Explicit audit constants retained in source for experiment inspection.
    src_epoch = 60
    prompt_len_src = 3
    use_ssp_lite = True
    use_sde_lite = True
    mixup_prob = 0.5
    assert src_epoch == cfg.src_epoch
    assert prompt_len_src == cfg.prompt_len_src
    assert use_ssp_lite == cfg.use_ssp_lite
    assert use_sde_lite == cfg.use_sde_lite
    assert mixup_prob == cfg.mixup_prob


@hydra.main(version_base=None, config_path="./Configs", config_name="defaults")
def run(cfg: omegaconf.DictConfig):
    force_protocol_a_source(cfg)
    base_src.run.__wrapped__(cfg)


if __name__ == "__main__":
    run()
