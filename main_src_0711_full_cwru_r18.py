#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""0711-Full ResNet18 source training on balanced CWRU, 60 epochs."""
from __future__ import annotations
import hydra,omegaconf
from omegaconf import open_dict
import Dataset
from Dataset.CWRU import CWRU as CWRUDataset
Dataset.CWRU=CWRUDataset
try:
    import main_Src_SDE_STABLE as base_src
except ImportError as exc:
    raise ImportError("main_Src_SDE_STABLE.py is required in the project root") from exc


def force_cwru_source(cfg):
    with open_dict(cfg):
        cfg.Dataset.data_name = "CWRU"
        cfg.Dataset.TL_list = [0,1,2,3]
        cfg.Dataset.input_kind = "fft"
        cfg.Dataset.norm_kind = "mean-std"
        cfg.src_epoch = 60
        cfg.source_only_mode = True
        cfg.batch_size = 128
        cfg.num_workers = int(getattr(cfg,"num_workers",4))
        cfg.process_wandb = False
        cfg.Model.bottleneck_num = 128
        cfg.Model.use_spectral_adapter = False
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

@hydra.main(version_base=None,config_path="./Configs",config_name="defaults")
def run(cfg:omegaconf.DictConfig):
    force_cwru_source(cfg)
    base_src.run.__wrapped__(cfg)
if __name__=="__main__": run()
