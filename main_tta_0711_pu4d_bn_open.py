#!/usr/bin/env python3
"""PU4D 0711 BN-open experiment.

This entry point is intentionally separate from the strict 0711 runner.  It
keeps the strict PU4D stream, physical evidence, and source-specific
checkpoints, but enables the historical generic 0711 BN policy:

* BN affine parameters are optimized;
* BN running mean/variance are updated online (the student is in train mode);
* a full EMA teacher is used, including BN buffers;
* all other target settings are the final PU4D strict-tuning profile.

Existing strict-online code and results are not modified.
"""

from pathlib import Path

import hydra
import omegaconf
import torch
from omegaconf import open_dict

import main_tta_0711 as generic
import main_tta_0711_strict_online as strict
from Lib.model import get_model


FULL_TRAINER = generic.EVMT0711Trainer


class PU4DBNOpenTrainer(strict.Strict0711ResNetTrainer):
    """Strict PU4D evidence/stream with BN affine and statistics open."""

    def initialize_models(self):
        # Preserve strict PU4D physical configuration and source checkpoint
        # resolution, then add a full teacher for BN-buffer EMA.
        super().initialize_models()
        ckpt_path = self._checkpoint_path()
        self.teacher = get_model(
            num_classes=self.num_classes, cfg=self.cfg, **self.cfg.Model
        ).to(self.device)
        self.load_source_checkpoint(self.teacher, ckpt_path)
        self.enable_frequency_warp(self.teacher[0])
        self.teacher.load_state_dict(self.student.state_dict(), strict=True)
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        self.bn_anchor = self._capture_bn_anchor(self.student)
        print("[PU4D BN-OPEN] full EMA teacher enabled")

    def configure_trainable_parameters(self):
        # Generic 0711 grouping adds BN affine parameters when the flag is on,
        # while retaining adapter and F-Warp groups.
        return FULL_TRAINER.configure_trainable_parameters(self)

    def _set_student_adaptation_mode(self):
        # train() updates running statistics; generic implementation disables
        # Dropout, keeping the comparison deterministic apart from BN state.
        return FULL_TRAINER._set_student_adaptation_mode(self)

    @torch.no_grad()
    def _ema_forward(self, x):
        self.teacher.eval()
        return self.forward_parts(self.teacher, x)

    @torch.no_grad()
    def ema_update_teacher(self):
        # Full-network EMA also tracks floating BN buffers.
        return FULL_TRAINER.ema_update_teacher(self)


def prepare_config(cfg: omegaconf.DictConfig) -> None:
    with open_dict(cfg):
        cfg.Dataset.data_name = "PU4D"
        cfg.Dataset.data_path = "/home/std04/Projects/DtCC_FOA_ViT_v1/DtCC_FOA_V1/Dataset/PU4D_CACHE"
        cfg.Dataset.TL_list = [0, 1, 2, 3]
        cfg.Dataset.input_kind = "fft"
        cfg.Model.model_name = "ResNet18_1D_SDE"
        cfg.Model.model_type = "linear"
        cfg.Model.use_spectral_adapter = True
        cfg.Model.band_num = 256
        cfg.Model.adapter_delta = 0.1
        cfg.Opt.lr_src = 0.001
        cfg.Opt.lr_tar = 0.024
        cfg.Opt.weight_decay_tar = 0.001
        cfg.batch_size = 128
        cfg.num_workers = 4
        cfg.gpu_id = "0"
        cfg.seed_runs = [2025]
        cfg.process_wandb = False

        # Final PU4D 0711 strict-tuning profile.
        if not hasattr(cfg, "TTA0711"):
            cfg.TTA0711 = {}
        cfg.TTA0711.mode = "full"
        cfg.TTA0711.passes = 1
        cfg.TTA0711.stream_seed = 2025
        cfg.TTA0711.alpha = 2.0
        cfg.TTA0711.eta = 0.05
        cfg.TTA0711.teacher_temp = 1.0
        cfg.TTA0711.mt_warmup_scale = 0.5
        cfg.TTA0711.view_style_strength = 0.05
        cfg.TTA0711.view_style_knots = 8
        cfg.TTA0711.view_warp_max = 0.5
        cfg.TTA0711.view_warp_knots = 8
        cfg.TTA0711.view_gain_strength = 0.03
        cfg.TTA0711.view_baseline_strength = 0.02
        cfg.TTA0711.view_noise_std = 0.01
        cfg.TTA0711.view_gamma = 5.0
        cfg.TTA0711.evidence_interval = 1
        cfg.TTA0711.evidence_metric = "margin"
        cfg.TTA0711.sampling_rate_hz = 64000
        cfg.TTA0711.fft_size = 1024
        cfg.TTA0711.spectrum_length = 512
        cfg.TTA0711.physical_harmonics = 8
        cfg.TTA0711.outer_sideband_orders = [0, 1]
        cfg.TTA0711.inner_sideband_orders = [0, 1, 2]
        cfg.TTA0711.mask_sigma_bins = 1.0
        cfg.TTA0711.physical_background_width = 7
        cfg.TTA0711.max_mask_ratio = 0.18
        cfg.TTA0711.mask_activity_threshold = 0.1
        cfg.TTA0711.exclude_dc = True
        cfg.TTA0711.min_pcl_classes = 8
        cfg.TTA0711.ncl_neighbors = 3
        cfg.TTA0711.min_ncl_classes = 16
        cfg.TTA0711.min_ncl_entries = 64
        cfg.TTA0711.use_frequency_warp = True
        cfg.TTA0711.warp_knots = 16
        cfg.TTA0711.max_warp = 2.0
        cfg.TTA0711.warp_smooth_weight = 2.0
        cfg.TTA0711.adapter_lr_scale = 1.0
        cfg.TTA0711.lambda_adapter = 0.001
        cfg.TTA0711.lambda_warp = 0.0002
        cfg.TTA0711.log_interval = 25
        cfg.TTA0711.warp_lr_scale = 0.2
        cfg.TTA0711.ema_beta = 0.995
        cfg.TTA0711.warmup_batches = 10
        cfg.TTA0711.aux_ramp_batches = 20
        cfg.TTA0711.min_reliability = 0.2
        cfg.TTA0711.lambda_mt = 0.02
        cfg.TTA0711.lambda_pcl = 0.02
        cfg.TTA0711.lambda_ncl = 0.01
        cfg.TTA0711.memory_per_class = 32
        cfg.TTA0711.pcl_temperature = 0.2
        cfg.TTA0711.ncl_temperature = 0.2
        cfg.TTA0711.update_bn_affine = True


@hydra.main(version_base=None, config_path="./Configs", config_name="defaults")
def run(cfg: omegaconf.DictConfig):
    prepare_config(cfg)
    original = strict.Strict0711ResNetTrainer
    strict.Strict0711ResNetTrainer = PU4DBNOpenTrainer
    try:
        return strict.run.__wrapped__(cfg)
    finally:
        strict.Strict0711ResNetTrainer = original


if __name__ == "__main__":
    run()
