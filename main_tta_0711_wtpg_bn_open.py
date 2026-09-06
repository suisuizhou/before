#!/usr/bin/env python3
"""WTPG 0711 comparison with BN affine/statistics updates enabled.

The WTPG robust Source checkpoint and final target hyperparameters are kept
unchanged.  This runner only changes the target parameter policy to the
historical generic 0711 behavior: BN affine parameters are trainable and the
student runs in train mode so BN running statistics adapt online.  It uses a
separate entry point and never overwrites strict adapter-only results.
"""

from pathlib import Path

import hydra
import omegaconf
import torch
from omegaconf import open_dict

import main_tta_0711 as generic
import main_tta_0711_wtpg_strict as wtpg
from Lib.model import get_model
from Lib.wtpg_saliency_evidence import contiguous_saliency_masks


FULL_TRAINER = generic.EVMT0711Trainer


class WTPGBNOpenTrainer(wtpg.WTPG0711Trainer):
    """WTPG strict stream with generic full-teacher EMA and adaptive BN."""

    def initialize_models(self):
        # Keep strict WTPG memory/physical metadata initialization, then add a
        # full EMA teacher.  The strict adaptation EMA remains unused.
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
        print("[WTPG BN-OPEN] full EMA teacher enabled")

    def configure_trainable_parameters(self):
        return FULL_TRAINER.configure_trainable_parameters(self)

    def _set_student_adaptation_mode(self):
        # train() updates BN running statistics; the shared method disables
        # Dropout while leaving BN in training mode.
        return FULL_TRAINER._set_student_adaptation_mode(self)

    @torch.no_grad()
    def _ema_forward(self, x):
        self.teacher.eval()
        return self.forward_parts(self.teacher, x)

    @torch.no_grad()
    def ema_update_teacher(self):
        # Generic full-network EMA, including floating buffers (BN statistics).
        return FULL_TRAINER.ema_update_teacher(self)

    def build_evidence_masks(self, pseudo_labels, batch_metadata=None):
        # Preserve WTPG's saliency-band evidence, but obtain gradients from
        # the full EMA teacher rather than the adaptation-only carrier.
        if not hasattr(self, "_wtpg_evidence_input"):
            raise RuntimeError("WTPG evidence input is unavailable")
        with torch.enable_grad():
            evidence_input = (
                self._wtpg_evidence_input.detach().clone().requires_grad_(True)
            )
            self.teacher.eval()
            _, logits = self.forward_parts(self.teacher, evidence_input)
            selected = logits.gather(1, pseudo_labels.view(-1, 1)).sum()
            gradient = torch.autograd.grad(
                selected, evidence_input, only_inputs=True
            )[0]
        saliency = (gradient.abs() * evidence_input.detach().abs()).detach()
        return contiguous_saliency_masks(
            saliency,
            pseudo_labels,
            bands=int(generic._get(self.tcfg, "saliency_bands", 8)),
            half_width=int(generic._get(self.tcfg, "saliency_half_width", 3)),
            max_mask_ratio=float(generic._get(self.tcfg, "max_mask_ratio", 0.18)),
        )


def prepare_config(cfg: omegaconf.DictConfig) -> None:
    wtpg.prepare_wtpg_config(cfg)
    with open_dict(cfg):
        # The requested WTPG experiment is the previously frozen four-domain
        # split: retain D0..D3 (20/25/30/35 Hz) and exclude D4..D7.
        cfg.Dataset.TL_list = [0, 1, 2, 3]
        cfg.wtpg_checkpoint_root = "TTA_Model_WTPG_STRICT_V2"
        cfg.source_variant = "robust"
        cfg.wtpg_candidate_id = "bn_open_current_source"
        cfg.wtpg_config_sha256 = "bn_open_current_source"
        cfg.hust_config_sha256 = cfg.wtpg_config_sha256
        cfg.hust_candidate_id = cfg.wtpg_candidate_id
        cfg.Opt.lr_tar = 0.12
        cfg.Opt.weight_decay_tar = 1e-4
        cfg.TTA0711.mode = "full"
        cfg.TTA0711.passes = 1
        cfg.TTA0711.adapter_lr_scale = 8.0
        cfg.TTA0711.warp_lr_scale = 1.0
        cfg.TTA0711.ema_beta = 0.98
        cfg.TTA0711.warmup_batches = 5
        cfg.TTA0711.aux_ramp_batches = 5
        cfg.TTA0711.min_reliability = 0.12
        cfg.TTA0711.lambda_mt = 0.02
        cfg.TTA0711.lambda_pcl = 0.02
        cfg.TTA0711.lambda_ncl = 0.01
        cfg.TTA0711.memory_per_class = 64
        cfg.TTA0711.pcl_temperature = 0.10
        cfg.TTA0711.ncl_temperature = 0.10
        cfg.TTA0711.update_bn_affine = True
        cfg.stream_seed = 2025
        cfg.process_wandb = False


@hydra.main(version_base=None, config_path="./Configs", config_name="defaults")
def run(cfg: omegaconf.DictConfig):
    prepare_config(cfg)
    original = wtpg.base.Strict0711ResNetTrainer
    wtpg.base.Strict0711ResNetTrainer = WTPGBNOpenTrainer
    try:
        return wtpg.base.run.__wrapped__(cfg)
    finally:
        wtpg.base.Strict0711ResNetTrainer = original


if __name__ == "__main__":
    run()
