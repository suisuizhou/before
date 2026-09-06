#!/usr/bin/env python3
"""UO 0711 adapter-only target adaptation.

This is an isolated UO runner.  It keeps the generic 0711 reliability/EMA
pipeline, but freezes BN and all backbone/classifier parameters; only the
spectral adapter and F-Warp carrier are optimized.
"""

from pathlib import Path

import hydra
import omegaconf
import torch
import torch.nn as nn
from omegaconf import open_dict

from Lib.fixed_random_stream import make_fixed_random_stream_loader
import main_tta_0711 as base


class UOAdapterOnlyTrainer(base.EVMT0711Trainer):
    """0711 target adaptation with only spectral adapter and F-Warp trainable."""

    def setup(self):
        super().setup()
        self.target_dataloader = make_fixed_random_stream_loader(
            dataset=self.datasets["target_data"],
            batch_size=self.cfg.batch_size,
            seed=int(getattr(self.cfg, "stream_seed", self.cfg.seed_run)),
            num_workers=self.cfg.num_workers,
            drop_last=False,
            pin_memory=(self.device.type == "cuda"),
        )

    def configure_trainable_parameters(self):
        for parameter in self.student.parameters():
            parameter.requires_grad = False

        adapter_params = []
        warp_params = []
        trainable_names = []
        for name, parameter in self.student.named_parameters():
            if "band_scale" in name or "band_bias" in name:
                parameter.requires_grad = True
                adapter_params.append(parameter)
                trainable_names.append(name)
            elif "warp_ctrl" in name:
                parameter.requires_grad = True
                warp_params.append(parameter)
                trainable_names.append(name)

        if not adapter_params and not warp_params:
            raise RuntimeError("UO adapter-only runner found no adaptation carriers")

        base_lr = float(self.cfg.Opt.lr_tar)
        weight_decay = float(self.cfg.Opt.weight_decay_tar)
        groups = []
        if adapter_params:
            groups.append({
                "params": adapter_params,
                "lr": base_lr * float(base._get(self.tcfg, "adapter_lr_scale", 1.0)),
                "weight_decay": weight_decay,
            })
        if warp_params:
            groups.append({
                "params": warp_params,
                "lr": base_lr * float(base._get(self.tcfg, "warp_lr_scale", 0.2)),
                "weight_decay": weight_decay,
            })

        print("[UO ADAPTER-ONLY TRAINABLE PARAMETERS]")
        for name in trainable_names:
            print("  ", name)
        return torch.optim.AdamW(groups)

    def _set_student_adaptation_mode(self):
        # Keep source BN running statistics and affine parameters fixed.
        self.student.eval()
        for module in self.student.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
            elif isinstance(module, nn.Dropout):
                module.eval()


def prepare_uo_config(cfg):
    with open_dict(cfg):
        cfg.Dataset.data_name = "UO"
        cfg.Dataset.data_path = "Dataset/UO_EXPERIMENT_20260903/UO_CACHE_CH1_256"
        cfg.Dataset.TL_list = [0, 1, 2, 3]
        cfg.Dataset.input_kind = "fft"
        cfg.Model.model_name = "ResNet18_1D_SDE"
        cfg.Model.use_spectral_adapter = True
        cfg.Model.band_num = 256
        cfg.Model.input_len = 512
        cfg.Opt.lr_src = 0.0008
        cfg.batch_size = 256
        cfg.num_workers = 4
        cfg.stream_seed = 2025
        if not hasattr(cfg, "TTA0711"):
            cfg.TTA0711 = {}
        cfg.TTA0711.mode = "full"
        cfg.TTA0711.passes = 1
        cfg.TTA0711.warmup_passes = 0
        cfg.TTA0711.aux_ramp_passes = 1
        cfg.TTA0711.update_bn_affine = False
        cfg.TTA0711.use_frequency_warp = True


@hydra.main(version_base=None, config_path="./Configs", config_name="defaults")
def run(cfg: omegaconf.DictConfig):
    prepare_uo_config(cfg)
    original = base.EVMT0711Trainer
    base.EVMT0711Trainer = UOAdapterOnlyTrainer
    try:
        return base.run.__wrapped__(cfg)
    finally:
        base.EVMT0711Trainer = original


if __name__ == "__main__":
    run()
