#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""0711-Full-ViT-Prompt under the strict single-pass PU4D protocol.

The validated strict 0711 implementation remains the algorithmic base:
EMA multi-view teacher, PU4D physical fault evidence, reliability routing,
evidence memory, PCL/NCL, reliability-weighted SEM/MT, causal memory update,
and pre-update strict Online evaluation.  Protocol A changes only the model
carrier from ResNet18 to ViT1D and expands the light-weight adaptation set to
prompt_embed + band_scale + band_bias + warp_ctrl.
"""
from __future__ import annotations

from pathlib import Path

import hydra
import omegaconf
from omegaconf import open_dict
import torch

try:
    import main_tta_0711_strict_randomstream as base0711
except ImportError as exc:
    raise ImportError(
        "main_tta_0711_strict_randomstream.py is required in the project root."
    ) from exc

from Lib.vit_protocol_a import (
    apply_vit_cfg,
    method_checkpoint_dir,
    select_trainable_parameters,
)


class ViT0711FullTrainer(base0711.Strict0711ResNetTrainer):
    """Strict 0711 Full with ViT Prompt + Adapter + F-Warp PEFT."""

    def _checkpoint_path(self):
        src = int(self.cfg.Dataset.TL_Task[0])
        ckpt_dir = Path(
            getattr(
                self.cfg,
                "source_ckpt_dir",
                str(method_checkpoint_dir("0711_FULL_VIT", "PU4D", src, self.cfg.seed_run)),
            )
        )
        model_name = f"ViT1D{int(self.cfg.seed_run)}fft_Linear.pt"
        best = ckpt_dir / ("best_source_" + model_name)
        final = ckpt_dir / model_name
        if best.exists():
            return best
        if final.exists():
            return final
        raise FileNotFoundError(f"0711-Full-ViT source checkpoint not found under {ckpt_dir}")

    def initialize_models(self):
        # Strict base loads the checkpoint, enables Adapter/F-Warp, creates the
        # evidence memory and physical masks. Rebuild only the lightweight EMA
        # so Prompt participates in the same temporal teacher as Adapter/F-Warp.
        super().initialize_models()
        self.adaptation_ema = base0711.AdaptationEMA(
            self.student,
            name_tokens=("prompt_embed", "band_scale", "band_bias", "warp_ctrl"),
        )
        prompt = getattr(self.student[0], "prompt_embed", None)
        self.prompt_anchor = prompt.detach().clone() if prompt is not None else None
        print("[VIT STRICT 0711] EMA parameters:")
        for name in self.adaptation_ema.names:
            print("  ", name)
        if self.prompt_anchor is not None:
            print(f"[PROMPT ANCHOR] shape={tuple(self.prompt_anchor.shape)}")

    def configure_trainable_parameters(self):
        selected = select_trainable_parameters(self.student, method="0711_full")
        prompt_params = []
        adapter_params = []
        warp_params = []
        for name, parameter in self.student.named_parameters():
            if not parameter.requires_grad:
                continue
            if "prompt_embed" in name:
                prompt_params.append(parameter)
            elif "warp_ctrl" in name:
                warp_params.append(parameter)
            elif "band_scale" in name or "band_bias" in name:
                adapter_params.append(parameter)

        base_lr = float(self.cfg.Opt.lr_tar)
        weight_decay = float(self.cfg.Opt.weight_decay_tar)
        groups = []
        if prompt_params:
            groups.append(
                {
                    "params": prompt_params,
                    "lr": base_lr * float(base0711._get(self.tcfg, "prompt_lr_scale", 1.0)),
                    "weight_decay": weight_decay,
                }
            )
        if adapter_params:
            groups.append(
                {
                    "params": adapter_params,
                    "lr": base_lr * float(base0711._get(self.tcfg, "adapter_lr_scale", 1.0)),
                    "weight_decay": weight_decay,
                }
            )
        if warp_params:
            groups.append(
                {
                    "params": warp_params,
                    "lr": base_lr * float(base0711._get(self.tcfg, "warp_lr_scale", 0.1)),
                    "weight_decay": weight_decay,
                }
            )
        if not groups:
            raise RuntimeError("No Prompt/Adapter/F-Warp parameters found for 0711-Full-ViT")
        print("[VIT STRICT TRAINABLE PARAMETERS]")
        for name in selected:
            print("  ", name)
        return torch.optim.AdamW(groups)

    def adapter_reg_loss(self):
        # PDF Lreg permits Adapter/Warp/Prompt regularization. The strict base
        # already weights adapter_reg_loss by lambda_adapter; append a source
        # prompt anchor term here so no extra algorithmic branch is introduced.
        loss = super().adapter_reg_loss()
        if self.prompt_anchor is None:
            return loss
        prompt = getattr(self.student[0], "prompt_embed", None)
        if prompt is None:
            return loss
        prompt_scale = float(base0711._get(self.tcfg, "prompt_reg_scale", 1.0))
        return loss + prompt_scale * (prompt - self.prompt_anchor).pow(2).mean()

    def adapt(self):
        result = super().adapt()
        src, tar = [int(v) for v in self.cfg.Dataset.TL_Task]
        print(
            f"[RESULT] method=0711_FULL_VIT task=[{src},{tar}] "
            f"before={float(result['before']):.4f} online={float(result['online']):.4f} "
            f"post={float(result['post_stream']):.4f} online_f1=NA post_f1=NA "
            f"batch_ms=NA peak_mb=NA"
        )
        return result


def force_protocol_a(cfg):
    with open_dict(cfg):
        cfg.Dataset.input_kind = "fft"
        cfg.Dataset.norm_kind = "mean-std"
        cfg.batch_size = int(getattr(cfg, "batch_size", 128))
        cfg.num_workers = int(getattr(cfg, "num_workers", 4))
        cfg.stream_seed = int(getattr(cfg, "stream_seed", 2025))
        cfg.process_wandb = False
        apply_vit_cfg(cfg, use_spectral_adapter=True)

        if not hasattr(cfg, "TTA0711"):
            cfg.TTA0711 = {}
        cfg.TTA0711.mode = "full"
        cfg.TTA0711.passes = 1
        cfg.TTA0711.stream_seed = cfg.stream_seed
        cfg.TTA0711.update_prompt = True
        cfg.TTA0711.update_bn_affine = False
        cfg.TTA0711.use_frequency_warp = True
        cfg.TTA0711.prompt_lr_scale = float(getattr(cfg.TTA0711, "prompt_lr_scale", 1.0))
        cfg.TTA0711.warp_knots = int(getattr(cfg.TTA0711, "warp_knots", 16))
        cfg.TTA0711.max_warp = float(getattr(cfg.TTA0711, "max_warp", 2.0))
        cfg.TTA0711.adapter_lr_scale = float(getattr(cfg.TTA0711, "adapter_lr_scale", 1.0))
        cfg.TTA0711.warp_lr_scale = float(getattr(cfg.TTA0711, "warp_lr_scale", 0.1))
        cfg.TTA0711.lambda_adapter = float(getattr(cfg.TTA0711, "lambda_adapter", 1e-3))
        cfg.TTA0711.lambda_warp = float(getattr(cfg.TTA0711, "lambda_warp", 2e-4))
        cfg.TTA0711.warp_smooth_weight = float(getattr(cfg.TTA0711, "warp_smooth_weight", 2.0))
        cfg.TTA0711.prompt_reg_scale = float(getattr(cfg.TTA0711, "prompt_reg_scale", 1.0))

    # Explicit aliases make the experiment contract easy to audit in the file.
    mode = "full"
    update_prompt = True
    update_bn_affine = False
    use_frequency_warp = True
    use_spectral_adapter = True
    assert mode == cfg.TTA0711.mode
    assert update_prompt == cfg.TTA0711.update_prompt
    assert update_bn_affine == cfg.TTA0711.update_bn_affine
    assert use_frequency_warp == cfg.TTA0711.use_frequency_warp
    assert use_spectral_adapter == cfg.Model.use_spectral_adapter


@hydra.main(version_base=None, config_path="./Configs", config_name="defaults")
def run(cfg: omegaconf.DictConfig):
    force_protocol_a(cfg)
    # The base Hydra loop instantiates Strict0711ResNetTrainer. Patch exactly
    # that symbol during the call, then restore it even if a run fails.
    original_cls = base0711.Strict0711ResNetTrainer
    base0711.Strict0711ResNetTrainer = ViT0711FullTrainer
    try:
        base0711.run.__wrapped__(cfg)
    finally:
        base0711.Strict0711ResNetTrainer = original_cls


if __name__ == "__main__":
    run()
