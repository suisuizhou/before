#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Strict single-pass ResNet implementation of the 0711 EVMT-DtCC scheme.

Differences from ``main_tta_0711_full_tuned.py``:
  * target stream is visited exactly once in a fixed-seed random order;
  * BN affine parameters and running statistics are frozen;
  * only band_scale, band_bias and warp_ctrl are optimized;
  * EMA stores only those adaptation parameters, not a full teacher network;
  * Teacher uses clean/style/warp/gain/baseline/noise task-preserving views;
  * fault-destructive views use PU4D physical BPFO/BPFI harmonics and sidebands;
  * current-batch samples enter memory only after losses and optimizer update;
  * Online Accuracy is measured from the pre-update prediction of each batch.

This file imports reusable losses and memory code from the existing tuned
implementation, but changes the protocol and evidence/teacher mechanisms.
"""

from __future__ import annotations

from itertools import permutations
from pathlib import Path
from pprint import pprint
from typing import Dict, Optional
import logging
import os
import time

import hydra
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import open_dict
from torch.utils.data import DataLoader

from Lib.adaptation_ema import AdaptationEMA
from Lib.fixed_random_stream import make_fixed_random_stream_loader
from Lib.physical_fault_evidence import (
    BearingGeometry,
    PhysicalEvidenceConfig,
    build_physical_masks,
    class_margin,
    construct_physical_destructive_view,
    mask_active_ratio,
    robust_fault_evidence_score,
)
from Lib.train_utils import AverageMeter, cal_acc, seed_torch

from main_tta_0711_full_tuned import (
    EVMT0711Trainer,
    EvidenceMemoryBank,
    _get,
    class_balanced_certain_mask,
    mean_js_divergence,
    neighborhood_loss,
    normalized_teacher_confidence,
    parse_only_task,
    parse_seed_runs,
    prototypical_loss,
    reliability_weighted_mt,
    reliability_weighted_sem,
    weak_style_view,
    weak_warp_view,
)


def _batch_parameter(x: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    shape = [x.shape[0]] + [1] * (x.dim() - 1)
    return value.view(*shape)


def weak_gain_view(x: torch.Tensor, strength: float = 0.03) -> torch.Tensor:
    if strength < 0:
        raise ValueError("gain strength must be non-negative")
    if strength == 0:
        return x.clone()
    gain = torch.empty(x.shape[0], device=x.device, dtype=x.dtype).uniform_(
        1.0 - float(strength), 1.0 + float(strength)
    )
    return x * _batch_parameter(x, gain)


def weak_baseline_view(x: torch.Tensor, strength: float = 0.02) -> torch.Tensor:
    if strength < 0:
        raise ValueError("baseline strength must be non-negative")
    if strength == 0:
        return x.clone()
    offset = torch.empty(x.shape[0], device=x.device, dtype=x.dtype).uniform_(
        -float(strength), float(strength)
    )
    return x + _batch_parameter(x, offset)


def weak_noise_view(x: torch.Tensor, std: float = 0.01) -> torch.Tensor:
    if std < 0:
        raise ValueError("noise std must be non-negative")
    if std == 0:
        return x.clone()
    return x + torch.randn_like(x) * float(std)


class Strict0711ResNetTrainer(EVMT0711Trainer):
    """Single-pass strict-online EVMT-DtCC for ResNet18_1D_SDE."""

    def __init__(self, cfg: omegaconf.DictConfig, run_obj=None):
        super().__init__(cfg, run_obj)
        self.adaptation_ema: Optional[AdaptationEMA] = None
        self.target_domain: Optional[int] = None
        self.physical_config: Optional[PhysicalEvidenceConfig] = None
        self.geometry = BearingGeometry()

    def setup(self):
        super().setup()
        # The cache is stored in class blocks. Build one deterministic random
        # permutation to remove that artificial order while retaining strict
        # online semantics: every target sample is visited exactly once.
        self.stream_seed = int(_get(self.tcfg, "stream_seed", self.cfg.seed_run))
        self.target_dataloader = make_fixed_random_stream_loader(
            dataset=self.datasets["target_data"],
            batch_size=self.cfg.batch_size,
            seed=self.stream_seed,
            num_workers=self.cfg.num_workers,
            drop_last=False,
            pin_memory=(self.device.type == "cuda"),
        )

        task = self.cfg.Dataset.TL_Task
        target = task[1]
        if not isinstance(target, int):
            raise ValueError("Strict PU4D physical evidence requires one integer target domain")
        self.target_domain = int(target)

    def _checkpoint_path(self):
        """Prefer the source-specific checkpoint produced by source-only training."""
        source = self.cfg.Dataset.TL_Task[0]
        if not isinstance(source, int):
            raise ValueError("Strict online runner supports one integer source domain")

        source_dir = (
            Path(self.cfg.save_model_path)
            / (str(self.cfg.Dataset.data_name) + str(self.cfg.Opt.lr_src))
            / f"source_{int(source)}"
            / f"seed_{int(self.cfg.seed_run)}"
        )
        best = source_dir / ("best_source_" + self.cfg.model_name)
        final = source_dir / self.cfg.model_name
        if best.exists():
            return best
        if final.exists():
            return final
        return super()._checkpoint_path()

    def _build_physical_config(self) -> PhysicalEvidenceConfig:
        return PhysicalEvidenceConfig(
            sampling_rate_hz=float(_get(self.tcfg, "sampling_rate_hz", 64000.0)),
            fft_size=int(_get(self.tcfg, "fft_size", 1024)),
            spectrum_length=int(_get(self.tcfg, "spectrum_length", 512)),
            harmonics=int(_get(self.tcfg, "physical_harmonics", 8)),
            outer_sideband_orders=tuple(
                int(v) for v in _get(self.tcfg, "outer_sideband_orders", [0, 1])
            ),
            inner_sideband_orders=tuple(
                int(v) for v in _get(self.tcfg, "inner_sideband_orders", [0, 1, 2])
            ),
            mask_sigma_bins=float(_get(self.tcfg, "mask_sigma_bins", 1.0)),
            background_width_bins=int(
                _get(self.tcfg, "physical_background_width", 7)
            ),
            max_mask_ratio=float(_get(self.tcfg, "max_mask_ratio", 0.18)),
            mask_activity_threshold=float(
                _get(self.tcfg, "mask_activity_threshold", 0.10)
            ),
            exclude_dc=bool(_get(self.tcfg, "exclude_dc", True)),
        )

    def initialize_models(self):
        ckpt_path = self._checkpoint_path()
        self.load_source_checkpoint(self.student, ckpt_path)
        self.enable_frequency_warp(self.student[0])

        mode = str(_get(self.tcfg, "mode", "full")).lower()
        if mode != "full":
            raise ValueError("Strict 0711 runner requires TTA0711.mode=full")

        self.memory = EvidenceMemoryBank(
            num_classes=self.num_classes,
            max_per_class=int(_get(self.tcfg, "memory_per_class", 64)),
        )
        self.physical_config = self._build_physical_config()
        self.physical_config.validate()
        if int(getattr(self.student[0], "input_len", 512)) != self.physical_config.spectrum_length:
            raise ValueError(
                "Model input length and physical evidence spectrum_length differ: "
                f"{getattr(self.student[0], 'input_len', None)} vs "
                f"{self.physical_config.spectrum_length}"
            )

        # There is no full teacher copy. Only the adaptation parameters have EMA state.
        self.adaptation_ema = AdaptationEMA(self.student)
        self.teacher = None
        self.bn_anchor = {}

        print("[STRICT 0711] full teacher copy disabled")
        print("[STRICT 0711] EMA adaptation parameters:")
        for name in self.adaptation_ema.names:
            print("  ", name)
        print("[STRICT 0711] EMA parameter count:", self.adaptation_ema.numel)
        print(
            "[STRICT 0711] target stream order: fixed random permutation "
            f"(seed={self.stream_seed})"
        )
        print("[STRICT 0711] target stream passes: 1")

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
            raise RuntimeError("No adapter or F-Warp parameters found")

        allowed = ("band_scale", "band_bias", "warp_ctrl")
        unexpected = [
            name for name, parameter in self.student.named_parameters()
            if parameter.requires_grad and not any(token in name for token in allowed)
        ]
        if unexpected:
            raise RuntimeError(f"Unexpected strict-online trainable parameters: {unexpected}")

        base_lr = float(self.cfg.Opt.lr_tar)
        weight_decay = float(self.cfg.Opt.weight_decay_tar)
        groups = []
        if adapter_params:
            groups.append(
                {
                    "params": adapter_params,
                    "lr": base_lr * float(_get(self.tcfg, "adapter_lr_scale", 1.0)),
                    "weight_decay": weight_decay,
                }
            )
        if warp_params:
            groups.append(
                {
                    "params": warp_params,
                    "lr": base_lr * float(_get(self.tcfg, "warp_lr_scale", 0.1)),
                    "weight_decay": weight_decay,
                }
            )

        print("[STRICT TRAINABLE PARAMETERS]")
        for name in trainable_names:
            print("  ", name)
        return torch.optim.AdamW(groups)

    def _set_student_adaptation_mode(self):
        # Gradients still work in eval mode. This freezes every BN running statistic
        # and disables dropout, so Adapter/F-Warp are the only adaptation channels.
        self.student.eval()
        for module in self.student.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.training:
                raise RuntimeError("BatchNorm must remain in eval mode in strict 0711")

    @torch.no_grad()
    def ema_update_teacher(self):
        if self.adaptation_ema is None:
            raise RuntimeError("Adaptation EMA is not initialized")
        beta = float(_get(self.tcfg, "ema_beta", 0.995))
        self.adaptation_ema.update(self.student, beta=beta)

    @torch.no_grad()
    def _ema_forward(self, x: torch.Tensor):
        if self.adaptation_ema is None:
            raise RuntimeError("Adaptation EMA is not initialized")
        with self.adaptation_ema.applied_to(self.student):
            self.student.eval()
            return self.forward_parts(self.student, x)

    def evidence_batch_metadata(self, sample_indices):
        return None

    def build_evidence_masks(self, pseudo_labels, batch_metadata=None):
        return build_physical_masks(
            pseudo_labels=pseudo_labels,
            target_domain=self.target_domain,
            config=self.physical_config,
            geometry=self.geometry,
        )

    def offline_diagnostics_update(self, labels, reliability_info):
        return None

    def strict_online_diagnostics_update(self, labels, predictions):
        """Observe detached pre-update student predictions (HUST overrides this)."""
        return None

    def offline_diagnostics_finalize(self):
        return None

    def post_stream_diagnostics(self, evaluation):
        """Return optional detached post-stream diagnostics for dataset wrappers."""
        return {}

    def offline_loss_diagnostics_update(
        self,
        *,
        loss_sem,
        loss_mt,
        loss_pcl,
        loss_ncl,
        loss_adapter,
        loss_warp,
        component_loss,
        total_loss,
    ):
        return None

    @torch.no_grad()
    def teacher_reliability(
        self, x: torch.Tensor, iter_num: int, batch_metadata=None
    ):
        if self.physical_config is None or self.target_domain is None:
            raise RuntimeError("Physical evidence is not initialized")

        temp = float(_get(self.tcfg, "teacher_temp", 1.0))
        gamma = float(_get(self.tcfg, "view_gamma", 5.0))
        warmup_batches = int(_get(self.tcfg, "warmup_batches", 10))
        evidence_interval = max(1, int(_get(self.tcfg, "evidence_interval", 1)))

        style_x = weak_style_view(
            x,
            strength=float(_get(self.tcfg, "view_style_strength", 0.05)),
            knots=int(_get(self.tcfg, "view_style_knots", 8)),
        )
        warp_x = weak_warp_view(
            x,
            max_warp=float(_get(self.tcfg, "view_warp_max", 0.5)),
            knots=int(_get(self.tcfg, "view_warp_knots", 8)),
        )
        gain_x = weak_gain_view(
            x, strength=float(_get(self.tcfg, "view_gain_strength", 0.03))
        )
        baseline_x = weak_baseline_view(
            x, strength=float(_get(self.tcfg, "view_baseline_strength", 0.02))
        )
        noise_x = weak_noise_view(
            x, std=float(_get(self.tcfg, "view_noise_std", 0.01))
        )

        views = [x, style_x, warp_x, gain_x, baseline_x, noise_x]
        features = []
        probabilities = []
        for view in views:
            feature, logits = self._ema_forward(view)
            features.append(feature)
            probabilities.append(torch.softmax(logits / temp, dim=1))

        mean_prob = torch.stack(probabilities, dim=0).mean(dim=0)
        pseudo = mean_prob.argmax(dim=1)
        confidence = normalized_teacher_confidence(mean_prob)
        disagreement = mean_js_divergence(probabilities, mean_prob)
        agreement = torch.exp(-gamma * disagreement).clamp(0.0, 1.0)

        masks, applicable = self.build_evidence_masks(pseudo, batch_metadata)
        evidence_active = bool(
            iter_num > warmup_batches and iter_num % evidence_interval == 0
        )

        if evidence_active and applicable.any():
            destroyed_x = construct_physical_destructive_view(
                x,
                masks,
                background_width_bins=self.physical_config.background_width_bins,
            )
            _, destroyed_logits = self._ema_forward(destroyed_x)
            destroyed_prob = torch.softmax(destroyed_logits / temp, dim=1)

            metric = str(_get(self.tcfg, "evidence_metric", "margin")).lower()
            if metric == "probability":
                original_score = mean_prob.gather(1, pseudo.unsqueeze(1)).squeeze(1)
                destroyed_score = destroyed_prob.gather(1, pseudo.unsqueeze(1)).squeeze(1)
            elif metric == "margin":
                original_score = class_margin(mean_prob, pseudo)
                destroyed_score = class_margin(destroyed_prob, pseudo)
            else:
                raise ValueError("evidence_metric must be margin or probability")

            evidence_drop = original_score - destroyed_score
            evidence = robust_fault_evidence_score(evidence_drop, applicable)
        else:
            evidence_drop = torch.zeros_like(confidence)
            evidence = torch.ones_like(confidence)

        reliability = (confidence * agreement * evidence).clamp(0.0, 1.0)
        certain = class_balanced_certain_mask(
            reliability,
            pseudo,
            num_classes=self.num_classes,
            min_reliability=float(_get(self.tcfg, "min_reliability", 0.20)),
        )

        return {
            "teacher_prob": mean_prob.detach(),
            "teacher_feature": features[0].detach(),
            "pseudo": pseudo.detach(),
            "confidence": confidence.detach(),
            "view_consistency": agreement.detach(),
            "evidence": evidence.detach(),
            "evidence_drop": evidence_drop.detach(),
            "reliability": reliability.detach(),
            "certain": certain.detach(),
            "evidence_active": evidence_active,
            "evidence_applicable": applicable.detach(),
            "mask_ratio": mask_active_ratio(
                masks,
                threshold=self.physical_config.mask_activity_threshold,
            ).detach(),
            "view_count": len(views),
        }

    def _auxiliary_schedule(self, iter_num: int) -> Dict[str, object]:
        warmup_batches = int(_get(self.tcfg, "warmup_batches", 10))
        ramp_batches = max(1, int(_get(self.tcfg, "aux_ramp_batches", 20)))
        bank_size = len(self.memory)
        bank_classes = self.memory.covered_classes()

        if iter_num <= warmup_batches:
            ramp = 0.0
        else:
            ramp = min(1.0, float(iter_num - warmup_batches) / float(ramp_batches))

        pcl_on = bool(
            ramp > 0
            and bank_classes >= int(_get(self.tcfg, "min_pcl_classes", 8))
        )
        ncl_on = bool(
            ramp > 0
            and bank_classes >= int(_get(self.tcfg, "min_ncl_classes", 16))
            and bank_size >= int(_get(self.tcfg, "min_ncl_entries", 64))
        )
        return {
            "ramp": ramp,
            "pcl_on": pcl_on,
            "ncl_on": ncl_on,
            "bank_size": bank_size,
            "bank_classes": bank_classes,
        }

    def adapt(self):
        cfg = self.cfg
        self.initialize_models()
        before_acc = cal_acc(self.dataloaders["target_data"], self.student)[0]
        print(f"Task: {cfg.Dataset.TL_Task}: Beginning Acc T = {before_acc:.2f}%;")

        passes = int(_get(self.tcfg, "passes", 1))
        if passes != 1:
            raise ValueError("Strict 0711 requires TTA0711.passes=1")

        optimizer = self.configure_trainable_parameters()
        self._set_student_adaptation_mode()

        alpha = float(_get(self.tcfg, "alpha", 2.0))
        eta = float(_get(self.tcfg, "eta", 0.05))
        lambda_mt = float(_get(self.tcfg, "lambda_mt", 0.02))
        lambda_pcl = float(_get(self.tcfg, "lambda_pcl", 0.02))
        lambda_ncl = float(_get(self.tcfg, "lambda_ncl", 0.01))
        lambda_adapter = float(_get(self.tcfg, "lambda_adapter", 0.001))
        lambda_warp = float(_get(self.tcfg, "lambda_warp", 0.0002))
        mt_warmup_scale = float(_get(self.tcfg, "mt_warmup_scale", 0.5))
        warmup_batches = int(_get(self.tcfg, "warmup_batches", 10))
        log_interval = max(1, int(_get(self.tcfg, "log_interval", 25)))

        total_batches = len(self.target_dataloader)
        online_meter = AverageMeter()
        batch_times = []
        pcl_active_batches = 0
        ncl_active_batches = 0

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        print(
            f"[STRICT 0711] batches={total_batches}, passes=1, "
            f"fixed_random_stream=True, stream_seed={self.stream_seed}, "
            f"warmup_batches={warmup_batches}, views=6"
        )

        for iter_num, (x, y, sample_indices) in enumerate(
            self.target_dataloader, start=1
        ):
            start_time = time.perf_counter()
            x = x.to(self.device)
            y = y.to(self.device)

            batch_metadata = self.evidence_batch_metadata(sample_indices)
            reliability_info = self.teacher_reliability(
                x=x, iter_num=iter_num, batch_metadata=batch_metadata
            )

            if x.size(0) <= 1:
                with torch.no_grad():
                    _, student_logits = self.forward_parts(self.student, x)
                    prediction = student_logits.argmax(dim=1)
                    batch_acc = (prediction == y).float().mean().item() * 100.0
                    online_meter.update(batch_acc, x.size(0))
                self.strict_online_diagnostics_update(y, prediction)
                self.offline_diagnostics_update(y, reliability_info)
                batch_times.append(time.perf_counter() - start_time)
                continue

            optimizer.zero_grad(set_to_none=True)
            student_feature, student_logits = self.forward_parts(self.student, x)

            # Strict online metric: prediction made before adapting on this batch.
            with torch.no_grad():
                prediction = student_logits.argmax(dim=1)
                batch_acc = (prediction == y).float().mean().item() * 100.0
                online_meter.update(batch_acc, x.size(0))

            self.strict_online_diagnostics_update(y, prediction)
            self.offline_diagnostics_update(y, reliability_info)

            loss_sem, sem_info = reliability_weighted_sem(
                logits=student_logits,
                reliability=reliability_info["reliability"],
                teacher_pseudo=reliability_info["pseudo"],
                certain_mask=reliability_info["certain"],
                alpha=alpha,
                eta=eta,
            )
            loss_mt = reliability_weighted_mt(
                student_logits,
                reliability_info["teacher_prob"],
                reliability_info["reliability"],
            )

            schedule = self._auxiliary_schedule(iter_num)
            loss_pcl = student_logits.new_tensor(0.0)
            loss_ncl = student_logits.new_tensor(0.0)
            pcl_count = 0
            ncl_count = 0

            # Both losses read only the historical bank. Current samples are added later.
            if schedule["pcl_on"]:
                loss_pcl, pcl_count = prototypical_loss(
                    student_features=student_feature,
                    pseudo_labels=reliability_info["pseudo"],
                    certain_mask=reliability_info["certain"],
                    memory=self.memory,
                    temperature=float(_get(self.tcfg, "pcl_temperature", 0.20)),
                )
                pcl_active_batches += int(pcl_count > 0)
            if schedule["ncl_on"]:
                loss_ncl, ncl_count = neighborhood_loss(
                    student_features=student_feature,
                    student_logits=student_logits,
                    uncertain_mask=~reliability_info["certain"],
                    memory=self.memory,
                    neighbors=int(_get(self.tcfg, "ncl_neighbors", 3)),
                    temperature=float(_get(self.tcfg, "ncl_temperature", 0.20)),
                )
                ncl_active_batches += int(ncl_count > 0)

            loss_adapter = self.adapter_reg_loss()
            loss_warp = self.warp_reg_loss()

            lambda_mt_eff = lambda_mt * (
                mt_warmup_scale if iter_num <= warmup_batches else 1.0
            )
            lambda_pcl_eff = (
                lambda_pcl * float(schedule["ramp"]) if schedule["pcl_on"] else 0.0
            )
            lambda_ncl_eff = (
                lambda_ncl * float(schedule["ramp"]) if schedule["ncl_on"] else 0.0
            )

            component_loss = (
                loss_sem
                + lambda_mt_eff * loss_mt
                + lambda_pcl_eff * loss_pcl
                + lambda_ncl_eff * loss_ncl
            )
            total_loss = (
                component_loss
                + lambda_adapter * loss_adapter
                + lambda_warp * loss_warp
            )
            self.offline_loss_diagnostics_update(
                loss_sem=loss_sem.detach(),
                loss_mt=loss_mt.detach(),
                loss_pcl=loss_pcl.detach(),
                loss_ncl=loss_ncl.detach(),
                loss_adapter=loss_adapter.detach(),
                loss_warp=loss_warp.detach(),
                component_loss=component_loss.detach(),
                total_loss=total_loss.detach(),
            )
            total_loss.backward()
            optimizer.step()
            self.ema_update_teacher()

            # Update memory last to prevent self-prototype/self-neighbor leakage.
            certain = reliability_info["certain"]
            if certain.any():
                self.memory.update(
                    reliability_info["teacher_feature"][certain],
                    reliability_info["teacher_prob"][certain],
                    reliability_info["reliability"][certain],
                    reliability_info["pseudo"][certain],
                )

            batch_times.append(time.perf_counter() - start_time)
            should_log = (
                iter_num == 1
                or iter_num % log_interval == 0
                or iter_num == total_batches
            )
            if should_log:
                applicable = reliability_info["evidence_applicable"]
                fault_evidence_mean = (
                    reliability_info["evidence"][applicable].mean().item()
                    if applicable.any()
                    else 1.0
                )
                print(
                    f"iter {iter_num}/{total_batches} | strict=1 | acc={batch_acc:.2f} | "
                    f"total={total_loss.item():.6f} | sem={loss_sem.item():.6f} | "
                    f"mt={loss_mt.item():.6f} | pcl={loss_pcl.item():.6f}({pcl_count}) | "
                    f"ncl={loss_ncl.item():.6f}({ncl_count}) | "
                    f"rel={reliability_info['reliability'].mean().item():.4f} | "
                    f"certain={certain.float().mean().item():.3f} | "
                    f"conf={reliability_info['confidence'].mean().item():.4f} | "
                    f"agree={reliability_info['view_consistency'].mean().item():.4f} | "
                    f"evidence={fault_evidence_mean:.4f} | "
                    f"ev_app={applicable.float().mean().item():.3f} | "
                    f"mask={reliability_info['mask_ratio'][applicable].mean().item() if applicable.any() else 0.0:.3f} | "
                    f"ev_on={int(reliability_info['evidence_active'])} | "
                    f"bank={len(self.memory)} | bank_cls={self.memory.covered_classes()}/{self.num_classes} | "
                    f"pcl_on={int(schedule['pcl_on'])} | ncl_on={int(schedule['ncl_on'])} | "
                    f"w_mt={lambda_mt_eff:.4f} | w_pcl={lambda_pcl_eff:.4f} | "
                    f"w_ncl={lambda_ncl_eff:.4f} | warp_max={self.warp_delta_max():.4f} | "
                    f"te={sem_info['l_te'].item():.6f} | div={sem_info['l_div'].item():.6f}"
                )

            if self.run is not None and cfg.process_wandb:
                self.run.log(
                    {
                        "strict_online_batch_acc": batch_acc,
                        "loss_total": total_loss.item(),
                        "mean_reliability": reliability_info["reliability"].mean().item(),
                        "certain_ratio": certain.float().mean().item(),
                        "memory_size": len(self.memory),
                        "memory_classes": self.memory.covered_classes(),
                    },
                    step=iter_num,
                )

        self.offline_diagnostics_finalize()
        print(
            f"Task: {cfg.Dataset.TL_Task}: Strict Online Acc = {online_meter.avg:.2f}%;"
        )
        self.student.eval()
        post_stream_evaluation = cal_acc(
            self.dataloaders["target_data"], self.student
        )
        post_stream_acc = post_stream_evaluation[0]
        post_stream_diagnostics = self.post_stream_diagnostics(
            post_stream_evaluation
        )
        print(
            f"Task: {cfg.Dataset.TL_Task}: Post-stream Full-Target Acc = "
            f"{post_stream_acc:.2f}%"
        )

        mean_batch_ms = 1000.0 * sum(batch_times) / max(len(batch_times), 1)
        peak_memory_mb = 0.0
        if self.device.type == "cuda":
            peak_memory_mb = torch.cuda.max_memory_allocated(self.device) / (1024.0 ** 2)
        print(
            f"[STRICT DIAGNOSTICS] mean_batch_ms={mean_batch_ms:.2f} | "
            f"peak_memory_mb={peak_memory_mb:.2f} | "
            f"pcl_active_batches={pcl_active_batches} | "
            f"ncl_active_batches={ncl_active_batches} | "
            f"bank={len(self.memory)} | bank_cls={self.memory.covered_classes()}/{self.num_classes}"
        )
        result = {
            "before": before_acc,
            "online": online_meter.avg,
            "post_stream": post_stream_acc,
        }
        result.update(post_stream_diagnostics)
        return result


@hydra.main(version_base=None, config_path="./Configs", config_name="defaults")
def run(cfg: omegaconf.DictConfig):
    only_task = parse_only_task(getattr(cfg, "only_task", None))
    task_list = [only_task] if only_task is not None else list(permutations(cfg.Dataset.TL_list, 2))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id).strip("\"'")
    seed_list = parse_seed_runs(getattr(cfg, "seed_runs", None))
    train_time = time.strftime("%m-%d %H:%M", time.localtime())
    last_cfg_dict = None

    for seed_run in seed_list:
        for task in task_list:
            with open_dict(cfg):
                cfg.train_time = train_time
                cfg.seed_run = int(seed_run)
                cfg.Dataset.TL_Task = task
                cfg.Dataset.input_kind = "fft"
                cfg.Model.bottleneck_num = 128
                cfg.batch_size = int(getattr(cfg, "batch_size", 128))
                cfg.num_workers = int(getattr(cfg, "num_workers", 4))
                cfg.PR = 0
                cfg.Opt.lr_src = float(getattr(cfg.Opt, "lr_src", 1e-3))
                cfg.Opt.lr_tar = float(getattr(cfg.Opt, "lr_tar", 1.5e-2))
                cfg.Opt.weight_decay_tar = float(
                    getattr(cfg.Opt, "weight_decay_tar", 1e-3)
                )
                if not hasattr(cfg, "TTA0711"):
                    cfg.TTA0711 = {}
                if hasattr(cfg, "wandb") and hasattr(cfg.wandb, "setup"):
                    cfg.wandb.setup.name = f"0711_strict_{task}_seed{seed_run}"
                    cfg.wandb.setup.group = (
                        f"{train_time}_0711_strict_{cfg.Model.model_name}"
                    )
                    cfg.wandb.setup.project = f"TTA_0711_STRICT_{cfg.Dataset.data_name}"

                last_cfg_dict = omegaconf.OmegaConf.to_container(
                    cfg, resolve=True, throw_on_missing=True
                )

            print("=" * 80)
            print(
                f"STRICT 0711 TTA | dataset={cfg.Dataset.data_name} | task={task} | "
                f"model={cfg.Model.model_name} | seed={seed_run}"
            )
            print("TTA0711 config:", cfg.TTA0711)
            print("=" * 80)

            if cfg.process_wandb:
                run_obj = wandb.init(config=last_cfg_dict, **cfg.wandb.setup)
                with run_obj:
                    trainer = Strict0711ResNetTrainer(cfg, run_obj)
                    trainer.setup()
                    trainer.adapt()
            else:
                trainer = Strict0711ResNetTrainer(cfg, None)
                trainer.setup()
                trainer.adapt()

    print("tasks:", task_list)
    print("final config:")
    pprint(last_cfg_dict)


if __name__ == "__main__":
    run()
