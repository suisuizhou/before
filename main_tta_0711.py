#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
EVMT-DtCC / 0711 reimplementation for the current SDE-FWarp project.

This file is intentionally independent from main_tta_SDE.py.  It keeps the
existing ResNet18 + spectral adapter + F-Warp path, then adds:
  1) multi-view EMA teacher predictions;
  2) fault-spectrum evidence verification;
  3) class-balanced reliability routing;
  4) reliability-weighted SEM and mean-teacher distillation;
  5) optional class-balanced memory, PCL and NCL (mode=full).

Recommended first run:
  ++TTA0711.mode=core
Then compare with:
  ++TTA0711.mode=full
"""

from __future__ import annotations

from pathlib import Path
from pprint import pprint
from itertools import permutations
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple
import logging
import math
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

import Dataset
from Lib.model import get_model
from Lib.train_utils import AverageMeter, cal_acc, seed_torch


# -----------------------------------------------------------------------------
# Generic config / naming utilities
# -----------------------------------------------------------------------------


def _to_plain_container(obj):
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(obj, (ListConfig, DictConfig)):
            return OmegaConf.to_container(obj, resolve=True)
    except Exception:
        pass
    return obj


def _normalize_task_node(node):
    node = _to_plain_container(node)
    if isinstance(node, (list, tuple)):
        return tuple(_normalize_task_node(x) for x in node)
    return int(node)


def parse_only_task(value):
    if value is None:
        return None

    import ast

    obj = ast.literal_eval(value) if isinstance(value, str) else _to_plain_container(value)
    if not isinstance(obj, (list, tuple)) or len(obj) != 2:
        raise ValueError(f"only_task must be like [src, tar], got: {value}")
    return (_normalize_task_node(obj[0]), _normalize_task_node(obj[1]))


def parse_seed_runs(value):
    if value is None:
        return [2025]

    import ast

    value = _to_plain_container(value)
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    raise ValueError(f"Unsupported seed_runs format: {value}")


def build_model_name(cfg):
    model_name = str(cfg.Model.model_name)
    seed_run = str(cfg.seed_run)
    input_kind = str(cfg.Dataset.input_kind)
    model_type = str(cfg.Model.model_type)

    if model_type == "linear":
        return f"{model_name}{seed_run}{input_kind}_Linear.pt"
    if model_type == "wn":
        return f"{model_name}{seed_run}{input_kind}_WN.pt"
    return f"{model_name}{seed_run}{input_kind}_Pro.pt"


def _get(section, name: str, default):
    return getattr(section, name, default)


# -----------------------------------------------------------------------------
# Tensor shape and weak-view utilities
# -----------------------------------------------------------------------------


def to_channel_first_1d(x: torch.Tensor):
    squeezed = False
    transposed = False

    if x.dim() == 2:
        x = x.unsqueeze(1)
        squeezed = True
    elif x.dim() == 3 and x.shape[-1] == 1:
        x = x.transpose(1, 2)
        transposed = True
    elif x.dim() != 3:
        raise ValueError(f"Unexpected 1D spectrum shape: {tuple(x.shape)}")

    return x, squeezed, transposed


def restore_1d_shape(x: torch.Tensor, squeezed: bool, transposed: bool):
    if transposed:
        x = x.transpose(1, 2)
    if squeezed:
        x = x.squeeze(1)
    return x


def weak_style_view(
    x: torch.Tensor,
    strength: float = 0.05,
    knots: int = 8,
) -> torch.Tensor:
    """Label-preserving weak smooth amplitude modulation."""
    x_cf, squeezed, transposed = to_channel_first_1d(x)
    batch, _, length = x_cf.shape
    knots = max(2, int(knots))

    ctrl = torch.empty(
        batch, 1, knots, device=x_cf.device, dtype=x_cf.dtype
    ).uniform_(-strength, strength)
    pos = torch.linspace(
        -1.0, 1.0, steps=knots, device=x_cf.device, dtype=x_cf.dtype
    ).view(1, 1, knots)
    tilt = torch.empty(
        batch, 1, 1, device=x_cf.device, dtype=x_cf.dtype
    ).uniform_(-0.5 * strength, 0.5 * strength)
    ctrl = ctrl + tilt * pos

    mask = F.interpolate(ctrl, size=length, mode="linear", align_corners=True).exp()
    out = x_cf * mask
    return restore_1d_shape(out, squeezed, transposed)


def weak_warp_view(
    x: torch.Tensor,
    max_warp: float = 0.5,
    knots: int = 8,
) -> torch.Tensor:
    """Label-preserving weak sample-wise frequency-axis perturbation."""
    x_cf, squeezed, transposed = to_channel_first_1d(x)
    batch, _, length = x_cf.shape
    knots = max(2, int(knots))

    ctrl = torch.empty(
        batch, 1, knots, device=x_cf.device, dtype=x_cf.dtype
    ).uniform_(-max_warp, max_warp)
    pos = torch.linspace(
        -1.0, 1.0, steps=knots, device=x_cf.device, dtype=x_cf.dtype
    ).view(1, 1, knots)
    tilt = torch.empty(
        batch, 1, 1, device=x_cf.device, dtype=x_cf.dtype
    ).uniform_(-0.2 * max_warp, 0.2 * max_warp)
    ctrl = ctrl + tilt * pos

    delta = F.interpolate(ctrl, size=length, mode="linear", align_corners=True)
    base = torch.arange(length, device=x_cf.device, dtype=x_cf.dtype).view(1, 1, length)
    sample_pos = (base + delta).clamp(0.0, float(length - 1))

    x_norm = 2.0 * sample_pos / float(length - 1) - 1.0
    x_norm = x_norm.squeeze(1)
    y_norm = torch.zeros_like(x_norm)
    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(1)

    out = F.grid_sample(
        x_cf.unsqueeze(2),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).squeeze(2)
    return restore_1d_shape(out, squeezed, transposed)


# -----------------------------------------------------------------------------
# Reliability and evidence verification
# -----------------------------------------------------------------------------


def normalized_teacher_confidence(prob: torch.Tensor) -> torch.Tensor:
    classes = max(int(prob.shape[1]), 2)
    entropy = -(prob.clamp_min(1e-8) * prob.clamp_min(1e-8).log()).sum(dim=1)
    return (1.0 - entropy / math.log(classes)).clamp(0.0, 1.0)


def mean_js_divergence(view_probs: Sequence[torch.Tensor], mean_prob: torch.Tensor):
    js_terms = []
    log_mean = mean_prob.clamp_min(1e-8).log()
    for prob in view_probs:
        prob_safe = prob.clamp_min(1e-8)
        js_terms.append((prob_safe * (prob_safe.log() - log_mean)).sum(dim=1))
    return torch.stack(js_terms, dim=0).mean(dim=0)


def robust_evidence_score(probability_drop: torch.Tensor) -> torch.Tensor:
    """Median/MAD-normalized evidence response mapped to [0, 1]."""
    median = probability_drop.median()
    mad = (probability_drop - median).abs().median()
    robust_scale = 1.4826 * mad + 1e-6
    return torch.sigmoid((probability_drop - median) / robust_scale)


def class_balanced_certain_mask(
    reliability: torch.Tensor,
    pseudo_labels: torch.Tensor,
    num_classes: int,
    min_reliability: float = 0.0,
) -> torch.Tensor:
    """Keep samples at or above their predicted-class mean reliability."""
    mask = torch.zeros_like(reliability, dtype=torch.bool)
    for class_id in range(int(num_classes)):
        class_mask = pseudo_labels == class_id
        if not class_mask.any():
            continue
        threshold = reliability[class_mask].mean()
        mask[class_mask] = reliability[class_mask] >= threshold
    if min_reliability > 0:
        mask &= reliability >= float(min_reliability)
    return mask


def contiguous_top_band_mask(
    saliency: torch.Tensor,
    num_bands: int = 4,
    band_width: int = 17,
) -> torch.Tensor:
    """Turn top saliency centers into continuous frequency bands."""
    if saliency.dim() != 2:
        raise ValueError(f"saliency must be [B, L], got {tuple(saliency.shape)}")

    batch, length = saliency.shape
    num_bands = max(1, min(int(num_bands), length))
    band_width = max(1, int(band_width))
    if band_width % 2 == 0:
        band_width += 1

    center_idx = saliency.topk(k=num_bands, dim=1, largest=True).indices
    centers = torch.zeros_like(saliency)
    centers.scatter_(1, center_idx, 1.0)
    mask = F.max_pool1d(
        centers.unsqueeze(1),
        kernel_size=band_width,
        stride=1,
        padding=band_width // 2,
    ).squeeze(1)
    return mask[:, :length].clamp(0.0, 1.0)


def construct_destructive_view(
    x: torch.Tensor,
    mask: torch.Tensor,
    background_width: int = 31,
) -> torch.Tensor:
    x_cf, squeezed, transposed = to_channel_first_1d(x)
    background_width = max(3, int(background_width))
    if background_width % 2 == 0:
        background_width += 1

    background = F.avg_pool1d(
        x_cf,
        kernel_size=background_width,
        stride=1,
        padding=background_width // 2,
    )
    mask_cf = mask.unsqueeze(1).to(dtype=x_cf.dtype)
    destroyed = (1.0 - mask_cf) * x_cf + mask_cf * background
    return restore_1d_shape(destroyed, squeezed, transposed)


# -----------------------------------------------------------------------------
# EVMT losses
# -----------------------------------------------------------------------------


def reliability_weighted_sem(
    logits: torch.Tensor,
    reliability: torch.Tensor,
    teacher_pseudo: torch.Tensor,
    certain_mask: torch.Tensor,
    alpha: float = 2.0,
    eta: float = 0.2,
):
    """Class-aware, reliability-weighted Tsallis entropy + diversity."""
    prob = torch.softmax(logits, dim=1)
    classes = prob.shape[1]

    if certain_mask.any():
        freq = torch.bincount(
            teacher_pseudo[certain_mask], minlength=classes
        ).to(device=prob.device, dtype=prob.dtype)
    else:
        freq = torch.zeros(classes, device=prob.device, dtype=prob.dtype)

    hat_p = prob / (freq.unsqueeze(0) + 1.0)
    hat_p = hat_p / hat_p.sum(dim=1, keepdim=True).clamp_min(1e-12)

    if abs(float(alpha) - 1.0) < 1e-6:
        per_sample = -(hat_p.clamp_min(1e-8) * hat_p.clamp_min(1e-8).log()).sum(dim=1)
    else:
        per_sample = (1.0 - hat_p.clamp_min(1e-8).pow(alpha).sum(dim=1)) / (
            alpha - 1.0
        )

    reliability = reliability.detach().clamp(0.0, 1.0)
    weights = float(eta) + (1.0 - float(eta)) * reliability
    l_te = (weights * per_sample).sum() / weights.sum().clamp_min(1e-8)

    p_bar = (weights.unsqueeze(1) * hat_p).sum(dim=0) / weights.sum().clamp_min(1e-8)
    l_div = (p_bar.clamp_min(1e-8) * p_bar.clamp_min(1e-8).log()).sum()
    return l_te + l_div, {"l_te": l_te.detach(), "l_div": l_div.detach()}


def reliability_weighted_mt(
    student_logits: torch.Tensor,
    teacher_prob: torch.Tensor,
    reliability: torch.Tensor,
) -> torch.Tensor:
    teacher_prob = teacher_prob.detach().clamp_min(1e-8)
    student_log_prob = F.log_softmax(student_logits, dim=1)
    per_sample = (
        teacher_prob * (teacher_prob.log() - student_log_prob)
    ).sum(dim=1)
    weights = reliability.detach().clamp_min(0.0)
    return (weights * per_sample).sum() / weights.sum().clamp_min(1e-8)


# -----------------------------------------------------------------------------
# Evidence-weighted class-balanced memory, PCL and NCL
# -----------------------------------------------------------------------------


class EvidenceMemoryBank:
    def __init__(self, num_classes: int, max_per_class: int = 128):
        self.num_classes = int(num_classes)
        self.max_per_class = int(max_per_class)
        self._features: List[deque] = [
            deque(maxlen=self.max_per_class) for _ in range(self.num_classes)
        ]
        self._probs: List[deque] = [
            deque(maxlen=self.max_per_class) for _ in range(self.num_classes)
        ]
        self._reliability: List[deque] = [
            deque(maxlen=self.max_per_class) for _ in range(self.num_classes)
        ]

    def __len__(self):
        return sum(len(x) for x in self._features)

    @torch.no_grad()
    def update(
        self,
        features: torch.Tensor,
        probs: torch.Tensor,
        reliability: torch.Tensor,
        labels: torch.Tensor,
    ):
        features = features.detach().cpu()
        probs = probs.detach().cpu()
        reliability = reliability.detach().cpu()
        labels = labels.detach().cpu()

        for feat, prob, score, label in zip(features, probs, reliability, labels):
            class_id = int(label.item())
            if class_id < 0 or class_id >= self.num_classes:
                continue
            self._features[class_id].append(feat.clone())
            self._probs[class_id].append(prob.clone())
            self._reliability[class_id].append(float(score.item()))

    def prototypes(self, device: torch.device):
        prototypes = []
        class_ids = []
        for class_id in range(self.num_classes):
            if not self._features[class_id]:
                continue
            feats = torch.stack(list(self._features[class_id])).to(device)
            rel = torch.tensor(
                list(self._reliability[class_id]), device=device, dtype=feats.dtype
            )
            proto = (rel.unsqueeze(1) * feats).sum(dim=0) / rel.sum().clamp_min(1e-8)
            prototypes.append(F.normalize(proto, dim=0))
            class_ids.append(class_id)

        if not prototypes:
            return None, None
        return torch.stack(prototypes), torch.tensor(class_ids, device=device, dtype=torch.long)

    def all_entries(self, device: torch.device):
        all_features = []
        all_probs = []
        all_reliability = []
        all_labels = []

        for class_id in range(self.num_classes):
            if not self._features[class_id]:
                continue
            count = len(self._features[class_id])
            all_features.extend(self._features[class_id])
            all_probs.extend(self._probs[class_id])
            all_reliability.extend(self._reliability[class_id])
            all_labels.extend([class_id] * count)

        if not all_features:
            return None, None, None, None

        features = F.normalize(torch.stack(list(all_features)).to(device), dim=1)
        probs = torch.stack(list(all_probs)).to(device)
        reliability = torch.tensor(
            list(all_reliability), device=device, dtype=features.dtype
        )
        labels = torch.tensor(all_labels, device=device, dtype=torch.long)
        return features, probs, reliability, labels


def prototypical_loss(
    student_features: torch.Tensor,
    pseudo_labels: torch.Tensor,
    certain_mask: torch.Tensor,
    memory: EvidenceMemoryBank,
    temperature: float = 0.1,
):
    zero = student_features.new_tensor(0.0)
    prototypes, class_ids = memory.prototypes(student_features.device)
    if prototypes is None or certain_mask.sum() == 0:
        return zero, 0

    mapping = torch.full(
        (memory.num_classes,), -1, device=student_features.device, dtype=torch.long
    )
    mapping[class_ids] = torch.arange(class_ids.numel(), device=student_features.device)

    selected_labels = pseudo_labels[certain_mask]
    mapped = mapping[selected_labels]
    valid = mapped >= 0
    if not valid.any():
        return zero, 0

    selected_features = F.normalize(student_features[certain_mask][valid], dim=1)
    logits = selected_features @ prototypes.t() / float(temperature)
    return F.cross_entropy(logits, mapped[valid]), int(valid.sum().item())


def neighborhood_loss(
    student_features: torch.Tensor,
    student_logits: torch.Tensor,
    uncertain_mask: torch.Tensor,
    memory: EvidenceMemoryBank,
    neighbors: int = 5,
    temperature: float = 0.1,
):
    zero = student_features.new_tensor(0.0)
    if uncertain_mask.sum() == 0:
        return zero, 0

    mem_feat, mem_prob, mem_rel, _ = memory.all_entries(student_features.device)
    if mem_feat is None:
        return zero, 0

    query = F.normalize(student_features[uncertain_mask], dim=1)
    similarities = query @ mem_feat.t()
    k = min(max(1, int(neighbors)), mem_feat.shape[0])
    top_sim, top_idx = similarities.topk(k=k, dim=1)

    neighbor_rel = mem_rel[top_idx]
    weights = torch.softmax(top_sim / float(temperature), dim=1) * neighbor_rel
    neighbor_prob = mem_prob[top_idx]
    soft_target = (weights.unsqueeze(2) * neighbor_prob).sum(dim=1)
    soft_target = soft_target / soft_target.sum(dim=1, keepdim=True).clamp_min(1e-8)

    student_log_prob = F.log_softmax(student_logits[uncertain_mask], dim=1)
    loss = (
        soft_target.detach()
        * (soft_target.detach().clamp_min(1e-8).log() - student_log_prob)
    ).sum(dim=1).mean()
    return loss, int(query.shape[0])


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------


class EVMT0711Trainer:
    def __init__(self, cfg: omegaconf.DictConfig, run_obj=None):
        seed_torch(cfg.seed_run)
        with open_dict(cfg):
            cfg.model_name = build_model_name(cfg)
            cfg.save_model_path = "./TTA_Model"
            cfg.model_path = (
                Path(cfg.save_model_path)
                / (str(cfg.Dataset.data_name) + str(cfg.Opt.lr_src))
                / (str(cfg.Dataset.TL_Task) + "_Task")
            )
            Path(cfg.model_path).mkdir(parents=True, exist_ok=True)

        self.cfg = cfg
        self.tcfg = getattr(cfg, "TTA0711", {})
        self.run = run_obj
        self.device = None
        self.num_classes = None
        self.datasets = {}
        self.dataloaders = {}
        self.target_dataloader = None
        self.student = None
        self.teacher = None
        self.memory = None
        self.bn_anchor: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Setup, loading and F-Warp

    def setup(self):
        cfg = self.cfg
        seed_torch(cfg.seed_run)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            device_count = torch.cuda.device_count()
            logging.info("using %s gpus", device_count)
            assert cfg.batch_size % device_count == 0
        else:
            self.device = torch.device("cpu")
            logging.warning("GPU is not available; using CPU")

        dataset_cls = getattr(Dataset, cfg.Dataset.data_name)
        self.num_classes = int(dataset_cls.num_classes)
        self.datasets["source_data"], self.datasets["target_data"] = dataset_cls(
            **cfg.Dataset
        ).data_generator()

        self.dataloaders = {
            name: DataLoader(
                self.datasets[name],
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                drop_last=False,
                pin_memory=(self.device.type == "cuda"),
            )
            for name in ["source_data", "target_data"]
        }
        self.target_dataloader = DataLoader(
            self.datasets["target_data"],
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            drop_last=False,
            pin_memory=(self.device.type == "cuda"),
        )

        self.student = get_model(
            num_classes=self.num_classes, cfg=cfg, **cfg.Model
        ).to(self.device)

    def _checkpoint_path(self):
        best = Path(self.cfg.model_path) / ("best_source_" + self.cfg.model_name)
        final = Path(self.cfg.model_path) / self.cfg.model_name
        if best.exists():
            return best
        if final.exists():
            return final
        raise FileNotFoundError(
            f"Neither best nor final source checkpoint exists: {best} / {final}"
        )

    def load_source_checkpoint(self, model: nn.Module, ckpt_path: Path):
        state = torch.load(ckpt_path, map_location=self.device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise TypeError(f"Unsupported checkpoint object: {type(state)}")

        current = model.state_dict()
        compatible = {}
        unexpected = []
        mismatched = []
        for key, value in state.items():
            normalized_key = key[7:] if key.startswith("module.") else key
            if normalized_key not in current:
                unexpected.append(normalized_key)
                continue
            if current[normalized_key].shape != value.shape:
                mismatched.append(
                    (normalized_key, tuple(value.shape), tuple(current[normalized_key].shape))
                )
                continue
            compatible[normalized_key] = value

        message = model.load_state_dict(compatible, strict=False)
        print("[SOURCE CKPT]", ckpt_path)
        print("  loaded tensors:", len(compatible))
        print("  missing keys:", message.missing_keys)
        print("  unexpected source keys:", unexpected[:20])
        print("  mismatched keys:", mismatched[:20])
        return message

    def _warp_1d(self, x_cf: torch.Tensor, backbone: nn.Module):
        batch, _, length = x_cf.shape
        ctrl = torch.tanh(
            backbone.warp_ctrl.to(device=x_cf.device, dtype=x_cf.dtype)
        )
        delta = F.interpolate(ctrl, size=length, mode="linear", align_corners=True)
        delta = delta * float(backbone.max_warp)

        base = torch.arange(length, device=x_cf.device, dtype=x_cf.dtype).view(
            1, 1, length
        )
        sample_pos = (base + delta).clamp(0.0, float(length - 1))
        x_norm = 2.0 * sample_pos / float(length - 1) - 1.0
        x_norm = x_norm.expand(batch, -1, -1).squeeze(1)
        y_norm = torch.zeros_like(x_norm)
        grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(1)

        return F.grid_sample(
            x_cf.unsqueeze(2),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(2)

    def enable_frequency_warp(self, backbone: nn.Module):
        use_warp = bool(_get(self.tcfg, "use_frequency_warp", True))
        if not use_warp:
            print("[FWARP] disabled")
            return
        if not hasattr(backbone, "apply_spectral_adapter"):
            raise AttributeError(
                f"{type(backbone).__name__} has no apply_spectral_adapter() interface"
            )

        knots = int(_get(self.tcfg, "warp_knots", 16))
        max_warp = float(_get(self.tcfg, "max_warp", 2.0))
        device = next(backbone.parameters()).device

        backbone.warp_ctrl = nn.Parameter(torch.zeros(1, 1, knots, device=device))
        backbone.max_warp = max_warp
        backbone.warp_knots = knots
        backbone._plain_apply_spectral_adapter = backbone.apply_spectral_adapter
        trainer = self

        def warped_apply_spectral_adapter(x):
            x_cf, squeezed, transposed = to_channel_first_1d(x)
            if x_cf.shape[-1] != int(backbone.input_len):
                x_cf = F.interpolate(
                    x_cf,
                    size=int(backbone.input_len),
                    mode="linear",
                    align_corners=False,
                )
            x_warped = trainer._warp_1d(x_cf, backbone)
            x_warped = restore_1d_shape(x_warped, squeezed, transposed)
            return backbone._plain_apply_spectral_adapter(x_warped)

        backbone.apply_spectral_adapter = warped_apply_spectral_adapter
        print(f"[FWARP] knots={knots}, max_warp={max_warp}")

    def initialize_models(self):
        ckpt_path = self._checkpoint_path()
        self.load_source_checkpoint(self.student, ckpt_path)
        self.enable_frequency_warp(self.student[0])

        self.teacher = get_model(
            num_classes=self.num_classes, cfg=self.cfg, **self.cfg.Model
        ).to(self.device)
        self.load_source_checkpoint(self.teacher, ckpt_path)
        self.enable_frequency_warp(self.teacher[0])
        self.teacher.load_state_dict(self.student.state_dict(), strict=True)
        self.teacher.requires_grad_(False)
        self.teacher.eval()

        self.bn_anchor = self._capture_bn_anchor(self.student)

        mode = str(_get(self.tcfg, "mode", "core")).lower()
        if mode not in {"core", "full"}:
            raise ValueError(f"TTA0711.mode must be core or full, got {mode}")
        if mode == "full":
            self.memory = EvidenceMemoryBank(
                num_classes=self.num_classes,
                max_per_class=int(_get(self.tcfg, "memory_per_class", 128)),
            )

    # ------------------------------------------------------------------
    # Parameter control and regularization

    def _capture_bn_anchor(self, model: nn.Module):
        anchor = {}
        bn_modules = {
            name
            for name, module in model.named_modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
        }
        for name, param in model.named_parameters():
            parent = name.rsplit(".", 1)[0] if "." in name else ""
            if parent in bn_modules and name.endswith(("weight", "bias")):
                anchor[name] = param.detach().clone()
        return anchor

    def bn_anchor_loss(self):
        zero = next(self.student.parameters()).new_tensor(0.0)
        terms = []
        for name, param in self.student.named_parameters():
            if name in self.bn_anchor:
                terms.append((param - self.bn_anchor[name]).pow(2).mean())
        return torch.stack(terms).mean() if terms else zero

    def adapter_reg_loss(self):
        backbone = self.student[0]
        if not hasattr(backbone, "band_scale") or backbone.band_scale is None:
            return next(self.student.parameters()).new_tensor(0.0)
        delta = float(getattr(backbone, "adapter_delta", 0.1))
        scale = 1.0 + delta * torch.tanh(backbone.band_scale)
        bias = delta * backbone.band_bias
        return (scale - 1.0).pow(2).mean() + bias.pow(2).mean()

    def warp_reg_loss(self):
        backbone = self.student[0]
        if not hasattr(backbone, "warp_ctrl"):
            return next(self.student.parameters()).new_tensor(0.0)
        delta = F.interpolate(
            torch.tanh(backbone.warp_ctrl),
            size=int(backbone.input_len),
            mode="linear",
            align_corners=True,
        ) * float(backbone.max_warp)
        denom = max(float(backbone.max_warp) ** 2, 1e-6)
        l2 = delta.pow(2).mean() / denom
        smooth = (delta[:, :, 1:] - delta[:, :, :-1]).pow(2).mean() / denom
        return l2 + float(_get(self.tcfg, "warp_smooth_weight", 2.0)) * smooth

    def warp_delta_max(self):
        backbone = self.student[0]
        if not hasattr(backbone, "warp_ctrl"):
            return 0.0
        with torch.no_grad():
            delta = F.interpolate(
                torch.tanh(backbone.warp_ctrl),
                size=int(backbone.input_len),
                mode="linear",
                align_corners=True,
            ) * float(backbone.max_warp)
            return float(delta.abs().max().item())

    def configure_trainable_parameters(self):
        for param in self.student.parameters():
            param.requires_grad = False

        update_bn = bool(_get(self.tcfg, "update_bn_affine", True))
        bn_parameter_ids = set()
        if update_bn:
            for module in self.student.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    if module.weight is not None:
                        module.weight.requires_grad = True
                        bn_parameter_ids.add(id(module.weight))
                    if module.bias is not None:
                        module.bias.requires_grad = True
                        bn_parameter_ids.add(id(module.bias))

        adapter_params = []
        warp_params = []
        bn_params = []
        other_params = []
        trainable_names = []

        for name, param in self.student.named_parameters():
            if "band_scale" in name or "band_bias" in name:
                param.requires_grad = True
            elif "warp_ctrl" in name:
                param.requires_grad = True
            elif bool(_get(self.tcfg, "update_prompt", False)) and "prompt_embed" in name:
                param.requires_grad = True

            if not param.requires_grad:
                continue
            trainable_names.append(name)
            if id(param) in bn_parameter_ids:
                bn_params.append(param)
            elif "warp_ctrl" in name:
                warp_params.append(param)
            elif "band_scale" in name or "band_bias" in name:
                adapter_params.append(param)
            else:
                other_params.append(param)

        base_lr = float(self.cfg.Opt.lr_tar)
        weight_decay = float(self.cfg.Opt.weight_decay_tar)
        groups = []
        if bn_params:
            groups.append(
                {
                    "params": bn_params,
                    "lr": base_lr * float(_get(self.tcfg, "bn_lr_scale", 0.1)),
                    "weight_decay": 0.0,
                }
            )
        if adapter_params:
            groups.append(
                {
                    "params": adapter_params,
                    "lr": base_lr
                    * float(_get(self.tcfg, "adapter_lr_scale", 1.0)),
                    "weight_decay": weight_decay,
                }
            )
        if warp_params:
            groups.append(
                {
                    "params": warp_params,
                    "lr": base_lr * float(_get(self.tcfg, "warp_lr_scale", 0.2)),
                    "weight_decay": weight_decay,
                }
            )
        if other_params:
            groups.append(
                {
                    "params": other_params,
                    "lr": base_lr,
                    "weight_decay": weight_decay,
                }
            )

        if not groups:
            raise RuntimeError("No trainable parameters found for 0711 TTA")

        print("[TRAINABLE PARAMETERS]")
        for name in trainable_names:
            print("  ", name)
        return torch.optim.AdamW(groups)

    def _set_student_adaptation_mode(self):
        self.student.train()
        # Keep target BN statistics adaptive but remove dropout noise.
        for module in self.student.modules():
            if isinstance(module, nn.Dropout):
                module.eval()

    # ------------------------------------------------------------------
    # Teacher inference, evidence and EMA

    def forward_parts(self, model: nn.Module, x: torch.Tensor):
        feature = model[0](x)
        bottleneck = model[1](feature)
        logits = model[2](bottleneck)
        return bottleneck, logits

    def teacher_reliability(
        self,
        x: torch.Tensor,
        current_pass: int,
        iter_num: int,
    ):
        temp = float(_get(self.tcfg, "teacher_temp", 1.0))
        style_strength = float(_get(self.tcfg, "view_style_strength", 0.05))
        style_knots = int(_get(self.tcfg, "view_style_knots", 8))
        view_warp_max = float(_get(self.tcfg, "view_warp_max", 0.5))
        view_warp_knots = int(_get(self.tcfg, "view_warp_knots", 8))
        gamma = float(_get(self.tcfg, "view_gamma", 5.0))
        warmup_passes = int(_get(self.tcfg, "warmup_passes", 1))
        evidence_interval = max(1, int(_get(self.tcfg, "evidence_interval", 1)))

        self.teacher.eval()

        # Clean view with input gradient retained for saliency.
        x_grad = x.detach().clone().requires_grad_(True)
        clean_feat, clean_logits = self.forward_parts(self.teacher, x_grad)
        clean_prob = torch.softmax(clean_logits / temp, dim=1)

        with torch.no_grad():
            style_x = weak_style_view(x, style_strength, style_knots)
            warp_x = weak_warp_view(x, view_warp_max, view_warp_knots)
            _, style_logits = self.forward_parts(self.teacher, style_x)
            _, warp_logits = self.forward_parts(self.teacher, warp_x)
            style_prob = torch.softmax(style_logits / temp, dim=1)
            warp_prob = torch.softmax(warp_logits / temp, dim=1)

        mean_prob = (clean_prob + style_prob + warp_prob) / 3.0
        pseudo = mean_prob.detach().argmax(dim=1)
        confidence = normalized_teacher_confidence(mean_prob.detach())
        disagreement = mean_js_divergence(
            [clean_prob.detach(), style_prob, warp_prob], mean_prob.detach()
        )
        view_consistency = torch.exp(-gamma * disagreement).clamp(0.0, 1.0)

        use_evidence = (
            current_pass > warmup_passes and iter_num % evidence_interval == 0
        )
        if use_evidence:
            selected_logit = clean_logits.gather(1, pseudo.unsqueeze(1)).sum()
            gradient = torch.autograd.grad(
                selected_logit,
                x_grad,
                retain_graph=False,
                create_graph=False,
                only_inputs=True,
            )[0]
            x_cf, _, _ = to_channel_first_1d(x_grad.detach())
            grad_cf, _, _ = to_channel_first_1d(gradient.detach())
            saliency = (x_cf * grad_cf).abs().mean(dim=1)

            smooth_width = max(1, int(_get(self.tcfg, "evidence_smooth_width", 9)))
            if smooth_width % 2 == 0:
                smooth_width += 1
            saliency = F.avg_pool1d(
                saliency.unsqueeze(1),
                kernel_size=smooth_width,
                stride=1,
                padding=smooth_width // 2,
            ).squeeze(1)

            mask = contiguous_top_band_mask(
                saliency,
                num_bands=int(_get(self.tcfg, "evidence_bands", 4)),
                band_width=int(_get(self.tcfg, "evidence_band_width", 17)),
            )
            destroyed_x = construct_destructive_view(
                x,
                mask,
                background_width=int(
                    _get(self.tcfg, "evidence_background_width", 31)
                ),
            )
            with torch.no_grad():
                _, destroyed_logits = self.forward_parts(self.teacher, destroyed_x)
                destroyed_prob = torch.softmax(destroyed_logits / temp, dim=1)
            original_class_prob = mean_prob.detach().gather(1, pseudo.unsqueeze(1)).squeeze(1)
            destroyed_class_prob = destroyed_prob.gather(1, pseudo.unsqueeze(1)).squeeze(1)
            probability_drop = original_class_prob - destroyed_class_prob
            evidence = robust_evidence_score(probability_drop)
        else:
            probability_drop = torch.zeros_like(confidence)
            evidence = torch.ones_like(confidence)

        reliability = (confidence * view_consistency * evidence).clamp(0.0, 1.0)
        certain = class_balanced_certain_mask(
            reliability,
            pseudo,
            num_classes=self.num_classes,
            min_reliability=float(_get(self.tcfg, "min_reliability", 0.0)),
        )

        return {
            "teacher_prob": mean_prob.detach(),
            "teacher_feature": clean_feat.detach(),
            "pseudo": pseudo.detach(),
            "confidence": confidence.detach(),
            "view_consistency": view_consistency.detach(),
            "evidence": evidence.detach(),
            "probability_drop": probability_drop.detach(),
            "reliability": reliability.detach(),
            "certain": certain.detach(),
            "evidence_active": use_evidence,
        }

    @torch.no_grad()
    def ema_update_teacher(self):
        beta = float(_get(self.tcfg, "ema_beta", 0.99))
        student_params = dict(self.student.named_parameters())
        for name, teacher_param in self.teacher.named_parameters():
            student_param = student_params[name]
            teacher_param.mul_(beta).add_(student_param.detach(), alpha=1.0 - beta)

        student_buffers = dict(self.student.named_buffers())
        for name, teacher_buffer in self.teacher.named_buffers():
            student_buffer = student_buffers[name]
            if torch.is_floating_point(teacher_buffer):
                teacher_buffer.mul_(beta).add_(
                    student_buffer.detach(), alpha=1.0 - beta
                )
            else:
                teacher_buffer.copy_(student_buffer)

    # ------------------------------------------------------------------
    # Main loop

    def adapt(self):
        cfg = self.cfg
        mode = str(_get(self.tcfg, "mode", "core")).lower()
        self.initialize_models()

        before_acc = cal_acc(self.dataloaders["target_data"], self.student)[0]
        print(f"Task: {cfg.Dataset.TL_Task}: Beginning Acc T = {before_acc:.2f}%;")

        optimizer = self.configure_trainable_parameters()
        self._set_student_adaptation_mode()

        passes = int(_get(self.tcfg, "passes", 5))
        interval_iter = len(self.target_dataloader)
        max_iter = passes * interval_iter
        alpha = float(_get(self.tcfg, "alpha", 2.0))
        eta = float(_get(self.tcfg, "eta", 0.2))

        lambda_mt = float(_get(self.tcfg, "lambda_mt", 0.5))
        lambda_pcl = float(_get(self.tcfg, "lambda_pcl", 0.1)) if mode == "full" else 0.0
        lambda_ncl = float(_get(self.tcfg, "lambda_ncl", 0.1)) if mode == "full" else 0.0
        lambda_bn = float(_get(self.tcfg, "lambda_bn", 1e-3))
        lambda_adapter = float(_get(self.tcfg, "lambda_adapter", 1e-3))
        lambda_warp = float(_get(self.tcfg, "lambda_warp", 2e-4))
        log_interval = max(1, int(_get(self.tcfg, "log_interval", 50)))

        print(
            f"[0711] mode={mode}, passes={passes}, warmup_passes="
            f"{int(_get(self.tcfg, 'warmup_passes', 1))}"
        )
        print(
            f"[0711] lambda_mt={lambda_mt}, lambda_pcl={lambda_pcl}, "
            f"lambda_ncl={lambda_ncl}, lambda_bn={lambda_bn}, "
            f"lambda_adapter={lambda_adapter}, lambda_warp={lambda_warp}"
        )

        online_meter = AverageMeter()
        target_iter = iter(self.target_dataloader)

        for iter_num in range(1, max_iter + 1):
            try:
                x, y, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(self.target_dataloader)
                x, y, _ = next(target_iter)

            if x.size(0) <= 1:
                continue

            x = x.to(self.device)
            y = y.to(self.device)
            current_pass = (iter_num - 1) // interval_iter + 1

            reliability_info = self.teacher_reliability(
                x=x,
                current_pass=current_pass,
                iter_num=iter_num,
            )

            optimizer.zero_grad(set_to_none=True)
            student_feature, student_logits = self.forward_parts(self.student, x)

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

            loss_pcl = student_logits.new_tensor(0.0)
            loss_ncl = student_logits.new_tensor(0.0)
            pcl_count = 0
            ncl_count = 0
            if mode == "full" and self.memory is not None and len(self.memory) > 0:
                loss_pcl, pcl_count = prototypical_loss(
                    student_features=student_feature,
                    pseudo_labels=reliability_info["pseudo"],
                    certain_mask=reliability_info["certain"],
                    memory=self.memory,
                    temperature=float(_get(self.tcfg, "pcl_temperature", 0.1)),
                )
                loss_ncl, ncl_count = neighborhood_loss(
                    student_features=student_feature,
                    student_logits=student_logits,
                    uncertain_mask=~reliability_info["certain"],
                    memory=self.memory,
                    neighbors=int(_get(self.tcfg, "ncl_neighbors", 5)),
                    temperature=float(_get(self.tcfg, "ncl_temperature", 0.1)),
                )

            loss_bn = self.bn_anchor_loss()
            loss_adapter = self.adapter_reg_loss()
            loss_warp = self.warp_reg_loss()

            total_loss = (
                loss_sem
                + lambda_mt * loss_mt
                + lambda_pcl * loss_pcl
                + lambda_ncl * loss_ncl
                + lambda_bn * loss_bn
                + lambda_adapter * loss_adapter
                + lambda_warp * loss_warp
            )
            total_loss.backward()
            optimizer.step()
            self.ema_update_teacher()

            # Update memory after using the current historical bank, preventing
            # each sample from becoming its own prototype/nearest neighbor.
            if mode == "full" and self.memory is not None:
                certain = reliability_info["certain"]
                if certain.any():
                    self.memory.update(
                        reliability_info["teacher_feature"][certain],
                        reliability_info["teacher_prob"][certain],
                        reliability_info["reliability"][certain],
                        reliability_info["pseudo"][certain],
                    )

            with torch.no_grad():
                prediction = student_logits.argmax(dim=1)
                batch_acc = (prediction == y).float().mean().item() * 100.0
                online_meter.update(batch_acc, x.size(0))

            should_log = (
                iter_num == 1
                or iter_num % log_interval == 0
                or iter_num == max_iter
                or iter_num % interval_iter == 0
            )
            if should_log:
                certain_ratio = reliability_info["certain"].float().mean().item()
                print(
                    f"iter {iter_num}/{max_iter} | pass={current_pass} | mode={mode} | "
                    f"acc={batch_acc:.2f} | total={total_loss.item():.6f} | "
                    f"sem={loss_sem.item():.6f} | mt={loss_mt.item():.6f} | "
                    f"pcl={loss_pcl.item():.6f}({pcl_count}) | "
                    f"ncl={loss_ncl.item():.6f}({ncl_count}) | "
                    f"rel={reliability_info['reliability'].mean().item():.4f} | "
                    f"certain={certain_ratio:.3f} | "
                    f"conf={reliability_info['confidence'].mean().item():.4f} | "
                    f"agree={reliability_info['view_consistency'].mean().item():.4f} | "
                    f"evidence={reliability_info['evidence'].mean().item():.4f} | "
                    f"drop={reliability_info['probability_drop'].mean().item():.4f} | "
                    f"ev_on={int(reliability_info['evidence_active'])} | "
                    f"bank={len(self.memory) if self.memory is not None else 0} | "
                    f"warp_max={self.warp_delta_max():.4f} | "
                    f"te={sem_info['l_te'].item():.6f} | div={sem_info['l_div'].item():.6f}"
                )

            if self.run is not None and cfg.process_wandb:
                self.run.log(
                    {
                        "loss_total": total_loss.item(),
                        "loss_sem": loss_sem.item(),
                        "loss_mt": loss_mt.item(),
                        "loss_pcl": loss_pcl.item(),
                        "loss_ncl": loss_ncl.item(),
                        "batch_acc": batch_acc,
                        "mean_reliability": reliability_info["reliability"].mean().item(),
                        "certain_ratio": reliability_info["certain"].float().mean().item(),
                        "memory_size": len(self.memory) if self.memory is not None else 0,
                    },
                    step=iter_num,
                )

        print(
            f"Task: {cfg.Dataset.TL_Task}: Online Acc = {online_meter.avg:.2f}%;"
        )
        self.student.eval()
        final_acc = cal_acc(self.dataloaders["target_data"], self.student)[0]
        print(
            f"Task: {cfg.Dataset.TL_Task}: Final Full-Target Acc = {final_acc:.2f}%"
        )
        return {"before": before_acc, "online": online_meter.avg, "final": final_acc}


# -----------------------------------------------------------------------------
# Hydra entry
# -----------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="./Configs", config_name="defaults")
def run(cfg: omegaconf.DictConfig):
    only_task = parse_only_task(getattr(cfg, "only_task", None))
    if only_task is not None:
        task_list = [only_task]
    else:
        task_list = list(permutations(cfg.Dataset.TL_list, 2))

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
                cfg.Opt.lr_tar = float(getattr(cfg.Opt, "lr_tar", 1e-2))
                cfg.Opt.weight_decay_tar = float(
                    getattr(cfg.Opt, "weight_decay_tar", 1e-3)
                )

                if not hasattr(cfg, "TTA0711"):
                    cfg.TTA0711 = {}

                if hasattr(cfg, "wandb") and hasattr(cfg.wandb, "setup"):
                    cfg.wandb.setup.name = f"0711_{task}_seed{seed_run}"
                    cfg.wandb.setup.group = (
                        f"{train_time}_0711_{cfg.Model.model_name}_"
                        f"{getattr(cfg.TTA0711, 'mode', 'core')}"
                    )
                    cfg.wandb.setup.project = f"TTA_0711_{cfg.Dataset.data_name}"

                last_cfg_dict = omegaconf.OmegaConf.to_container(
                    cfg, resolve=True, throw_on_missing=True
                )

            print("=" * 80)
            print(
                f"0711 TTA | dataset={cfg.Dataset.data_name} | task={task} | "
                f"model={cfg.Model.model_name} | seed={seed_run}"
            )
            print("TTA0711 config:", cfg.TTA0711)
            print("=" * 80)

            if cfg.process_wandb:
                run_obj = wandb.init(config=last_cfg_dict, **cfg.wandb.setup)
                with run_obj:
                    trainer = EVMT0711Trainer(cfg, run_obj)
                    trainer.setup()
                    trainer.adapt()
            else:
                trainer = EVMT0711Trainer(cfg, None)
                trainer.setup()
                trainer.adapt()

    print("tasks:", task_list)
    print("final config:")
    pprint(last_cfg_dict)


if __name__ == "__main__":
    run()
