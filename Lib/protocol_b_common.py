#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared contracts for PU4D Protocol-B source and target experiments."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def method_checkpoint_dir(
    root: Path | str,
    method: str,
    dataset: str,
    source: int,
    seed: int,
) -> Path:
    method_name = str(method).strip().upper()
    if method_name not in {"DTCC", "WU"}:
        raise ValueError(f"Unsupported Protocol-B method: {method}")
    return Path(root) / method_name / str(dataset) / f"source_{int(source)}" / f"seed_{int(seed)}"


def _unwrap_state_dict(obj):
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        raise TypeError(f"Checkpoint must contain a state dict, got {type(obj)}")
    if obj and all(str(k).startswith("module.") for k in obj):
        obj = {str(k)[7:]: v for k, v in obj.items()}
    return obj


def strict_load_state_dict(model: nn.Module, path: Path | str, device: torch.device) -> None:
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Protocol-B checkpoint not found: {checkpoint}")
    state = _unwrap_state_dict(torch.load(checkpoint, map_location=device))
    current = model.state_dict()
    missing = sorted(set(current) - set(state))
    unexpected = sorted(set(state) - set(current))
    mismatched = [
        (key, tuple(state[key].shape), tuple(current[key].shape))
        for key in sorted(set(current) & set(state))
        if tuple(state[key].shape) != tuple(current[key].shape)
    ]
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "Protocol-B strict checkpoint mismatch: "
            f"missing={missing[:10]} unexpected={unexpected[:10]} mismatched={mismatched[:10]}"
        )
    model.load_state_dict(state, strict=True)
    print(f"[PROTOCOL-B STRICT LOAD] checkpoint={checkpoint} tensors={len(state)}")


def dtcc_lr_multiplier(progress: float) -> float:
    p = float(progress)
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"progress must be in [0,1], got {progress}")
    return (1.0 + p) ** (-1.0 / math.log(2.0))


def set_optimizer_lr(optimizer: torch.optim.Optimizer, initial_lr: float, progress: float) -> float:
    lr = float(initial_lr) * dtcc_lr_multiplier(progress)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def label_smoothed_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    num_classes: int,
) -> torch.Tensor:
    if logits.ndim != 2:
        raise ValueError("logits must have shape [B,K]")
    if int(num_classes) != logits.size(1):
        raise ValueError("num_classes must match logits.shape[1]")
    eps = float(epsilon)
    if not 0.0 <= eps < 1.0:
        raise ValueError("epsilon must be in [0,1)")
    with torch.no_grad():
        target = torch.full_like(logits, eps / int(num_classes))
        target.scatter_(1, labels.long().view(-1, 1), 1.0 - eps + eps / int(num_classes))
    return -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def four_view_cross_entropy(
    model: nn.Module,
    views: Sequence[torch.Tensor],
    labels: torch.Tensor,
) -> torch.Tensor:
    if len(views) != 4:
        raise ValueError(f"Wu source loss requires exactly four views, got {len(views)}")
    losses = [F.cross_entropy(model(view), labels.long()) for view in views]
    return torch.stack(losses).mean()


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def save_training_metadata(path: Path | str, metadata: Dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def read_training_metadata(path: Path | str) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
