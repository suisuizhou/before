#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Target-side utilities for Wu et al. with a shared ResNet18 source model."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_noise(x: torch.Tensor, std: float = 0.2) -> torch.Tensor:
    if std < 0:
        raise ValueError("std must be non-negative")
    if std == 0:
        return x.clone()
    return x + torch.randn_like(x) * float(std)


def uniform_noise(x: torch.Tensor, amplitude: float = 0.1) -> torch.Tensor:
    if amplitude < 0:
        raise ValueError("amplitude must be non-negative")
    if amplitude == 0:
        return x.clone()
    return x + torch.empty_like(x).uniform_(-float(amplitude), float(amplitude))


def impulse_noise(
    x: torch.Tensor,
    probability: float = 0.2,
    intensity: float = 0.2,
) -> torch.Tensor:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0,1]")
    if intensity < 0:
        raise ValueError("intensity must be non-negative")
    if probability == 0 or intensity == 0:
        return x.clone()
    out = x.clone()
    hit = torch.rand_like(out) < float(probability)
    sign = torch.where(
        torch.rand_like(out) < 0.5,
        out.new_tensor(-1.0),
        out.new_tensor(1.0),
    )
    return torch.where(hit, out + sign * float(intensity), out)


def soft_target_cross_entropy(
    student_logits: torch.Tensor,
    teacher_probabilities: torch.Tensor,
) -> torch.Tensor:
    target = teacher_probabilities.detach()
    return -(target * F.log_softmax(student_logits, dim=1)).sum(dim=1).mean()


@dataclass
class _MemoryBatch:
    features: torch.Tensor
    pseudo: torch.Tensor


class RecentBatchMemory:
    """FIFO memory holding features and pseudo-labels from recent time points."""

    def __init__(self, max_batches: int = 5):
        if int(max_batches) < 1:
            raise ValueError("max_batches must be positive")
        self.max_batches = int(max_batches)
        self._batches: Deque[_MemoryBatch] = deque(maxlen=self.max_batches)

    @torch.no_grad()
    def append(self, features: torch.Tensor, pseudo: torch.Tensor) -> None:
        self._batches.append(
            _MemoryBatch(features=features.detach().cpu().clone(), pseudo=pseudo.detach().cpu().clone())
        )

    def all_entries(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._batches:
            raise RuntimeError("memory is empty")
        features = torch.cat([batch.features for batch in self._batches], dim=0).to(device)
        pseudo = torch.cat([batch.pseudo for batch in self._batches], dim=0).to(device)
        return features, pseudo

    def covered_classes(self) -> int:
        if not self._batches:
            return 0
        labels = torch.cat([batch.pseudo for batch in self._batches], dim=0)
        return int(labels.unique().numel())

    def __len__(self) -> int:
        return len(self._batches)


def feature_aggregation_loss(
    current_features: torch.Tensor,
    pseudo: torch.Tensor,
    memory: RecentBatchMemory,
) -> torch.Tensor:
    """Eq. (7): mean cosine distance to the corresponding memory centroid."""
    if len(memory) == 0 or current_features.numel() == 0:
        return current_features.new_tensor(0.0)
    mem_features, mem_pseudo = memory.all_entries(current_features.device)
    centroids = {}
    for class_id in mem_pseudo.unique().tolist():
        class_mask = mem_pseudo == int(class_id)
        centroids[int(class_id)] = mem_features[class_mask].mean(dim=0)

    selected_features = []
    selected_centroids = []
    for feature, label in zip(current_features, pseudo):
        class_id = int(label.item())
        if class_id in centroids:
            selected_features.append(feature)
            selected_centroids.append(centroids[class_id])
    if not selected_features:
        return current_features.new_tensor(0.0)
    feat = F.normalize(torch.stack(selected_features), dim=1)
    centroid = F.normalize(torch.stack(selected_centroids), dim=1)
    return (1.0 - (feat * centroid).sum(dim=1)).mean()


@torch.no_grad()
def ema_update_model(teacher: nn.Module, student: nn.Module, alpha: float = 0.999) -> None:
    if not 0.0 <= alpha < 1.0:
        raise ValueError("alpha must be in [0,1)")
    student_params = dict(student.named_parameters())
    for name, teacher_param in teacher.named_parameters():
        teacher_param.mul_(float(alpha)).add_(
            student_params[name].detach(), alpha=1.0 - float(alpha)
        )
    student_buffers = dict(student.named_buffers())
    for name, teacher_buffer in teacher.named_buffers():
        teacher_buffer.copy_(student_buffers[name].detach())
