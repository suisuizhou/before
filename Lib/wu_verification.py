#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnostics and memory variants for validating Wu-style online TTA."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, Tuple

import torch
import torch.nn.functional as F


SUPPORTED_MEMORY_VARIANTS = {
    "ts_only",
    "feature_pre",
    "feature_post",
    "input_reencode",
}


def validate_memory_variant(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in SUPPORTED_MEMORY_VARIANTS:
        raise ValueError(
            f"Unsupported Wu verification variant: {value}; "
            f"expected one of {sorted(SUPPORTED_MEMORY_VARIANTS)}"
        )
    return normalized


@dataclass
class _InputBatch:
    inputs: torch.Tensor
    pseudo: torch.Tensor
    true: torch.Tensor


class RecentInputMemory:
    """FIFO memory storing recent inputs, pseudo-labels, and diagnostic labels."""

    def __init__(self, max_batches: int = 5):
        if int(max_batches) < 1:
            raise ValueError("max_batches must be positive")
        self.max_batches = int(max_batches)
        self._batches: Deque[_InputBatch] = deque(maxlen=self.max_batches)

    @torch.no_grad()
    def append(
        self,
        inputs: torch.Tensor,
        pseudo: torch.Tensor,
        true: torch.Tensor,
    ) -> None:
        self._batches.append(
            _InputBatch(
                inputs=inputs.detach().cpu().clone(),
                pseudo=pseudo.detach().long().cpu().clone(),
                true=true.detach().long().cpu().clone(),
            )
        )

    def all_entries(
        self,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._batches:
            raise RuntimeError("memory is empty")
        inputs = torch.cat([batch.inputs for batch in self._batches], dim=0).to(device)
        pseudo = torch.cat([batch.pseudo for batch in self._batches], dim=0).to(device)
        true = torch.cat([batch.true for batch in self._batches], dim=0).to(device)
        return inputs, pseudo, true

    def covered_classes(self) -> int:
        if not self._batches:
            return 0
        pseudo = torch.cat([batch.pseudo for batch in self._batches], dim=0)
        return int(pseudo.unique().numel())

    def pseudo_accuracy(self) -> float:
        if not self._batches:
            return 0.0
        pseudo = torch.cat([batch.pseudo for batch in self._batches], dim=0)
        true = torch.cat([batch.true for batch in self._batches], dim=0)
        return float((pseudo == true).float().mean().item() * 100.0)

    def __len__(self) -> int:
        return len(self._batches)



@dataclass
class _FeatureBatch:
    features: torch.Tensor
    pseudo: torch.Tensor
    true: torch.Tensor


class RecentFeatureMemory:
    """FIFO memory storing feature snapshots and diagnostic labels."""

    def __init__(self, max_batches: int = 5):
        if int(max_batches) < 1:
            raise ValueError("max_batches must be positive")
        self.max_batches = int(max_batches)
        self._batches: Deque[_FeatureBatch] = deque(maxlen=self.max_batches)

    @torch.no_grad()
    def append(
        self,
        features: torch.Tensor,
        pseudo: torch.Tensor,
        true: torch.Tensor,
    ) -> None:
        self._batches.append(
            _FeatureBatch(
                features=features.detach().cpu().clone(),
                pseudo=pseudo.detach().long().cpu().clone(),
                true=true.detach().long().cpu().clone(),
            )
        )

    def all_entries(
        self,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._batches:
            raise RuntimeError("memory is empty")
        features = torch.cat([batch.features for batch in self._batches], dim=0).to(device)
        pseudo = torch.cat([batch.pseudo for batch in self._batches], dim=0).to(device)
        true = torch.cat([batch.true for batch in self._batches], dim=0).to(device)
        return features, pseudo, true

    def covered_classes(self) -> int:
        if not self._batches:
            return 0
        pseudo = torch.cat([batch.pseudo for batch in self._batches], dim=0)
        return int(pseudo.unique().numel())

    def pseudo_accuracy(self) -> float:
        if not self._batches:
            return 0.0
        pseudo = torch.cat([batch.pseudo for batch in self._batches], dim=0)
        true = torch.cat([batch.true for batch in self._batches], dim=0)
        return float((pseudo == true).float().mean().item() * 100.0)

    def __len__(self) -> int:
        return len(self._batches)

def _centroids_from_features(
    features: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[int, torch.Tensor]:
    return {
        int(class_id): features[labels == int(class_id)].mean(dim=0)
        for class_id in labels.unique().tolist()
    }


def reencoded_feature_aggregation_loss(
    current_features: torch.Tensor,
    pseudo: torch.Tensor,
    memory: RecentInputMemory,
    feature_fn: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    reencode_batch_size: int = 128,
) -> Tuple[torch.Tensor, int]:
    """Re-encode historical inputs with the current student and use detached centroids."""
    if len(memory) == 0 or current_features.numel() == 0:
        return current_features.new_tensor(0.0), 0
    memory_inputs, memory_pseudo, _ = memory.all_entries(device)
    batch_size = int(reencode_batch_size)
    if batch_size < 1:
        raise ValueError("reencode_batch_size must be positive")
    encoded_chunks = []
    with torch.no_grad():
        for start in range(0, memory_inputs.size(0), batch_size):
            encoded_chunks.append(feature_fn(memory_inputs[start:start + batch_size]).detach())
        memory_features = torch.cat(encoded_chunks, dim=0)
        centroids = _centroids_from_features(memory_features, memory_pseudo)

    selected_features = []
    selected_centroids = []
    for feature, label in zip(current_features, pseudo):
        class_id = int(label.item())
        if class_id in centroids:
            selected_features.append(feature)
            selected_centroids.append(centroids[class_id])
    if not selected_features:
        return current_features.new_tensor(0.0), 0
    features = F.normalize(torch.stack(selected_features), dim=1)
    centroid_tensor = F.normalize(torch.stack(selected_centroids), dim=1)
    loss = (1.0 - (features * centroid_tensor).sum(dim=1)).mean()
    return loss, len(selected_features)


class PseudoDiagnostics:
    """Streaming diagnostic metrics that never participate in optimization."""

    def __init__(self):
        self.samples = 0
        self.pseudo_correct = 0
        self.confidence_sum = 0.0
        self.agreement = 0
        self.fa_used_samples = 0

    @torch.no_grad()
    def update_batch(
        self,
        teacher_prob: torch.Tensor,
        pseudo: torch.Tensor,
        truth: torch.Tensor,
        student_prediction: torch.Tensor,
        fa_used_samples: int,
        batch_size: int,
    ) -> None:
        count = int(batch_size)
        if count <= 0:
            return
        self.samples += count
        self.pseudo_correct += int((pseudo == truth).sum().item())
        self.confidence_sum += float(teacher_prob.max(dim=1).values.sum().item())
        self.agreement += int((pseudo == student_prediction).sum().item())
        self.fa_used_samples += int(fa_used_samples)

    def metrics(self) -> Dict[str, float]:
        denominator = max(self.samples, 1)
        return {
            "teacher_pseudo_accuracy": 100.0 * self.pseudo_correct / denominator,
            "teacher_mean_confidence": 100.0 * self.confidence_sum / denominator,
            "teacher_student_agreement": 100.0 * self.agreement / denominator,
            "fa_sample_ratio": 100.0 * self.fa_used_samples / denominator,
        }



def feature_snapshot_aggregation_loss(
    current_features: torch.Tensor,
    pseudo: torch.Tensor,
    memory: RecentFeatureMemory,
) -> Tuple[torch.Tensor, int]:
    if len(memory) == 0 or current_features.numel() == 0:
        return current_features.new_tensor(0.0), 0
    memory_features, memory_pseudo, _ = memory.all_entries(current_features.device)
    centroids = _centroids_from_features(memory_features, memory_pseudo)
    selected_features = []
    selected_centroids = []
    for feature, label in zip(current_features, pseudo):
        class_id = int(label.item())
        if class_id in centroids:
            selected_features.append(feature)
            selected_centroids.append(centroids[class_id])
    if not selected_features:
        return current_features.new_tensor(0.0), 0
    features = F.normalize(torch.stack(selected_features), dim=1)
    centroid_tensor = F.normalize(torch.stack(selected_centroids), dim=1)
    loss = (1.0 - (features * centroid_tensor).sum(dim=1)).mean()
    return loss, len(selected_features)


def compute_wu_feature_loss(
    variant: str,
    enabled: bool,
    current_features: torch.Tensor,
    pseudo: torch.Tensor,
    feature_memory: RecentFeatureMemory,
    input_memory: RecentInputMemory,
    feature_fn: Callable[[torch.Tensor], torch.Tensor],
    device: torch.device,
    reencode_batch_size: int = 128,
) -> Tuple[torch.Tensor, int]:
    selected = validate_memory_variant(variant)
    if not enabled or selected == "ts_only":
        return current_features.new_tensor(0.0), 0
    if selected in {"feature_pre", "feature_post"}:
        return feature_snapshot_aggregation_loss(current_features, pseudo, feature_memory)
    return reencoded_feature_aggregation_loss(
        current_features=current_features,
        pseudo=pseudo,
        memory=input_memory,
        feature_fn=feature_fn,
        device=device,
        reencode_batch_size=reencode_batch_size,
    )
