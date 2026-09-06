#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Faithful DtCC target-side utilities for a shared ResNet18 source model.

The equations and update order follow the official DtCC implementation:
- certain/uncertain division from batch-mean confidence and spectral entropy;
- class-balanced memory initialized from classifier weights;
- SEM = Tsallis entropy - prediction diversity;
- PCL with other prototypes and current uncertain features as negatives;
- NCL with nearest memory predictions as positives and other uncertain
  predictions as negatives;
- only BN affine parameters are optimized, with current-batch statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_bcl(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        return x.unsqueeze(1)
    if x.dim() == 3 and x.shape[-1] == 1:
        return x.transpose(1, 2)
    if x.dim() == 3:
        return x
    raise ValueError(f"Expected [B,L], [B,C,L], or [B,L,1], got {tuple(x.shape)}")


def spectral_entropy(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute the DtCC spectral entropy for each Fourier-amplitude sample."""
    x_cf = _to_bcl(x)
    power = x_cf.pow(2).sum(dim=1)
    density = power / power.sum(dim=1, keepdim=True).clamp_min(eps)
    return -(density.clamp_min(eps) * density.clamp_min(eps).log()).sum(dim=1)


def dynamic_data_division(
    probabilities: torch.Tensor,
    entropy_values: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply Eq. (3): high confidence AND low spectral entropy are certain."""
    if probabilities.dim() != 2:
        raise ValueError("probabilities must be [B,K]")
    if entropy_values.dim() != 1 or entropy_values.numel() != probabilities.size(0):
        raise ValueError("entropy_values must be [B]")
    confidence = probabilities.max(dim=1).values
    certain = (confidence >= confidence.mean()) & (entropy_values <= entropy_values.mean())
    return certain, ~certain


def balance_probabilities(
    probabilities: torch.Tensor,
    certain_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply Eq. (11) exactly as the official implementation."""
    classes = probabilities.size(1)
    if certain_mask.any():
        pseudo = probabilities[certain_mask].argmax(dim=1)
        frequency = torch.stack(
            [(pseudo == class_id).float().sum() for class_id in range(classes)]
        ).view(1, -1)
    else:
        frequency = probabilities.new_zeros(1, classes)
    return probabilities / (frequency.to(probabilities.device) + 1.0)


def tsallis_entropy(
    probabilities: torch.Tensor,
    alpha: float = 2.0,
) -> torch.Tensor:
    if abs(float(alpha) - 1.0) < 1e-8:
        return -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=1).mean()
    return (
        1.0 - probabilities.pow(float(alpha)).sum(dim=1)
    ).div(float(alpha) - 1.0).mean()


def diversity_entropy(probabilities: torch.Tensor) -> torch.Tensor:
    mean_prob = probabilities.mean(dim=0)
    return -(mean_prob * mean_prob.clamp_min(1e-5).log()).sum()


def dtcc_sem_loss(
    probabilities: torch.Tensor,
    certain_mask: torch.Tensor,
    alpha: float = 2.0,
) -> torch.Tensor:
    balanced = balance_probabilities(probabilities, certain_mask)
    return tsallis_entropy(balanced, alpha=alpha) - diversity_entropy(balanced)


def dtcc_pcl_loss(
    features_certain: torch.Tensor,
    features_uncertain: torch.Tensor,
    prototypes: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Official DtCC PCL. ``prototypes`` has shape [D,K]."""
    if features_certain.numel() == 0:
        return prototypes.new_tensor(0.0)
    features_certain = F.normalize(features_certain, dim=1)
    pieces = [prototypes]
    if features_uncertain is not None and features_uncertain.numel() > 0:
        pieces.append(F.normalize(features_uncertain, dim=1).T)
    contrast_features = torch.cat(pieces, dim=1)
    logits = features_certain @ contrast_features / float(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values
    return F.nll_loss(F.log_softmax(logits, dim=1), labels.long())


def dtcc_ncl_loss(
    features_uncertain: torch.Tensor,
    supports: torch.Tensor,
    scores: torch.Tensor,
    neighbor_k: int,
    probs_uncertain: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Official DtCC NCL with robust guards for degenerate mini-batches."""
    if features_uncertain.numel() == 0 or supports.numel() == 0:
        return probs_uncertain.new_tensor(0.0)
    count = features_uncertain.size(0)
    k = min(max(1, int(neighbor_k)), supports.size(0))

    similarity = F.normalize(features_uncertain.detach(), dim=1) @ F.normalize(
        supports.detach(), dim=1
    ).T
    near_index = similarity.topk(k=k, dim=1, largest=True).indices
    p_near = scores[near_index]

    if count > 1:
        all_idx = torch.arange(count, device=features_uncertain.device)
        far_idx = all_idx.expand(count, count)
        keep = ~torch.eye(count, dtype=torch.bool, device=features_uncertain.device)
        far_idx = far_idx[keep].view(count, count - 1)
        p_far = probs_uncertain[far_idx]
        p_nf = torch.cat([p_near, p_far], dim=1)
    else:
        p_nf = p_near

    p_query = probs_uncertain.unsqueeze(1)
    logits = torch.bmm(p_query, p_nf.permute(0, 2, 1)) / float(temperature)
    logits = logits - logits.max(dim=2, keepdim=True).values
    normalized = torch.softmax(logits, dim=2)
    positive_mean = normalized[:, :, :k].mean(dim=2).clamp_min(1e-10)
    return -positive_mean.log().mean()


def _classifier_weight(classifier: nn.Module) -> torch.Tensor:
    if hasattr(classifier, "fc") and hasattr(classifier.fc, "weight"):
        return classifier.fc.weight
    if hasattr(classifier, "weight"):
        return classifier.weight
    raise AttributeError("Classifier must expose .fc.weight or .weight")


@dataclass
class DtCCMemoryBank:
    supports: torch.Tensor
    labels: torch.Tensor
    scores: torch.Tensor
    num_classes: int

    @classmethod
    @torch.no_grad()
    def from_classifier(cls, classifier: nn.Module, num_classes: int) -> "DtCCMemoryBank":
        """Initialize class ``k`` with classifier weight ``w_k``.

        DtCC defines the initial class-wise memory as
        ``M_0^k = {(w_s^k, f_s(w_s^k))}``.  Therefore, the k-th classifier
        weight belongs to class k by construction; its predicted score vector
        is stored as auxiliary information but must not be used to relabel the
        support.
        """
        num_classes = int(num_classes)
        supports = _classifier_weight(classifier).detach().clone()
        if supports.dim() != 2 or supports.size(0) != num_classes:
            raise ValueError(
                "Classifier weight rows must equal num_classes: "
                f"got shape={tuple(supports.shape)}, num_classes={num_classes}"
            )
        logits = classifier(supports)
        scores = torch.softmax(logits, dim=1)
        if scores.shape != (num_classes, num_classes):
            raise ValueError(
                "Classifier(supports) must produce [K,K] scores: "
                f"got {tuple(scores.shape)} for K={num_classes}"
            )
        labels = torch.eye(
            num_classes,
            device=supports.device,
            dtype=supports.dtype,
        )
        return cls(
            supports=supports,
            labels=labels,
            scores=scores,
            num_classes=num_classes,
        )

    def snapshot(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.supports.clone(), self.labels.clone(), self.scores.clone()

    @torch.no_grad()
    def update(
        self,
        features: torch.Tensor,
        probabilities: torch.Tensor,
        certain_mask: torch.Tensor,
        base_snapshot: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        base_supports, base_labels, base_scores = (
            base_snapshot if base_snapshot is not None else self.snapshot()
        )
        if certain_mask.any():
            current_features = features[certain_mask].detach()
            current_probs = probabilities[certain_mask].detach()
            current_labels = F.one_hot(
                current_probs.argmax(dim=1), num_classes=self.num_classes
            ).float()
            self.supports = torch.cat([base_supports, current_features], dim=0)
            self.labels = torch.cat([base_labels, current_labels], dim=0)
            self.scores = torch.cat([base_scores, current_probs], dim=0)
        else:
            self.supports = base_supports
            self.labels = base_labels
            self.scores = base_scores

    def prototypes(self) -> torch.Tensor:
        normalized = F.normalize(self.supports, dim=1)
        counts = self.labels.sum(dim=0, keepdim=True).clamp_min(1.0)
        prototypes = (normalized.T @ self.labels) / counts
        return F.normalize(prototypes, dim=0).detach()

    @torch.no_grad()
    def slim(self, max_per_class: int = 50) -> None:
        if int(max_per_class) < 1:
            raise ValueError("max_per_class must be positive")
        confidence = self.scores.max(dim=1).values
        pseudo = self.labels.argmax(dim=1)
        selected = []
        for class_id in range(self.num_classes):
            idx = torch.nonzero(pseudo == class_id, as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            order = confidence[idx].argsort(descending=True)
            selected.append(idx[order[: int(max_per_class)]])
        if not selected:
            return
        keep = torch.cat(selected, dim=0)
        self.supports = self.supports[keep]
        self.labels = self.labels[keep]
        self.scores = self.scores[keep]

    def covered_classes(self) -> int:
        return int((self.labels.sum(dim=0) > 0).sum().item())

    def __len__(self) -> int:
        return int(self.supports.size(0))


def configure_bn_only(model: nn.Module) -> list[nn.Parameter]:
    """Freeze everything except BN affine and force local batch statistics."""
    model.train()
    model.requires_grad_(False)
    params: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.train()
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
            if module.weight is not None:
                module.weight.requires_grad_(True)
                params.append(module.weight)
            if module.bias is not None:
                module.bias.requires_grad_(True)
                params.append(module.bias)
        elif isinstance(module, nn.Dropout):
            module.eval()
    return params
