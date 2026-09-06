#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""Utilities for class-balanced reliability-weighted SEM.

This module is deliberately independent from the project trainer so its
selection, agreement, augmentation, and loss behavior can be unit-tested.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Sequence, Tuple

import torch
import torch.nn.functional as F


def _to_channel_first_1d(x: torch.Tensor) -> Tuple[torch.Tensor, bool, bool]:
    """Return [B, C, L] plus flags needed to restore the original shape."""
    squeezed = False
    transposed = False
    if x.dim() == 2:
        x = x.unsqueeze(1)
        squeezed = True
    elif x.dim() == 3 and x.shape[-1] == 1:
        x = x.transpose(1, 2)
        transposed = True
    elif x.dim() != 3:
        raise ValueError(f"Expected [B,L], [B,C,L], or [B,L,1], got {tuple(x.shape)}")
    return x, squeezed, transposed


def _restore_1d_shape(
    x: torch.Tensor, squeezed: bool, transposed: bool
) -> torch.Tensor:
    if transposed:
        x = x.transpose(1, 2)
    if squeezed:
        x = x.squeeze(1)
    return x


def weak_spectral_style(
    x: torch.Tensor,
    strength: float = 0.05,
    knots: int = 8,
) -> torch.Tensor:
    """Apply a weak, smooth multiplicative envelope to each sample.

    The perturbation is expressed in log-amplitude space, so the multiplier is
    always positive. ``strength=0`` is exactly the identity transform.
    """
    if strength < 0:
        raise ValueError("strength must be non-negative")
    if knots < 2:
        raise ValueError("knots must be at least 2")
    if strength == 0:
        return x.clone()

    x_cf, squeezed, transposed = _to_channel_first_1d(x)
    batch, channels, length = x_cf.shape
    ctrl = torch.randn(
        batch,
        1,
        knots,
        device=x_cf.device,
        dtype=x_cf.dtype,
    ) * float(strength)
    envelope = F.interpolate(ctrl, size=length, mode="linear", align_corners=True)
    envelope = torch.exp(envelope).expand(-1, channels, -1)
    out = x_cf * envelope
    return _restore_1d_shape(out, squeezed, transposed)


def weak_spectral_warp(
    x: torch.Tensor,
    max_shift: float = 0.5,
    knots: int = 8,
) -> torch.Tensor:
    """Apply a weak, smooth, per-sample frequency-axis displacement."""
    if max_shift < 0:
        raise ValueError("max_shift must be non-negative")
    if knots < 2:
        raise ValueError("knots must be at least 2")
    if max_shift == 0:
        return x.clone()

    x_cf, squeezed, transposed = _to_channel_first_1d(x)
    batch, _, length = x_cf.shape

    ctrl = torch.empty(
        batch,
        1,
        knots,
        device=x_cf.device,
        dtype=x_cf.dtype,
    ).uniform_(-1.0, 1.0)
    delta = F.interpolate(ctrl, size=length, mode="linear", align_corners=True)
    delta = delta * float(max_shift)

    base = torch.arange(length, device=x_cf.device, dtype=x_cf.dtype).view(1, 1, -1)
    sample_pos = (base + delta).clamp(0.0, float(length - 1))
    x_norm = 2.0 * sample_pos / float(max(length - 1, 1)) - 1.0
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
    return _restore_1d_shape(out, squeezed, transposed)


def js_agreement(logits_list: Sequence[torch.Tensor], gamma: float = 5.0) -> torch.Tensor:
    """Convert multi-view Jensen-Shannon divergence into [0,1] agreement."""
    if len(logits_list) < 2:
        raise ValueError("At least two views are required")
    if gamma < 0:
        raise ValueError("gamma must be non-negative")

    probabilities = [torch.softmax(logits, dim=1) for logits in logits_list]
    shape = probabilities[0].shape
    if any(p.shape != shape for p in probabilities[1:]):
        raise ValueError("All view logits must have the same shape")

    mean_prob = torch.stack(probabilities, dim=0).mean(dim=0).clamp_min(1e-8)
    js_terms = []
    for prob in probabilities:
        prob = prob.clamp_min(1e-8)
        js_terms.append((prob * (prob.log() - mean_prob.log())).sum(dim=1))
    js = torch.stack(js_terms, dim=0).mean(dim=0)
    return torch.exp(-float(gamma) * js).clamp(0.0, 1.0)


def class_balanced_top_mask(
    reliability: torch.Tensor,
    confidence: torch.Tensor,
    pseudo: torch.Tensor,
    num_classes: int,
    keep_ratio: float = 0.5,
    min_per_class: int = 1,
    min_confidence: float = 0.5,
) -> torch.Tensor:
    """Select the most reliable samples independently inside each class.

    The confidence threshold is applied before ranking. If it rejects the
    entire batch, a global reliability top-k fallback keeps adaptation active.
    """
    if reliability.dim() != 1 or confidence.dim() != 1 or pseudo.dim() != 1:
        raise ValueError("reliability, confidence, and pseudo must be 1-D")
    if not (reliability.numel() == confidence.numel() == pseudo.numel()):
        raise ValueError("reliability, confidence, and pseudo must have equal length")
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0,1]")
    if min_per_class < 1:
        raise ValueError("min_per_class must be at least 1")
    if num_classes < 1:
        raise ValueError("num_classes must be positive")

    mask = torch.zeros_like(reliability, dtype=torch.bool)
    for class_id in range(int(num_classes)):
        candidates = torch.nonzero(
            (pseudo == class_id) & (confidence >= float(min_confidence)),
            as_tuple=False,
        ).flatten()
        if candidates.numel() == 0:
            continue
        keep = max(int(min_per_class), int(math.ceil(float(keep_ratio) * candidates.numel())))
        keep = min(keep, candidates.numel())
        local_order = torch.topk(reliability[candidates], k=keep, largest=True).indices
        mask[candidates[local_order]] = True

    if not mask.any():
        keep = max(1, int(math.ceil(float(keep_ratio) * reliability.numel())))
        keep = min(keep, reliability.numel())
        fallback = torch.topk(reliability, k=keep, largest=True).indices
        mask[fallback] = True

    return mask


def weighted_sem_loss(
    logits: torch.Tensor,
    weights: torch.Tensor,
    pseudo: torch.Tensor,
    alpha: float = 2.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Class-aware Tsallis entropy plus diversity using hard-gated weights."""
    if logits.dim() != 2:
        raise ValueError("logits must have shape [B,K]")
    if weights.dim() != 1 or pseudo.dim() != 1:
        raise ValueError("weights and pseudo must be 1-D")
    if logits.size(0) != weights.numel() or weights.numel() != pseudo.numel():
        raise ValueError("Batch dimensions do not match")

    prob = torch.softmax(logits, dim=1)
    num_classes = prob.size(1)
    weights = weights.to(device=prob.device, dtype=prob.dtype).clamp_min(0.0)
    pseudo = pseudo.to(device=prob.device, dtype=torch.long)
    selected = weights > 0

    freq = torch.bincount(pseudo[selected], minlength=num_classes).to(prob.dtype)
    hat_p = prob / (freq.unsqueeze(0) + 1.0)
    hat_p = hat_p / hat_p.sum(dim=1, keepdim=True).clamp_min(1e-12)

    if abs(float(alpha) - 1.0) < 1e-6:
        per_sample_te = -(hat_p * torch.log(hat_p.clamp_min(1e-8))).sum(dim=1)
    else:
        per_sample_te = (
            1.0 - hat_p.clamp_min(1e-8).pow(float(alpha)).sum(dim=1)
        ) / (float(alpha) - 1.0)

    weight_sum = weights.sum().clamp_min(1e-8)
    l_te = (weights * per_sample_te).sum() / weight_sum

    p_bar = (weights.unsqueeze(1) * hat_p).sum(dim=0) / weight_sum
    p_bar = p_bar / p_bar.sum().clamp_min(1e-12)
    l_div = (p_bar * torch.log(p_bar.clamp_min(1e-8))).sum()
    loss = l_te + l_div

    info = {
        "l_te": float(l_te.detach().item()),
        "l_div": float(l_div.detach().item()),
        "selected_ratio": float(selected.sum().item() / max(selected.numel(), 1)),
        "weight_mean": float(weights.mean().item()),
        "weight_selected_mean": float(weights[selected].mean().item()) if selected.any() else 0.0,
    }
    return loss, info
