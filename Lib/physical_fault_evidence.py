#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Physical fault-spectrum evidence for PU4D strict 0711 TTA.

The cache uses a 1024-point FFT and keeps bins 0..511. With fs=64 kHz,
frequency resolution is 62.5 Hz/bin and DC is retained.

Label groups follow Dataset/PU4D.py:
  0..5   healthy
  6..17  outer-race faults
  18..20 combined inner/outer-race faults
  21..31 inner-race faults
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, Sequence, Tuple

import torch
import torch.nn.functional as F


DOMAIN_SPEED_RPM: Dict[int, float] = {
    0: 1500.0,
    1: 900.0,
    2: 1500.0,
    3: 1500.0,
}

HEALTH_LABELS = frozenset(range(0, 6))
OUTER_RACE_LABELS = frozenset(range(6, 18))
COMBINED_LABELS = frozenset({18, 19, 20})
INNER_RACE_LABELS = frozenset(range(21, 32))


@dataclass(frozen=True)
class BearingGeometry:
    rolling_elements: int = 8
    rolling_element_diameter_mm: float = 6.75
    pitch_diameter_mm: float = 28.55
    contact_angle_deg: float = 0.0


@dataclass(frozen=True)
class PhysicalEvidenceConfig:
    sampling_rate_hz: float = 64000.0
    fft_size: int = 1024
    spectrum_length: int = 512
    harmonics: int = 8
    outer_sideband_orders: Tuple[int, ...] = (0, 1)
    inner_sideband_orders: Tuple[int, ...] = (0, 1, 2)
    mask_sigma_bins: float = 1.0
    background_width_bins: int = 7
    max_mask_ratio: float = 0.18
    mask_activity_threshold: float = 0.10
    exclude_dc: bool = True

    def validate(self) -> None:
        if self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive")
        if self.fft_size < 2 or self.fft_size % 2 != 0:
            raise ValueError("fft_size must be an even integer >= 2")
        if self.spectrum_length < 2 or self.spectrum_length > self.fft_size // 2:
            raise ValueError("spectrum_length must be in [2, fft_size/2]")
        if self.harmonics < 1:
            raise ValueError("harmonics must be >= 1")
        if self.mask_sigma_bins <= 0:
            raise ValueError("mask_sigma_bins must be positive")
        if not 0 < self.max_mask_ratio <= 1:
            raise ValueError("max_mask_ratio must be in (0, 1]")
        if not 0 < self.mask_activity_threshold < 1:
            raise ValueError("mask_activity_threshold must be in (0, 1)")


DEFAULT_GEOMETRY = BearingGeometry()
DEFAULT_CONFIG = PhysicalEvidenceConfig()


def fault_group_from_label(label: int) -> str:
    label = int(label)
    if label in HEALTH_LABELS:
        return "healthy"
    if label in OUTER_RACE_LABELS:
        return "outer"
    if label in COMBINED_LABELS:
        return "combined"
    if label in INNER_RACE_LABELS:
        return "inner"
    raise ValueError(f"Unsupported PU4D label: {label}")


def characteristic_frequencies(
    speed_rpm: float,
    geometry: BearingGeometry = DEFAULT_GEOMETRY,
) -> Dict[str, float]:
    """Return shaft and bearing kinematic frequencies in hertz."""
    if speed_rpm <= 0:
        raise ValueError("speed_rpm must be positive")
    z = float(geometry.rolling_elements)
    d = float(geometry.rolling_element_diameter_mm)
    D = float(geometry.pitch_diameter_mm)
    if z <= 0 or d <= 0 or D <= 0 or d >= D:
        raise ValueError("Invalid bearing geometry")

    fr = float(speed_rpm) / 60.0
    theta = math.radians(float(geometry.contact_angle_deg))
    ratio = (d / D) * math.cos(theta)

    bpfo = 0.5 * z * fr * (1.0 - ratio)
    bpfi = 0.5 * z * fr * (1.0 + ratio)
    bsf = (D / (2.0 * d)) * fr * (1.0 - ratio * ratio)
    ftf = 0.5 * fr * (1.0 - ratio)
    return {
        "shaft": fr,
        "bpfo": bpfo,
        "bpfi": bpfi,
        "bsf": bsf,
        "ftf": ftf,
    }


def _candidate_frequencies(
    base_frequency_hz: float,
    shaft_frequency_hz: float,
    harmonics: int,
    sideband_orders: Sequence[int],
    max_frequency_hz: float,
) -> Tuple[float, ...]:
    values = set()
    for harmonic in range(1, int(harmonics) + 1):
        center = float(harmonic) * float(base_frequency_hz)
        for order in sideband_orders:
            order = int(order)
            offsets = (0.0,) if order == 0 else (
                -float(order) * shaft_frequency_hz,
                float(order) * shaft_frequency_hz,
            )
            for offset in offsets:
                frequency = center + offset
                if 0.0 < frequency <= max_frequency_hz:
                    values.add(round(float(frequency), 10))
    return tuple(sorted(values))


def _soft_frequency_mask(
    center_frequencies_hz: Iterable[float],
    config: PhysicalEvidenceConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Construct one soft mask of shape [L] from physical center frequencies."""
    centers = tuple(float(v) for v in center_frequencies_hz)
    if not centers:
        return torch.zeros(config.spectrum_length, device=device, dtype=dtype)

    resolution_hz = config.sampling_rate_hz / float(config.fft_size)
    bin_axis = torch.arange(config.spectrum_length, device=device, dtype=dtype)
    center_bins = torch.tensor(centers, device=device, dtype=dtype) / resolution_hz
    distance = bin_axis.unsqueeze(0) - center_bins.unsqueeze(1)
    masks = torch.exp(-0.5 * (distance / float(config.mask_sigma_bins)).pow(2))
    mask = masks.amax(dim=0).clamp(0.0, 1.0)

    if config.exclude_dc:
        mask[0] = 0.0

    active = mask >= float(config.mask_activity_threshold)
    max_active = max(1, int(math.floor(config.max_mask_ratio * config.spectrum_length)))
    if int(active.sum().item()) > max_active:
        keep_idx = torch.topk(mask, k=max_active, largest=True).indices
        clipped = torch.zeros_like(mask)
        clipped[keep_idx] = mask[keep_idx]
        mask = clipped
    return mask


def build_physical_masks(
    pseudo_labels: torch.Tensor,
    target_domain: int,
    config: PhysicalEvidenceConfig = DEFAULT_CONFIG,
    geometry: BearingGeometry = DEFAULT_GEOMETRY,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``mask [B,L]`` and ``applicable [B]`` for PU4D pseudo-labels."""
    config.validate()
    if pseudo_labels.dim() != 1:
        raise ValueError("pseudo_labels must be 1-D")
    if int(target_domain) not in DOMAIN_SPEED_RPM:
        raise ValueError(f"Unsupported PU4D target domain: {target_domain}")

    labels = pseudo_labels.to(dtype=torch.long)
    batch = labels.numel()
    device = labels.device
    dtype = torch.float32

    speed_rpm = DOMAIN_SPEED_RPM[int(target_domain)]
    freq = characteristic_frequencies(speed_rpm, geometry)
    resolution_hz = config.sampling_rate_hz / float(config.fft_size)
    max_frequency_hz = (config.spectrum_length - 1) * resolution_hz

    outer_freqs = _candidate_frequencies(
        freq["bpfo"],
        freq["shaft"],
        config.harmonics,
        config.outer_sideband_orders,
        max_frequency_hz,
    )
    inner_freqs = _candidate_frequencies(
        freq["bpfi"],
        freq["shaft"],
        config.harmonics,
        config.inner_sideband_orders,
        max_frequency_hz,
    )

    outer_mask = _soft_frequency_mask(
        outer_freqs, config, device=device, dtype=dtype
    )
    inner_mask = _soft_frequency_mask(
        inner_freqs, config, device=device, dtype=dtype
    )
    combined_mask = torch.maximum(outer_mask, inner_mask)

    masks = torch.zeros(
        batch,
        config.spectrum_length,
        device=device,
        dtype=dtype,
    )
    applicable = torch.zeros(batch, device=device, dtype=torch.bool)

    outer_sel = (labels >= 6) & (labels <= 17)
    combined_sel = (labels >= 18) & (labels <= 20)
    inner_sel = (labels >= 21) & (labels <= 31)

    masks[outer_sel] = outer_mask
    masks[combined_sel] = combined_mask
    masks[inner_sel] = inner_mask
    applicable = outer_sel | combined_sel | inner_sel
    return masks, applicable


def _to_channel_first_1d(x: torch.Tensor) -> Tuple[torch.Tensor, bool, bool]:
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


def construct_physical_destructive_view(
    x: torch.Tensor,
    mask: torch.Tensor,
    background_width_bins: int = 7,
) -> torch.Tensor:
    """Replace physically selected bins with a local smooth background."""
    x_cf, squeezed, transposed = _to_channel_first_1d(x)
    if mask.dim() != 2 or mask.shape[0] != x_cf.shape[0] or mask.shape[1] != x_cf.shape[-1]:
        raise ValueError(
            f"mask must be [B,L]={x_cf.shape[0], x_cf.shape[-1]}, got {tuple(mask.shape)}"
        )

    width = max(3, int(background_width_bins))
    if width % 2 == 0:
        width += 1
    pad = width // 2
    padded = F.pad(x_cf, (pad, pad), mode="replicate")
    background = F.avg_pool1d(padded, kernel_size=width, stride=1)

    mask_cf = mask.to(device=x_cf.device, dtype=x_cf.dtype).unsqueeze(1)
    destroyed = (1.0 - mask_cf) * x_cf + mask_cf * background
    return _restore_1d_shape(destroyed, squeezed, transposed)


def class_margin(probability: torch.Tensor, pseudo_labels: torch.Tensor) -> torch.Tensor:
    """Return q_y - max(q_other) for each sample."""
    if probability.dim() != 2 or pseudo_labels.dim() != 1:
        raise ValueError("probability must be [B,C] and pseudo_labels must be [B]")
    if probability.size(0) != pseudo_labels.numel():
        raise ValueError("Batch dimensions do not match")

    labels = pseudo_labels.to(device=probability.device, dtype=torch.long)
    selected = probability.gather(1, labels.unsqueeze(1)).squeeze(1)
    other = probability.clone()
    other.scatter_(1, labels.unsqueeze(1), float("-inf"))
    runner_up = other.max(dim=1).values
    return selected - runner_up


def robust_fault_evidence_score(
    evidence_drop: torch.Tensor,
    applicable: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """MAD-normalize fault-only drops; healthy predictions receive neutral 1.0."""
    if evidence_drop.dim() != 1 or applicable.dim() != 1:
        raise ValueError("evidence_drop and applicable must be 1-D")
    if evidence_drop.numel() != applicable.numel():
        raise ValueError("evidence_drop and applicable must have equal length")

    score = torch.ones_like(evidence_drop)
    values = evidence_drop[applicable]
    if values.numel() == 0:
        return score

    median = values.median()
    mad = (values - median).abs().median()
    robust_scale = 1.4826 * mad + float(eps)
    score[applicable] = torch.sigmoid((values - median) / robust_scale)
    return score.clamp(0.0, 1.0)


def mask_active_ratio(mask: torch.Tensor, threshold: float = 0.10) -> torch.Tensor:
    if mask.dim() != 2:
        raise ValueError("mask must be [B,L]")
    return (mask >= float(threshold)).to(mask.dtype).mean(dim=1)
