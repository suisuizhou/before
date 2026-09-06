#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HUST-specific bearing-frequency evidence for strict 0711 adaptation.

The strict HUST cache retains 512 bins from a 2048-point FFT at 51.2 kHz,
so its frequency grid is 25 Hz.  Unlike the older fixed-speed evidence path,
this module uses the shaft frequency stored with every cached sample.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
import operator
from typing import Dict, Sequence, Tuple

import torch

from Lib.physical_fault_evidence import (
    PhysicalEvidenceConfig,
    class_margin,
    construct_physical_destructive_view,
    mask_active_ratio,
    robust_fault_evidence_score,
)


@dataclass(frozen=True)
class BearingGeometry:
    rolling_elements: int
    rolling_element_diameter_mm: float
    pitch_diameter_mm: float
    contact_angle_deg: float = 0.0


# Domain IDs are the primary bearing split: 6205, 6206, 6207, and 6208.
# The pitch diameter is approximated by (inner diameter + outer diameter) / 2.
HUST_GEOMETRIES: Dict[int, BearingGeometry] = {
    0: BearingGeometry(9, 7.8, (25.0 + 52.0) / 2.0),
    1: BearingGeometry(9, 9.0, (30.0 + 62.0) / 2.0),
    2: BearingGeometry(9, 11.0, (35.0 + 72.0) / 2.0),
    3: BearingGeometry(9, 12.0, (40.0 + 80.0) / 2.0),
}


@dataclass(frozen=True)
class HUSTPhysicalEvidenceConfig(PhysicalEvidenceConfig):
    sampling_rate_hz: float = 51200.0
    fft_size: int = 2048
    spectrum_length: int = 512
    ball_sideband_orders: Tuple[int, ...] = (0, 1, 2)

    def validate(self) -> None:
        super().validate()
        for field_name in (
            "inner_sideband_orders",
            "outer_sideband_orders",
            "ball_sideband_orders",
        ):
            _validate_sideband_orders(getattr(self, field_name), field_name)


FAULT_COMPONENTS = {
    0: (),
    1: ("inner",),
    2: ("outer",),
    3: ("ball",),
    4: ("inner", "ball"),
    5: ("inner", "outer"),
    6: ("outer", "ball"),
}


def _validate_geometry(geometry: BearingGeometry) -> None:
    rolling_elements = int(geometry.rolling_elements)
    rolling_diameter = float(geometry.rolling_element_diameter_mm)
    pitch_diameter = float(geometry.pitch_diameter_mm)
    contact_angle = float(geometry.contact_angle_deg)
    if (
        rolling_elements <= 0
        or not math.isfinite(rolling_diameter)
        or not math.isfinite(pitch_diameter)
        or not math.isfinite(contact_angle)
        or rolling_diameter <= 0
        or pitch_diameter <= 0
        or rolling_diameter >= pitch_diameter
    ):
        raise ValueError("invalid bearing geometry")


def _validate_shaft_hz(shaft_hz: torch.Tensor) -> None:
    if not isinstance(shaft_hz, torch.Tensor):
        raise TypeError("shaft_hz must be a torch.Tensor")
    if not shaft_hz.is_floating_point():
        raise ValueError("shaft_hz must have a floating-point dtype")
    if not bool(torch.isfinite(shaft_hz).all()) or not bool((shaft_hz > 0).all()):
        raise ValueError("shaft_hz values must be finite and positive")


def characteristic_frequencies(
    shaft_hz: torch.Tensor,
    geometry: BearingGeometry,
) -> Dict[str, torch.Tensor]:
    """Compute exact bearing kinematic frequencies from shaft frequency."""
    _validate_geometry(geometry)
    _validate_shaft_hz(shaft_hz)

    rolling_elements = float(geometry.rolling_elements)
    rolling_diameter = float(geometry.rolling_element_diameter_mm)
    pitch_diameter = float(geometry.pitch_diameter_mm)
    angle_radians = math.radians(float(geometry.contact_angle_deg))
    ratio = (rolling_diameter / pitch_diameter) * math.cos(angle_radians)

    bpfo = 0.5 * rolling_elements * shaft_hz * (1.0 - ratio)
    bpfi = 0.5 * rolling_elements * shaft_hz * (1.0 + ratio)
    bsf = (
        (pitch_diameter / (2.0 * rolling_diameter))
        * shaft_hz
        * (1.0 - ratio * ratio)
    )
    ftf = 0.5 * shaft_hz * (1.0 - ratio)
    return {"shaft": shaft_hz, "bpfo": bpfo, "bpfi": bpfi, "bsf": bsf, "ftf": ftf}


def _validate_sideband_orders(
    orders: Sequence[int], field_name: str
) -> Tuple[int, ...]:
    try:
        values = tuple(orders)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be a non-empty sequence") from exc
    if not values:
        raise ValueError(f"{field_name} must be non-empty")

    validated = []
    for raw_order in values:
        if isinstance(raw_order, bool) or not isinstance(raw_order, Real):
            raise ValueError(
                f"{field_name} must contain finite non-negative integer orders"
            )
        numeric_order = float(raw_order)
        if (
            not math.isfinite(numeric_order)
            or numeric_order < 0
            or not numeric_order.is_integer()
        ):
            raise ValueError(
                f"{field_name} must contain finite non-negative integer orders"
            )
        validated.append(int(numeric_order))
    return tuple(validated)


def _sideband_multipliers(orders: Sequence[int]) -> Tuple[int, ...]:
    multipliers = set()
    for order in orders:
        if order == 0:
            multipliers.add(0)
        else:
            multipliers.update((-order, order))
    return tuple(sorted(multipliers))


def _coerce_target_domain(target_domain: int) -> int:
    if isinstance(target_domain, bool):
        raise ValueError("target_domain must be an integer scalar")
    if isinstance(target_domain, torch.Tensor):
        if (
            target_domain.dim() != 0
            or target_domain.dtype == torch.bool
            or target_domain.is_floating_point()
            or target_domain.is_complex()
        ):
            raise ValueError("target_domain must be an integer scalar")
    try:
        domain = operator.index(target_domain)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("target_domain must be an integer scalar") from exc
    if domain not in HUST_GEOMETRIES:
        raise ValueError(f"unsupported HUST target domain: {domain}")
    return domain


def _component_mask(
    base_hz: torch.Tensor,
    sideband_hz: torch.Tensor,
    sideband_orders: Sequence[int],
    config: HUSTPhysicalEvidenceConfig,
) -> torch.Tensor:
    """Build uncapped Gaussian masks for one fault component, shape ``[B,L]``."""
    device = base_hz.device
    dtype = base_hz.dtype
    harmonics = torch.arange(
        1, config.harmonics + 1, device=device, dtype=dtype
    ).view(1, -1, 1)
    offsets = torch.tensor(
        _sideband_multipliers(sideband_orders), device=device, dtype=dtype
    ).view(1, 1, -1)
    centers_hz = (
        base_hz.view(-1, 1, 1) * harmonics
        + sideband_hz.view(-1, 1, 1) * offsets
    ).flatten(1)

    resolution_hz = config.sampling_rate_hz / float(config.fft_size)
    max_frequency_hz = (config.spectrum_length - 1) * resolution_hz
    valid = (centers_hz > 0) & (centers_hz <= max_frequency_hz)
    bin_axis = torch.arange(config.spectrum_length, device=device, dtype=dtype)
    center_bins = centers_hz / resolution_hz
    distance = bin_axis.view(1, 1, -1) - center_bins.unsqueeze(-1)
    gaussian = torch.exp(
        -0.5 * (distance / float(config.mask_sigma_bins)).pow(2)
    )
    gaussian = gaussian * valid.unsqueeze(-1).to(dtype=dtype)
    mask = gaussian.amax(dim=1).clamp_(0.0, 1.0)
    if config.exclude_dc:
        mask[:, 0] = 0.0
    return mask


def _cap_active_bins(
    masks: torch.Tensor, config: HUSTPhysicalEvidenceConfig
) -> torch.Tensor:
    """Apply the common active-bin budget independently to every mask row."""
    capped = masks.clone()
    max_active = int(math.floor(config.max_mask_ratio * config.spectrum_length))
    for row in range(capped.size(0)):
        active_count = int(
            (capped[row] >= float(config.mask_activity_threshold)).sum().item()
        )
        if active_count <= max_active:
            continue
        if max_active == 0:
            capped[row].zero_()
            continue
        keep = torch.topk(capped[row], k=max_active, largest=True).indices
        clipped = torch.zeros_like(capped[row])
        clipped[keep] = capped[row, keep]
        capped[row] = clipped
    return capped


def _validate_labels(pseudo_labels: torch.Tensor) -> torch.Tensor:
    if not isinstance(pseudo_labels, torch.Tensor):
        raise TypeError("pseudo_labels must be a torch.Tensor")
    if pseudo_labels.dim() != 1:
        raise ValueError("pseudo_labels must be 1-D")
    if pseudo_labels.is_floating_point():
        if not bool(torch.isfinite(pseudo_labels).all()) or not bool(
            pseudo_labels.eq(pseudo_labels.round()).all()
        ):
            raise ValueError("pseudo_labels must contain integer label values")
    labels = pseudo_labels.to(dtype=torch.long)
    if labels.numel() and not bool(((labels >= 0) & (labels <= 6)).all()):
        raise ValueError("unsupported HUST predicted label")
    return labels


def build_hust_physical_masks(
    pseudo_labels: torch.Tensor,
    target_domain: int,
    shaft_hz: torch.Tensor,
    config: HUSTPhysicalEvidenceConfig = HUSTPhysicalEvidenceConfig(),
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return predicted-class-selected masks and applicability for HUST samples."""
    config.validate()
    labels = _validate_labels(pseudo_labels)
    domain = _coerce_target_domain(target_domain)
    if not isinstance(shaft_hz, torch.Tensor) or shaft_hz.dim() != 1:
        raise ValueError("shaft_hz must be 1-D")
    if shaft_hz.numel() != labels.numel():
        raise ValueError("pseudo_labels and shaft_hz must have the same length")
    _validate_shaft_hz(shaft_hz)

    device = labels.device
    shaft = shaft_hz.to(device=device, dtype=torch.float32)
    frequencies = characteristic_frequencies(shaft, HUST_GEOMETRIES[domain])
    component_masks = {
        "inner": _component_mask(
            frequencies["bpfi"],
            frequencies["shaft"],
            config.inner_sideband_orders,
            config,
        ),
        "outer": _component_mask(
            frequencies["bpfo"],
            frequencies["shaft"],
            config.outer_sideband_orders,
            config,
        ),
        "ball": _component_mask(
            frequencies["bsf"],
            frequencies["ftf"],
            config.ball_sideband_orders,
            config,
        ),
    }

    masks = torch.zeros(
        labels.numel(), config.spectrum_length, device=device, dtype=torch.float32
    )
    for label, components in FAULT_COMPONENTS.items():
        selected = labels.eq(label)
        if not components or not bool(selected.any()):
            continue
        union = component_masks[components[0]]
        for component in components[1:]:
            union = torch.maximum(union, component_masks[component])
        masks[selected] = union[selected]

    masks = _cap_active_bins(masks, config)
    applicable = labels.ne(0)
    return masks, applicable


__all__ = [
    "BearingGeometry",
    "FAULT_COMPONENTS",
    "HUST_GEOMETRIES",
    "HUSTPhysicalEvidenceConfig",
    "build_hust_physical_masks",
    "characteristic_frequencies",
    "class_margin",
    "construct_physical_destructive_view",
    "mask_active_ratio",
    "robust_fault_evidence_score",
]
