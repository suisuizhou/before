#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""Strictly monotonic, endpoint-anchored 1-D frequency warp utilities.

The learnable tensor ``warp_ctrl`` parameterizes positive segment lengths.
Their cumulative sum defines a strictly increasing sampling map.  The map is
then blended with the identity map so that its maximum displacement is bounded
by ``max_warp`` frequency bins.  Consequently the map preserves frequency
order and is invertible on the sampled interval.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def _validate_inputs(warp_ctrl: torch.Tensor, length: int, max_warp: float) -> None:
    if warp_ctrl.ndim != 3 or warp_ctrl.shape[0] != 1 or warp_ctrl.shape[1] != 1:
        raise ValueError(
            "warp_ctrl must have shape [1, 1, K], "
            f"got {tuple(warp_ctrl.shape)}"
        )
    if warp_ctrl.shape[-1] < 2:
        raise ValueError("warp_ctrl must contain at least two monotonic segments")
    if int(length) < 2:
        raise ValueError(f"length must be >= 2, got {length}")
    if float(max_warp) < 0.0:
        raise ValueError(f"max_warp must be non-negative, got {max_warp}")


def build_monotonic_positions(
    warp_ctrl: torch.Tensor,
    length: int,
    max_warp: float,
    log_slope_limit: float = 2.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Build a bounded, strictly increasing sampling map.

    Args:
        warp_ctrl: Learnable segment logits with shape ``[1, 1, K]``.
        length: Number of frequency bins in the full-resolution spectrum.
        max_warp: Maximum absolute displacement in frequency bins.
        log_slope_limit: Bounds each unnormalised segment multiplier to
            ``[exp(-limit), exp(limit)]`` through ``exp(limit*tanh(ctrl))``.
        eps: Numerical stabiliser.

    Returns:
        Sampling positions with shape ``[1, 1, length]``.  The first and last
        positions are fixed at 0 and ``length-1``; every adjacent difference is
        positive; and the maximum displacement from identity is at most
        ``max_warp``.
    """
    _validate_inputs(warp_ctrl, length, max_warp)

    length = int(length)
    max_warp = float(max_warp)
    dtype = warp_ctrl.dtype
    device = warp_ctrl.device

    # Positive segment lengths guarantee an order-preserving control polygon.
    segment_lengths = torch.exp(float(log_slope_limit) * torch.tanh(warp_ctrl))
    cumulative = torch.cumsum(segment_lengths, dim=-1)
    zero = torch.zeros_like(cumulative[..., :1])
    raw_knots = torch.cat([zero, cumulative], dim=-1)
    raw_knots = raw_knots / raw_knots[..., -1:].clamp_min(eps)
    raw_knots = raw_knots * float(length - 1)

    identity_knots = torch.linspace(
        0.0,
        float(length - 1),
        steps=raw_knots.shape[-1],
        device=device,
        dtype=dtype,
    ).view(1, 1, -1)

    raw_delta = raw_knots - identity_knots
    raw_max = raw_delta.abs().amax(dim=-1, keepdim=True)

    if max_warp == 0.0:
        blend = torch.zeros_like(raw_max)
    else:
        blend = torch.clamp(max_warp / raw_max.clamp_min(eps), max=1.0)

    # A convex combination of two strictly increasing maps remains strictly
    # increasing.  It also makes the displacement bound exact by construction.
    bounded_delta_knots = blend * raw_delta
    delta = F.interpolate(
        bounded_delta_knots,
        size=length,
        mode="linear",
        align_corners=True,
    )
    identity = torch.arange(length, device=device, dtype=dtype).view(1, 1, length)
    positions = identity + delta

    # Both endpoint displacements are analytically zero.  Re-anchor them to
    # remove any interpolation round-off while preserving the interior graph.
    positions = torch.cat(
        [
            identity[..., :1],
            positions[..., 1:-1],
            identity[..., -1:],
        ],
        dim=-1,
    )
    return positions


def sample_1d_with_positions(
    x_cf: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Resample a channel-first 1-D tensor with differentiable interpolation."""
    if x_cf.ndim != 3:
        raise ValueError(f"x_cf must have shape [B, C, L], got {tuple(x_cf.shape)}")
    if positions.ndim != 3 or positions.shape[0:2] != (1, 1):
        raise ValueError(
            "positions must have shape [1, 1, L], "
            f"got {tuple(positions.shape)}"
        )

    batch, channels, length = x_cf.shape
    if positions.shape[-1] != length:
        raise ValueError(
            f"positions length {positions.shape[-1]} != input length {length}"
        )

    positions = positions.to(device=x_cf.device, dtype=x_cf.dtype)
    positions = positions.clamp(0.0, float(length - 1))
    left = torch.floor(positions).long()
    right = (left + 1).clamp(max=length - 1)
    weight_right = positions - left.to(dtype=positions.dtype)
    weight_left = 1.0 - weight_right

    left_idx = left.expand(batch, channels, length)
    right_idx = right.expand(batch, channels, length)
    x_left = torch.gather(x_cf, dim=2, index=left_idx)
    x_right = torch.gather(x_cf, dim=2, index=right_idx)

    return (
        weight_left.expand(batch, channels, length) * x_left
        + weight_right.expand(batch, channels, length) * x_right
    )


def monotonic_warp_regularizer(
    positions: torch.Tensor,
    max_warp: float,
    smooth_weight: float = 2.0,
    slope_weight: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Regularize identity displacement, curvature and local slope distortion."""
    if positions.ndim != 3 or positions.shape[0:2] != (1, 1):
        raise ValueError(
            "positions must have shape [1, 1, L], "
            f"got {tuple(positions.shape)}"
        )

    length = positions.shape[-1]
    identity = torch.arange(
        length,
        device=positions.device,
        dtype=positions.dtype,
    ).view(1, 1, length)
    delta = positions - identity

    displacement_denom = max(float(max_warp) ** 2, 1e-6)
    identity_loss = delta.square().mean() / displacement_denom

    if length >= 3:
        curvature = delta[..., 2:] - 2.0 * delta[..., 1:-1] + delta[..., :-2]
        smooth_loss = curvature.square().mean() / displacement_denom
    else:
        smooth_loss = positions.new_zeros(())

    slopes = positions[..., 1:] - positions[..., :-1]
    slope_loss = (slopes - 1.0).square().mean()

    total = (
        identity_loss
        + float(smooth_weight) * smooth_loss
        + float(slope_weight) * slope_loss
    )
    return total, {
        "identity": identity_loss,
        "smooth": smooth_loss,
        "slope": slope_loss,
    }


def monotonic_warp_stats(positions: torch.Tensor) -> Dict[str, float]:
    """Return interpretable diagnostics for a monotonic sampling map."""
    length = positions.shape[-1]
    identity = torch.arange(
        length,
        device=positions.device,
        dtype=positions.dtype,
    ).view(1, 1, length)
    delta = positions - identity
    slopes = positions[..., 1:] - positions[..., :-1]

    return {
        "delta_max": float(delta.abs().max().detach().item()),
        "min_slope": float(slopes.min().detach().item()),
        "max_slope": float(slopes.max().detach().item()),
        "violations": int((slopes <= 0.0).sum().detach().item()),
    }
