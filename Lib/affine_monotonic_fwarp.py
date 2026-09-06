#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""Affine-monotonic 1-D frequency transport.

The sampling map is

    phi(f) = center + scale * (psi(f) - center) + shift,

where ``psi`` is an endpoint-anchored strictly increasing local map,
``scale`` is positive and bounded, and ``shift`` is bounded. Therefore the
complete map remains strictly increasing. Values sampled just outside the
observed frequency interval use differentiable linear extrapolation instead
of edge clamping, avoiding constant boundary plateaus.
"""

from __future__ import annotations

from typing import Dict, Tuple

import math
import torch
import torch.nn.functional as F


def _validate_ctrl(warp_ctrl: torch.Tensor, length: int) -> None:
    if warp_ctrl.ndim != 3 or tuple(warp_ctrl.shape[:2]) != (1, 1):
        raise ValueError(
            "warp_ctrl must have shape [1, 1, K], "
            f"got {tuple(warp_ctrl.shape)}"
        )
    if warp_ctrl.shape[-1] < 2:
        raise ValueError("warp_ctrl must contain at least two segments")
    if int(length) < 2:
        raise ValueError(f"length must be >= 2, got {length}")


def _build_local_monotonic_positions(
    warp_ctrl: torch.Tensor,
    length: int,
    residual_max_warp: float,
    log_slope_limit: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Create an endpoint-anchored strictly increasing local map."""
    _validate_ctrl(warp_ctrl, length)
    if float(residual_max_warp) < 0.0:
        raise ValueError("residual_max_warp must be non-negative")
    if float(log_slope_limit) < 0.0:
        raise ValueError("log_slope_limit must be non-negative")

    length = int(length)
    dtype = warp_ctrl.dtype
    device = warp_ctrl.device

    segment_lengths = torch.exp(
        float(log_slope_limit) * torch.tanh(warp_ctrl)
    )
    cumulative = torch.cumsum(segment_lengths, dim=-1)
    raw_knots = torch.cat([torch.zeros_like(cumulative[..., :1]), cumulative], dim=-1)
    raw_knots = raw_knots / raw_knots[..., -1:].clamp_min(eps)
    raw_knots = raw_knots * float(length - 1)

    identity_knots = torch.linspace(
        0.0,
        float(length - 1),
        steps=raw_knots.shape[-1],
        dtype=dtype,
        device=device,
    ).view(1, 1, -1)

    raw_delta = raw_knots - identity_knots
    raw_max = raw_delta.abs().amax(dim=-1, keepdim=True)
    if float(residual_max_warp) == 0.0:
        blend = torch.zeros_like(raw_max)
    else:
        blend = torch.clamp(
            float(residual_max_warp) / raw_max.clamp_min(eps),
            max=1.0,
        )

    delta_knots = blend * raw_delta
    delta = F.interpolate(
        delta_knots,
        size=length,
        mode="linear",
        align_corners=True,
    )
    identity = torch.arange(length, dtype=dtype, device=device).view(1, 1, -1)
    local_positions = identity + delta

    # Remove interpolation round-off at the analytically anchored endpoints.
    local_positions = torch.cat(
        [identity[..., :1], local_positions[..., 1:-1], identity[..., -1:]],
        dim=-1,
    )
    return local_positions


def bounded_affine_parameters(
    raw_scale: torch.Tensor,
    raw_shift: torch.Tensor,
    max_scale: float,
    max_shift: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert unconstrained scalars to a positive scale and bounded shift."""
    if raw_scale.numel() != 1 or raw_shift.numel() != 1:
        raise ValueError("raw_scale and raw_shift must each contain one scalar")
    if float(max_scale) < 1.0:
        raise ValueError(f"max_scale must be >= 1, got {max_scale}")
    if float(max_shift) < 0.0:
        raise ValueError(f"max_shift must be non-negative, got {max_shift}")

    if float(max_scale) == 1.0:
        scale = torch.ones_like(raw_scale)
    else:
        # Symmetric in log-space: [1/max_scale, max_scale].
        scale = torch.exp(math.log(float(max_scale)) * torch.tanh(raw_scale))
    shift = float(max_shift) * torch.tanh(raw_shift)
    return scale.reshape(()), shift.reshape(())


def build_affine_monotonic_positions(
    warp_ctrl: torch.Tensor,
    raw_scale: torch.Tensor,
    raw_shift: torch.Tensor,
    length: int,
    residual_max_warp: float = 1.0,
    max_scale: float = 1.05,
    max_shift: float = 2.0,
    log_slope_limit: float = 2.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Build the complete strictly increasing affine-monotonic map."""
    local_positions = _build_local_monotonic_positions(
        warp_ctrl=warp_ctrl,
        length=length,
        residual_max_warp=residual_max_warp,
        log_slope_limit=log_slope_limit,
    )
    scale, shift = bounded_affine_parameters(
        raw_scale=raw_scale,
        raw_shift=raw_shift,
        max_scale=max_scale,
        max_shift=max_shift,
    )

    center = local_positions.new_tensor((int(length) - 1) / 2.0)
    positions = center + scale * (local_positions - center) + shift
    return positions, {
        "local_positions": local_positions,
        "scale": scale,
        "shift": shift,
    }


def sample_1d_linear_extrapolation(
    x_cf: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Differentiably sample [B,C,L] with linear edge extrapolation."""
    if x_cf.ndim != 3:
        raise ValueError(f"x_cf must have shape [B,C,L], got {tuple(x_cf.shape)}")
    if positions.ndim != 3 or tuple(positions.shape[:2]) != (1, 1):
        raise ValueError(
            "positions must have shape [1,1,L], "
            f"got {tuple(positions.shape)}"
        )

    batch, channels, length = x_cf.shape
    if positions.shape[-1] != length:
        raise ValueError(
            f"positions length {positions.shape[-1]} != input length {length}"
        )
    if length < 2:
        raise ValueError("linear extrapolation requires length >= 2")

    pos = positions.to(device=x_cf.device, dtype=x_cf.dtype)
    pos_inside = pos.clamp(0.0, float(length - 1))
    left = torch.floor(pos_inside).long().clamp(max=length - 2)
    right = left + 1
    frac = pos_inside - left.to(dtype=pos_inside.dtype)

    left_idx = left.expand(batch, channels, length)
    right_idx = right.expand(batch, channels, length)
    x_left = torch.gather(x_cf, dim=2, index=left_idx)
    x_right = torch.gather(x_cf, dim=2, index=right_idx)
    y_inside = x_left + frac.expand(batch, channels, length) * (x_right - x_left)

    x0 = x_cf[..., :1]
    x1 = x_cf[..., 1:2]
    low = x0 + pos.expand(batch, channels, length) * (x1 - x0)

    x_last = x_cf[..., -1:]
    x_prev = x_cf[..., -2:-1]
    high = x_last + (
        pos.expand(batch, channels, length) - float(length - 1)
    ) * (x_last - x_prev)

    pos_expanded = pos.expand(batch, channels, length)
    return torch.where(
        pos_expanded < 0.0,
        low,
        torch.where(pos_expanded > float(length - 1), high, y_inside),
    )


def affine_monotonic_regularizer(
    positions: torch.Tensor,
    local_positions: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    residual_max_warp: float,
    max_scale: float,
    max_shift: float,
    residual_identity_weight: float = 1.0,
    residual_smooth_weight: float = 2.0,
    residual_slope_weight: float = 0.1,
    scale_weight: float = 1.0,
    shift_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Regularize local distortion and global affine departure from identity."""
    del positions  # retained in the public signature for explicit call sites

    length = local_positions.shape[-1]
    identity = torch.arange(
        length,
        dtype=local_positions.dtype,
        device=local_positions.device,
    ).view(1, 1, -1)
    local_delta = local_positions - identity

    residual_denom = max(float(residual_max_warp) ** 2, 1e-6)
    residual_identity = local_delta.square().mean() / residual_denom

    if length >= 3:
        curvature = (
            local_delta[..., 2:]
            - 2.0 * local_delta[..., 1:-1]
            + local_delta[..., :-2]
        )
        residual_smooth = curvature.square().mean() / residual_denom
    else:
        residual_smooth = local_positions.new_zeros(())

    local_slopes = local_positions[..., 1:] - local_positions[..., :-1]
    residual_slope = (local_slopes - 1.0).square().mean()

    if float(max_scale) == 1.0:
        scale_loss = local_positions.new_zeros(())
    else:
        scale_loss = (
            torch.log(scale) / math.log(float(max_scale))
        ).square()

    if float(max_shift) == 0.0:
        shift_loss = local_positions.new_zeros(())
    else:
        shift_loss = (shift / float(max_shift)).square()

    total = (
        float(residual_identity_weight) * residual_identity
        + float(residual_smooth_weight) * residual_smooth
        + float(residual_slope_weight) * residual_slope
        + float(scale_weight) * scale_loss
        + float(shift_weight) * shift_loss
    )
    return total, {
        "residual_identity": residual_identity,
        "residual_smooth": residual_smooth,
        "residual_slope": residual_slope,
        "scale": scale_loss,
        "shift": shift_loss,
    }


def affine_monotonic_stats(
    positions: torch.Tensor,
    local_positions: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    length: int,
) -> Dict[str, float]:
    """Return diagnostics for physical validity and optimization drift."""
    identity = torch.arange(
        int(length),
        dtype=positions.dtype,
        device=positions.device,
    ).view(1, 1, -1)
    delta = positions - identity
    local_delta = local_positions - identity
    slopes = positions[..., 1:] - positions[..., :-1]
    oob = (positions < 0.0) | (positions > float(int(length) - 1))

    return {
        "delta_max": float(delta.abs().max().detach().item()),
        "local_delta_max": float(local_delta.abs().max().detach().item()),
        "scale": float(scale.detach().item()),
        "shift": float(shift.detach().item()),
        "min_slope": float(slopes.min().detach().item()),
        "max_slope": float(slopes.max().detach().item()),
        "violations": int((slopes <= 0.0).sum().detach().item()),
        "oob_fraction": float(oob.float().mean().detach().item()),
        "left_oob": float(torch.relu(-positions.min()).detach().item()),
        "right_oob": float(
            torch.relu(positions.max() - float(int(length) - 1)).detach().item()
        ),
    }
