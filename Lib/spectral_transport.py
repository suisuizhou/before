#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""Unified forward/inverse affine-monotonic spectral transport.

A transport map ``phi`` maps source frequency coordinates ``f`` to shifted
coordinates ``u = phi(f)``.  The forward transported spectrum is

    y(u) = x(phi^{-1}(u)),

while target-domain inverse correction is

    x_corr(f) = y(phi(f)).

Both operations therefore use the same monotonic map family.  Source
augmentation explicitly samples the inverse coordinate map; target TTA learns
``phi`` and samples the observed target spectrum at ``phi(f)``.
"""

from __future__ import annotations

from typing import Dict, Tuple

import math
import torch
import torch.nn.functional as F


def _validate_ctrl(warp_ctrl: torch.Tensor, length: int) -> None:
    if warp_ctrl.ndim != 3 or warp_ctrl.shape[1] != 1:
        raise ValueError(
            "warp_ctrl must have shape [N, 1, K], "
            f"got {tuple(warp_ctrl.shape)}"
        )
    if warp_ctrl.shape[-1] < 2:
        raise ValueError("warp_ctrl must contain at least two segments")
    if int(length) < 2:
        raise ValueError(f"length must be >= 2, got {length}")


def _expand_scalar_parameter(param: torch.Tensor, batch: int, name: str) -> torch.Tensor:
    """Return a [N,1,1] view from a scalar or one value per map."""
    if param.numel() == 1:
        return param.reshape(1, 1, 1).expand(batch, 1, 1)
    if param.numel() == batch:
        return param.reshape(batch, 1, 1)
    raise ValueError(
        f"{name} must contain one scalar or {batch} values, got {param.numel()}"
    )


def _build_local_monotonic_positions(
    warp_ctrl: torch.Tensor,
    length: int,
    residual_max_warp: float,
    log_slope_limit: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Create endpoint-anchored, strictly increasing local maps [N,1,L]."""
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
    raw_knots = torch.cat(
        [torch.zeros_like(cumulative[..., :1]), cumulative], dim=-1
    )
    raw_knots = raw_knots / raw_knots[..., -1:].clamp_min(eps)
    raw_knots = raw_knots * float(length - 1)

    identity_knots = torch.linspace(
        0.0,
        float(length - 1),
        steps=raw_knots.shape[-1],
        dtype=dtype,
        device=device,
    ).view(1, 1, -1)

    # Convexly blend the raw monotonic map with identity until the maximum
    # local displacement obeys residual_max_warp. Convex combinations of two
    # strictly increasing maps remain strictly increasing.
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
    identity_expanded = identity.expand(warp_ctrl.shape[0], -1, -1)
    local_positions = identity_expanded + delta

    # Preserve analytically anchored endpoints exactly.
    local_positions = torch.cat(
        [
            identity_expanded[..., :1],
            local_positions[..., 1:-1],
            identity_expanded[..., -1:],
        ],
        dim=-1,
    )
    return local_positions


def bounded_affine_parameters(
    raw_scale: torch.Tensor,
    raw_shift: torch.Tensor,
    max_scale: float,
    max_shift: float,
    batch: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert raw values to positive scales and bounded shifts."""
    if float(max_scale) < 1.0:
        raise ValueError(f"max_scale must be >= 1, got {max_scale}")
    if float(max_shift) < 0.0:
        raise ValueError(f"max_shift must be non-negative, got {max_shift}")

    if batch is None:
        batch = max(raw_scale.numel(), raw_shift.numel())
    scale_raw = _expand_scalar_parameter(raw_scale, batch, "raw_scale")
    shift_raw = _expand_scalar_parameter(raw_shift, batch, "raw_shift")

    if float(max_scale) == 1.0:
        scale = torch.ones_like(scale_raw)
    else:
        scale = torch.exp(
            math.log(float(max_scale)) * torch.tanh(scale_raw)
        )
    shift = float(max_shift) * torch.tanh(shift_raw)
    return scale, shift


def build_affine_monotonic_positions(
    warp_ctrl: torch.Tensor,
    raw_scale: torch.Tensor,
    raw_shift: torch.Tensor,
    length: int,
    residual_max_warp: float = 1.0,
    max_scale: float = 1.01,
    max_shift: float = 0.5,
    log_slope_limit: float = 2.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Build learnable affine-monotonic transport maps [N,1,L]."""
    local_positions = _build_local_monotonic_positions(
        warp_ctrl=warp_ctrl,
        length=length,
        residual_max_warp=residual_max_warp,
        log_slope_limit=log_slope_limit,
    )
    batch = local_positions.shape[0]
    scale, shift = bounded_affine_parameters(
        raw_scale=raw_scale,
        raw_shift=raw_shift,
        max_scale=max_scale,
        max_shift=max_shift,
        batch=batch,
    )

    center = local_positions.new_tensor((int(length) - 1) / 2.0)
    positions = center + scale * (local_positions - center) + shift
    return positions, {
        "local_positions": local_positions,
        "scale": scale,
        "shift": shift,
    }


def build_random_affine_monotonic_positions(
    batch: int,
    length: int,
    knots: int,
    residual_max_warp: float,
    max_scale: float,
    max_shift: float,
    log_slope_limit: float,
    ctrl_std: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Sample one physically valid forward transport map per source sample."""
    if int(batch) < 1:
        raise ValueError("batch must be positive")
    if int(knots) < 2:
        raise ValueError("knots must be >= 2")
    if float(ctrl_std) < 0.0:
        raise ValueError("ctrl_std must be non-negative")

    ctrl = torch.randn(
        int(batch), 1, int(knots), device=device, dtype=dtype
    ) * float(ctrl_std)
    local_positions = _build_local_monotonic_positions(
        warp_ctrl=ctrl,
        length=length,
        residual_max_warp=residual_max_warp,
        log_slope_limit=log_slope_limit,
    )

    if float(max_scale) == 1.0:
        scale = torch.ones(int(batch), 1, 1, device=device, dtype=dtype)
    else:
        log_limit = math.log(float(max_scale))
        scale = torch.exp(
            torch.empty(int(batch), 1, 1, device=device, dtype=dtype).uniform_(
                -log_limit, log_limit
            )
        )
    shift = torch.empty(
        int(batch), 1, 1, device=device, dtype=dtype
    ).uniform_(-float(max_shift), float(max_shift))

    center = local_positions.new_tensor((int(length) - 1) / 2.0)
    positions = center + scale * (local_positions - center) + shift
    return positions, {
        "local_positions": local_positions,
        "scale": scale,
        "shift": shift,
        "ctrl": ctrl,
    }


def invert_monotonic_positions(positions: torch.Tensor) -> torch.Tensor:
    """Numerically invert strictly increasing maps with linear extrapolation."""
    if positions.ndim != 3 or positions.shape[1] != 1:
        raise ValueError(
            "positions must have shape [N,1,L], "
            f"got {tuple(positions.shape)}"
        )
    batch, _, length = positions.shape
    if length < 2:
        raise ValueError("positions length must be >= 2")

    p = positions.squeeze(1)
    slopes = p[:, 1:] - p[:, :-1]
    if torch.any(slopes <= 0):
        raise ValueError("positions must be strictly increasing")

    output_coords = torch.arange(
        length, device=p.device, dtype=p.dtype
    ).view(1, length).expand(batch, length)

    # searchsorted is piecewise constant in the interval index, but the
    # interpolation itself is differentiable. Random source maps do not need
    # gradients; gradients to the source signal are fully preserved.
    right = torch.searchsorted(p.contiguous(), output_coords.contiguous(), right=True)
    right = right.clamp(1, length - 1)
    left = right - 1

    p_left = torch.gather(p, 1, left)
    p_right = torch.gather(p, 1, right)
    frac = (output_coords - p_left) / (p_right - p_left).clamp_min(1e-8)
    inverse = left.to(dtype=p.dtype) + frac
    return inverse.unsqueeze(1)


def sample_1d_linear_extrapolation(
    x_cf: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Differentiably sample [B,C,L] at shared or per-sample positions."""
    if x_cf.ndim != 3:
        raise ValueError(f"x_cf must have shape [B,C,L], got {tuple(x_cf.shape)}")
    if positions.ndim != 3 or positions.shape[1] != 1:
        raise ValueError(
            "positions must have shape [N,1,L], "
            f"got {tuple(positions.shape)}"
        )

    batch, channels, length = x_cf.shape
    if positions.shape[-1] != length:
        raise ValueError(
            f"positions length {positions.shape[-1]} != input length {length}"
        )
    if positions.shape[0] not in (1, batch):
        raise ValueError(
            f"positions batch must be 1 or {batch}, got {positions.shape[0]}"
        )
    if length < 2:
        raise ValueError("linear extrapolation requires length >= 2")

    pos = positions.to(device=x_cf.device, dtype=x_cf.dtype)
    if pos.shape[0] == 1 and batch > 1:
        pos = pos.expand(batch, -1, -1)

    pos_inside = pos.clamp(0.0, float(length - 1))
    left = torch.floor(pos_inside).long().clamp(max=length - 2)
    right = left + 1
    frac = pos_inside - left.to(dtype=pos_inside.dtype)

    left_idx = left.expand(batch, channels, length)
    right_idx = right.expand(batch, channels, length)
    x_left = torch.gather(x_cf, dim=2, index=left_idx)
    x_right = torch.gather(x_cf, dim=2, index=right_idx)
    y_inside = x_left + frac.expand(batch, channels, length) * (x_right - x_left)

    pos_expanded = pos.expand(batch, channels, length)
    x0 = x_cf[..., :1]
    x1 = x_cf[..., 1:2]
    low = x0 + pos_expanded * (x1 - x0)

    x_last = x_cf[..., -1:]
    x_prev = x_cf[..., -2:-1]
    high = x_last + (
        pos_expanded - float(length - 1)
    ) * (x_last - x_prev)

    return torch.where(
        pos_expanded < 0.0,
        low,
        torch.where(pos_expanded > float(length - 1), high, y_inside),
    )


def _to_channel_first_1d(x: torch.Tensor) -> Tuple[torch.Tensor, bool, bool]:
    squeezed = False
    transposed = False
    if x.ndim == 2:
        x = x.unsqueeze(1)
        squeezed = True
    elif x.ndim == 3 and x.shape[-1] == 1:
        x = x.transpose(1, 2)
        transposed = True
    elif x.ndim != 3:
        raise ValueError(f"Unexpected x shape: {tuple(x.shape)}")
    return x, squeezed, transposed


def _restore_1d_shape(
    x: torch.Tensor, squeezed: bool, transposed: bool
) -> torch.Tensor:
    if transposed:
        x = x.transpose(1, 2)
    if squeezed:
        x = x.squeeze(1)
    return x


def random_forward_transport_augment(
    x: torch.Tensor,
    prob: float = 0.7,
    knots: int = 16,
    residual_max_warp: float = 1.0,
    max_scale: float = 1.01,
    max_shift: float = 0.5,
    log_slope_limit: float = 2.0,
    ctrl_std: float = 0.5,
) -> torch.Tensor:
    """Apply source-side forward transport y(u)=x(phi^{-1}(u))."""
    if not 0.0 <= float(prob) <= 1.0:
        raise ValueError("prob must be in [0,1]")
    if torch.rand((), device=x.device).item() > float(prob):
        return x

    x_cf, squeezed, transposed = _to_channel_first_1d(x)
    batch, _, length = x_cf.shape
    forward_positions, _ = build_random_affine_monotonic_positions(
        batch=batch,
        length=length,
        knots=knots,
        residual_max_warp=residual_max_warp,
        max_scale=max_scale,
        max_shift=max_shift,
        log_slope_limit=log_slope_limit,
        ctrl_std=ctrl_std,
        device=x_cf.device,
        dtype=x_cf.dtype,
    )
    inverse_positions = invert_monotonic_positions(forward_positions)
    transported = sample_1d_linear_extrapolation(x_cf, inverse_positions)
    return _restore_1d_shape(transported, squeezed, transposed)


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
    """Regularize target transport departure from identity."""
    del positions
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
        ).square().mean()

    if float(max_shift) == 0.0:
        shift_loss = local_positions.new_zeros(())
    else:
        shift_loss = (shift / float(max_shift)).square().mean()

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
    """Diagnostics for physical validity and optimization drift."""
    identity = torch.arange(
        int(length), dtype=positions.dtype, device=positions.device
    ).view(1, 1, -1)
    delta = positions - identity
    local_delta = local_positions - identity
    slopes = positions[..., 1:] - positions[..., :-1]
    oob = (positions < 0.0) | (positions > float(int(length) - 1))

    return {
        "delta_max": float(delta.abs().max().detach().item()),
        "local_delta_max": float(local_delta.abs().max().detach().item()),
        "scale": float(scale.mean().detach().item()),
        "shift": float(shift.mean().detach().item()),
        "min_slope": float(slopes.min().detach().item()),
        "max_slope": float(slopes.max().detach().item()),
        "violations": int((slopes <= 0.0).sum().detach().item()),
        "oob_fraction": float(oob.float().mean().detach().item()),
        "left_oob": float(torch.relu(-positions.min()).detach().item()),
        "right_oob": float(
            torch.relu(positions.max() - float(int(length) - 1)).detach().item()
        ),
    }
