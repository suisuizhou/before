#!/usr/bin/python
# -*- coding: UTF-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralCalibrator1D(nn.Module):
    """
    SCSC: Self-Calibrating Spectral Canonicalization.

    For each input sample, predict:
      1) global frequency scale
      2) global frequency shift
      3) low-dimensional residual warp
      4) coarse band gain

    Then canonicalize the input spectrum before ViT.
    """

    def __init__(
        self,
        input_len=512,
        summary_bands=32,
        warp_knots=8,
        gain_bands=16,
        hidden_dim=64,
        scale_range=0.10,
        max_shift=2.0,
        residual_max=1.0,
        gain_delta=0.05,
    ):
        super().__init__()

        self.input_len = int(input_len)
        self.summary_bands = int(summary_bands)
        self.warp_knots = int(warp_knots)
        self.gain_bands = int(gain_bands)
        self.hidden_dim = int(hidden_dim)

        self.scale_range = float(scale_range)
        self.max_shift = float(max_shift)
        self.residual_max = float(residual_max)
        self.gain_delta = float(gain_delta)

        in_dim = self.summary_bands + 3
        out_dim = 2 + self.warp_knots + self.gain_bands

        self.net = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, out_dim),
        )

        # Important: identity transform at initialization.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        self._last_reg = None
        self._last_info = {}

    def _to_channel_first_1d(self, x):
        squeezed = False
        transposed = False

        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeezed = True
        elif x.dim() == 3 and x.shape[-1] == 1:
            x = x.transpose(1, 2)
            transposed = True
        elif x.dim() != 3:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        return x, squeezed, transposed

    def _restore_1d_shape(self, x, squeezed, transposed):
        if transposed:
            x = x.transpose(1, 2)
        if squeezed:
            x = x.squeeze(1)
        return x

    def _summary(self, x_cf):
        # x_cf: [B, C, L]
        B, C, L = x_cf.shape

        energy = x_cf.abs().mean(dim=1).pow(2)
        energy = energy / energy.sum(dim=1, keepdim=True).clamp_min(1e-8)

        bands = F.adaptive_avg_pool1d(
            energy.unsqueeze(1),
            self.summary_bands
        ).squeeze(1)

        bands = bands / bands.sum(dim=1, keepdim=True).clamp_min(1e-8)

        entropy = -(bands * torch.log(bands + 1e-8)).sum(dim=1, keepdim=True)
        entropy = entropy / math.log(max(self.summary_bands, 2))

        freq = torch.linspace(
            0.0, 1.0,
            steps=self.summary_bands,
            device=x_cf.device,
            dtype=x_cf.dtype
        ).view(1, -1)

        centroid = (bands * freq).sum(dim=1, keepdim=True)
        spread = torch.sqrt(
            (bands * (freq - centroid).pow(2)).sum(dim=1, keepdim=True).clamp_min(1e-8)
        )

        return torch.cat([bands, entropy, centroid, spread], dim=1)

    def forward(self, x):
        x_cf, squeezed, transposed = self._to_channel_first_1d(x)

        if x_cf.shape[-1] != self.input_len:
            x_cf = F.interpolate(
                x_cf,
                size=self.input_len,
                mode="linear",
                align_corners=False,
            )

        B, C, L = x_cf.shape
        device = x_cf.device
        dtype = x_cf.dtype

        summary = self._summary(x_cf)
        raw = self.net(summary)

        scale_raw = raw[:, 0:1]
        shift_raw = raw[:, 1:2]
        warp_raw = raw[:, 2:2 + self.warp_knots].view(B, 1, self.warp_knots)
        gain_raw = raw[:, 2 + self.warp_knots:].view(B, 1, self.gain_bands)

        scale = 1.0 + self.scale_range * torch.tanh(scale_raw)
        shift = self.max_shift * torch.tanh(shift_raw)

        base_pos = torch.arange(L, device=device, dtype=dtype).view(1, 1, L)
        center = float(L - 1) / 2.0

        if self.residual_max > 0:
            delta = F.interpolate(
                torch.tanh(warp_raw),
                size=L,
                mode="linear",
                align_corners=True,
            ) * self.residual_max
        else:
            delta = torch.zeros(B, 1, L, device=device, dtype=dtype)

        sample_pos = center + scale.view(B, 1, 1) * (base_pos - center) + shift.view(B, 1, 1) + delta
        sample_pos = sample_pos.clamp(0.0, float(L - 1))

        x_norm = 2.0 * sample_pos / float(L - 1) - 1.0
        x_norm = x_norm.squeeze(1)
        y_norm = torch.zeros_like(x_norm)

        grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(1)
        x_4d = x_cf.unsqueeze(2)

        y_4d = F.grid_sample(
            x_4d,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        x_warped = y_4d.squeeze(2)

        if self.gain_delta > 0:
            gain_curve = F.interpolate(
                torch.tanh(gain_raw),
                size=L,
                mode="linear",
                align_corners=True,
            )
            gain = 1.0 + self.gain_delta * gain_curve
            x_warped = x_warped * gain
        else:
            gain_curve = torch.zeros(B, 1, L, device=device, dtype=dtype)

        scale_reg = ((scale - 1.0) / max(self.scale_range, 1e-6)).pow(2).mean()
        shift_reg = (shift / max(self.max_shift, 1e-6)).pow(2).mean()
        warp_reg = (delta / max(self.residual_max, 1e-6)).pow(2).mean()
        warp_smooth = (delta[:, :, 1:] - delta[:, :, :-1]).pow(2).mean()
        gain_reg = gain_curve.pow(2).mean()
        gain_smooth = (gain_curve[:, :, 1:] - gain_curve[:, :, :-1]).pow(2).mean()

        self._last_reg = (
            scale_reg
            + shift_reg
            + warp_reg
            + 5.0 * warp_smooth
            + gain_reg
            + 5.0 * gain_smooth
        )

        with torch.no_grad():
            self._last_info = {
                "scale_mean": float(scale.mean().item()),
                "shift_mean": float(shift.mean().item()),
                "delta_abs_max": float(delta.abs().max().item()),
                "gain_abs_mean": float(gain_curve.abs().mean().item()),
            }

        return self._restore_1d_shape(x_warped, squeezed, transposed)

    def regularization(self):
        if self._last_reg is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self._last_reg

    def info_string(self):
        if not self._last_info:
            return "scsc=NA"
        return (
            f"scsc_scale={self._last_info['scale_mean']:.5f} "
            f"scsc_shift={self._last_info['shift_mean']:.5f} "
            f"scsc_delta={self._last_info['delta_abs_max']:.5f} "
            f"scsc_gain={self._last_info['gain_abs_mean']:.5f}"
        )
