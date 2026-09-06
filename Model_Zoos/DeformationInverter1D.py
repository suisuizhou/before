#!/usr/bin/python
# -*- coding: UTF-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformationInverter1D(nn.Module):
    """
    DSI: Deformation-Supervised Inverter.

    It predicts a sample-wise inverse spectral transform:
        scale correction
        shift correction
        coarse band gain correction

    It is identity-initialized.
    Source training provides synthetic deformation supervision.
    """

    def __init__(
        self,
        input_len=512,
        summary_bands=32,
        gain_bands=16,
        hidden_dim=64,
        scale_range=0.12,
        max_shift=4.0,
        gain_delta=0.08,
    ):
        super().__init__()

        self.input_len = int(input_len)
        self.summary_bands = int(summary_bands)
        self.gain_bands = int(gain_bands)
        self.hidden_dim = int(hidden_dim)

        self.scale_range = float(scale_range)
        self.max_shift = float(max_shift)
        self.gain_delta = float(gain_delta)

        in_dim = self.summary_bands + 3
        out_dim = 2 + self.gain_bands

        self.net = nn.Sequential(
            nn.Linear(in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, out_dim),
        )

        # exact identity at initialization
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        self.last_scale_norm = None
        self.last_shift_norm = None
        self.last_gain_norm = None
        self.last_reg = None
        self.last_info = {}

    def _to_cf(self, x):
        squeezed = False
        transposed = False

        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeezed = True
        elif x.dim() == 3 and x.shape[-1] == 1:
            x = x.transpose(1, 2)
            transposed = True
        elif x.dim() != 3:
            raise ValueError(f"Unexpected x shape: {x.shape}")

        return x, squeezed, transposed

    def _restore(self, x, squeezed, transposed):
        if transposed:
            x = x.transpose(1, 2)
        if squeezed:
            x = x.squeeze(1)
        return x

    def _summary(self, x_cf):
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
            dtype=x_cf.dtype,
        ).view(1, -1)

        centroid = (bands * freq).sum(dim=1, keepdim=True)
        spread = torch.sqrt(
            (bands * (freq - centroid).pow(2)).sum(dim=1, keepdim=True).clamp_min(1e-8)
        )

        return torch.cat([bands, entropy, centroid, spread], dim=1)

    def forward(self, x):
        x_cf, squeezed, transposed = self._to_cf(x)

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

        raw = self.net(self._summary(x_cf))

        scale_norm = torch.tanh(raw[:, 0:1])
        shift_norm = torch.tanh(raw[:, 1:2])
        gain_norm = torch.tanh(raw[:, 2:].view(B, 1, self.gain_bands))

        scale = 1.0 + self.scale_range * scale_norm
        shift = self.max_shift * shift_norm

        base_pos = torch.arange(L, device=device, dtype=dtype).view(1, 1, L)
        center = float(L - 1) / 2.0

        sample_pos = center + scale.view(B, 1, 1) * (base_pos - center) + shift.view(B, 1, 1)
        sample_pos = sample_pos.clamp(0.0, float(L - 1))

        x_norm = 2.0 * sample_pos / float(L - 1) - 1.0
        x_norm = x_norm.squeeze(1)
        y_norm = torch.zeros_like(x_norm)
        grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(1)

        y = F.grid_sample(
            x_cf.unsqueeze(2),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(2)

        if self.gain_delta > 0:
            gain_curve = F.interpolate(
                gain_norm,
                size=L,
                mode="linear",
                align_corners=True,
            )
            y = y * (1.0 + self.gain_delta * gain_curve)
        else:
            gain_curve = torch.zeros(B, 1, L, device=device, dtype=dtype)

        self.last_scale_norm = scale_norm
        self.last_shift_norm = shift_norm
        self.last_gain_norm = gain_norm

        gain_smooth = (gain_curve[:, :, 1:] - gain_curve[:, :, :-1]).pow(2).mean()
        self.last_reg = (
            scale_norm.pow(2).mean()
            + shift_norm.pow(2).mean()
            + 0.2 * gain_norm.pow(2).mean()
            + 2.0 * gain_smooth
        )

        with torch.no_grad():
            self.last_info = {
                "scale": float(scale.mean().item()),
                "shift": float(shift.mean().item()),
                "gain_abs": float(gain_norm.abs().mean().item()),
            }

        return self._restore(y, squeezed, transposed)

    def param_loss(self, target_scale_norm, target_shift_norm):
        if self.last_scale_norm is None or self.last_shift_norm is None:
            return torch.tensor(0.0, device=target_scale_norm.device)

        return (
            F.mse_loss(self.last_scale_norm, target_scale_norm)
            + F.mse_loss(self.last_shift_norm, target_shift_norm)
        )

    def regularization(self):
        if self.last_reg is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self.last_reg

    def info_string(self):
        if not self.last_info:
            return "dsi=NA"
        return (
            f"dsi_scale={self.last_info['scale']:.5f} "
            f"dsi_shift={self.last_info['shift']:.5f} "
            f"dsi_gain={self.last_info['gain_abs']:.5f}"
        )
