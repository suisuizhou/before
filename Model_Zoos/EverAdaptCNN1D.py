#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class EverAdaptCNN1D(nn.Module):
    """
    EverAdapt-style 1D CNN encoder:
      Block1: Conv1D 128 channels, kernel 5, BN, ReLU, MaxPool, Dropout 0.5
      Block2: Conv1D 256 channels, kernel 8, BN, ReLU, MaxPool
      Block3: Conv1D 128 channels, kernel 8, BN, ReLU, MaxPool
      AdaptiveAvgPool -> feature dim 128

    This version accepts current FFT input length 512 by default.
    It also keeps the weak spectral adapter interface so that the existing
    F-Warp TTA script can update band_scale / band_bias / warp_ctrl.
    """

    def __init__(self, cfg=None):
        super().__init__()

        mcfg = cfg.Model if hasattr(cfg, "Model") else cfg

        self.input_len = int(getattr(mcfg, "input_len", 512))
        self.in_chans = int(getattr(mcfg, "in_chans", 1))
        self.drop_rate = float(getattr(mcfg, "drop_rate", 0.5))

        self.use_spectral_adapter = bool(getattr(mcfg, "use_spectral_adapter", False))
        self.band_num = int(getattr(mcfg, "band_num", 256))
        self.adapter_delta = float(getattr(mcfg, "adapter_delta", 0.1))

        if self.use_spectral_adapter:
            assert self.input_len % self.band_num == 0, "input_len must be divisible by band_num"
            self.band_width = self.input_len // self.band_num
            self.band_scale = nn.Parameter(torch.zeros(1, 1, self.band_num))
            self.band_bias = nn.Parameter(torch.zeros(1, 1, self.band_num))
        else:
            self.band_width = None
            self.band_scale = None
            self.band_bias = None

        self.conv1 = nn.Sequential(
            nn.Conv1d(self.in_chans, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(p=self.drop_rate),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=8, padding=4, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=8, padding=4, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_num = 128

        self._init_weights()

    def _to_channel_first(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.shape[-1] == 1:
            x = x.transpose(1, 2)

        if x.dim() != 3:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        if x.shape[-1] != self.input_len:
            x = F.interpolate(x, size=self.input_len, mode="linear", align_corners=False)

        return x

    def apply_spectral_adapter(self, x):
        x = self._to_channel_first(x)

        if (not self.use_spectral_adapter) or (self.band_scale is None):
            return x

        B, C, L = x.shape
        if L != self.input_len:
            x = F.interpolate(x, size=self.input_len, mode="linear", align_corners=False)
            L = self.input_len

        x = x.view(B, C, self.band_num, self.band_width)

        scale = 1.0 + self.adapter_delta * torch.tanh(self.band_scale)
        bias = self.adapter_delta * self.band_bias

        x = x * scale.unsqueeze(-1) + bias.unsqueeze(-1)
        x = x.view(B, C, L)
        return x

    def forward_features(self, x):
        x = self.apply_spectral_adapter(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x).squeeze(-1)
        return x

    def forward(self, x):
        return self.forward_features(x)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


if __name__ == "__main__":
    model = EverAdaptCNN1D(cfg=type("cfg", (), {
        "Model": type("m", (), {
            "input_len": 512,
            "in_chans": 1,
            "drop_rate": 0.5,
            "use_spectral_adapter": True,
            "band_num": 256,
            "adapter_delta": 0.1,
        })()
    })())
    y = model(torch.randn(2, 512))
    print(y.shape)
