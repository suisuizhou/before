#!/usr/bin/python
# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout=0.0):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_planes, planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(planes)

        self.conv2 = nn.Conv1d(
            planes, planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(planes)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_planes, planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm1d(planes)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = F.relu(out, inplace=True)
        return out


class ResNet18_1D_SDE(nn.Module):
    """
    1D ResNet-18 backbone for FFT input.

    Input:
        [B, 512] or [B, 1, 512]

    Output:
        [B, 512]

    It keeps the same spectral-adapter interface used by current F-Warp TTA:
        apply_spectral_adapter()
        band_scale / band_bias
        input_len
        adapter_delta
    """

    def __init__(self, cfg=None):
        super().__init__()

        mcfg = cfg.Model if hasattr(cfg, "Model") else cfg

        self.input_len = int(getattr(mcfg, "input_len", 512))
        self.in_chans = int(getattr(mcfg, "in_chans", 1))
        self.drop_rate = float(getattr(mcfg, "drop_rate", 0.0))

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

        self.in_planes = 64

        self.stem = nn.Sequential(
            nn.Conv1d(
                self.in_chans, 64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_num = 512

        self._init_weights()

    def _make_layer(self, planes, blocks, stride):
        layers = []
        layers.append(BasicBlock1D(self.in_planes, planes, stride=stride, dropout=self.drop_rate))
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(self.in_planes, planes, stride=1, dropout=self.drop_rate))
        return nn.Sequential(*layers)

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

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

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
