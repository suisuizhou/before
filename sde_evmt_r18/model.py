import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _channel_first(x, length=512):
    if x.ndim == 2:
        x = x[:, None, :]
    elif x.ndim == 3 and x.shape[-1] == 1:
        x = x.transpose(1, 2)
    if x.ndim != 3:
        raise ValueError(f"expected [B,L] or [B,C,L], got {tuple(x.shape)}")
    if x.shape[-1] != length:
        x = F.interpolate(x, size=length, mode="linear", align_corners=False)
    return x


def _gate_logit(probability):
    probability = float(probability)
    if not 0.0 < probability < 1.0:
        raise ValueError("initial gate must be in (0,1)")
    return math.log(probability / (1.0 - probability))


class RobustSpectrumNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(self, x):
        x = _channel_first(x)
        x = torch.log1p(x.clamp_min(0.0))
        median = x.median(dim=-1, keepdim=True).values
        q1 = torch.quantile(x, 0.25, dim=-1, keepdim=True)
        q3 = torch.quantile(x, 0.75, dim=-1, keepdim=True)
        return (x - median) / (q3 - q1).clamp_min(self.eps)


class GatedFrequencyWarp(nn.Module):
    def __init__(self, input_len=512, knots=16, max_warp=2.0, initial_gate=0.05):
        super().__init__()
        self.input_len = int(input_len)
        self.max_warp = float(max_warp)
        self.warp_ctrl = nn.Parameter(torch.zeros(1, 1, int(knots)))
        self.warp_gate_logit = nn.Parameter(torch.tensor(_gate_logit(initial_gate)))

    @property
    def gate(self):
        return self.warp_gate_logit.sigmoid()

    @property
    def gate_value(self):
        return float(self.gate.detach())

    def displacement(self):
        control = F.interpolate(
            torch.tanh(self.warp_ctrl),
            size=self.input_len,
            mode="linear",
            align_corners=True,
        )
        return self.gate * self.max_warp * control

    def forward(self, x):
        x = _channel_first(x, self.input_len)
        batch, _, length = x.shape
        delta = self.displacement()
        base = torch.arange(length, device=x.device, dtype=x.dtype)[None, None]
        position = (base + delta.to(x)).clamp(0, length - 1)
        left = position.floor().long()
        right = (left + 1).clamp_max(length - 1)
        weight = position - left.to(position.dtype)
        channels = x.shape[1]
        left = left.expand(batch, channels, length)
        right = right.expand(batch, channels, length)
        weight = weight.expand(batch, channels, length)
        return (
            x.gather(2, left) * (1.0 - weight)
            + x.gather(2, right) * weight
        )


class GatedSpectralAdapter(nn.Module):
    def __init__(self, input_len=512, bands=64, delta=0.1, initial_gate=0.1):
        super().__init__()
        self.input_len = int(input_len)
        self.adapter_delta = float(delta)
        self.band_scale = nn.Parameter(torch.zeros(1, 1, int(bands)))
        self.band_bias = nn.Parameter(torch.zeros(1, 1, int(bands)))
        self.adapter_gate_logit = nn.Parameter(torch.tensor(_gate_logit(initial_gate)))

    @property
    def gate(self):
        return self.adapter_gate_logit.sigmoid()

    @property
    def gate_value(self):
        return float(self.gate.detach())

    def transformed_parameters(self):
        scale_control = F.interpolate(
            torch.tanh(self.band_scale), self.input_len, mode="linear", align_corners=True
        )
        bias_control = F.interpolate(
            self.band_bias, self.input_len, mode="linear", align_corners=True
        )
        scale = 1.0 + self.gate * self.adapter_delta * scale_control
        bias = self.gate * self.adapter_delta * bias_control
        return scale, bias

    def forward(self, x):
        x = _channel_first(x, self.input_len)
        scale, bias = self.transformed_parameters()
        return x * scale.to(x) + bias.to(x)


class ResidualFeatureAdapter(nn.Module):
    def __init__(self, dim=512, hidden_dim=64, initial_gate=0.01):
        super().__init__()
        self.norm = nn.LayerNorm(int(dim))
        self.down = nn.Linear(int(dim), int(hidden_dim))
        self.up = nn.Linear(int(hidden_dim), int(dim))
        self.feature_gate_logit = nn.Parameter(torch.tensor(_gate_logit(initial_gate)))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    @property
    def gate(self):
        return self.feature_gate_logit.sigmoid()

    @property
    def gate_value(self):
        return float(self.gate.detach())

    def forward(self, feature):
        residual = self.up(F.gelu(self.down(self.norm(feature))))
        return feature + self.gate * residual


def _group_count(channels, preferred=8):
    for groups in range(min(int(preferred), int(channels)), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class BasicBlock1DGN(nn.Module):
    def __init__(self, in_channels, channels, stride=1, groups=8):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, channels, 3, stride, 1, bias=False)
        self.gn1 = nn.GroupNorm(_group_count(channels, groups), channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, 1, 1, bias=False)
        self.gn2 = nn.GroupNorm(_group_count(channels, groups), channels)
        self.downsample = (
            nn.Sequential(
                nn.Conv1d(in_channels, channels, 1, stride, bias=False),
                nn.GroupNorm(_group_count(channels, groups), channels),
            )
            if stride != 1 or in_channels != channels
            else nn.Identity()
        )

    def forward(self, x):
        identity = self.downsample(x)
        x = F.relu(self.gn1(self.conv1(x)), inplace=True)
        x = self.gn2(self.conv2(x))
        return F.relu(x + identity, inplace=True)


class SDEEVMTResNet18(nn.Module):
    def __init__(
        self,
        num_classes=32,
        input_len=512,
        bottleneck_dim=256,
        feature_hidden_dim=64,
        gn_groups=8,
        warp_knots=16,
        max_warp=2.0,
        spectral_bands=64,
        adapter_delta=0.1,
    ):
        super().__init__()
        self.input_len = int(input_len)
        self.num_classes = int(num_classes)
        self.robust_norm = RobustSpectrumNorm()
        self.frequency_warp = GatedFrequencyWarp(input_len, warp_knots, max_warp, 0.05)
        self.spectral_adapter = GatedSpectralAdapter(
            input_len, spectral_bands, adapter_delta, 0.1
        )
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, 7, 2, 3, bias=False),
            nn.GroupNorm(_group_count(64, gn_groups), 64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, 2, 1),
        )
        channels = [64, 128, 256, 512]
        in_channels = 64
        layers = []
        for index, channel in enumerate(channels):
            stride = 1 if index == 0 else 2
            layers.append(
                nn.Sequential(
                    BasicBlock1DGN(in_channels, channel, stride, gn_groups),
                    BasicBlock1DGN(channel, channel, 1, gn_groups),
                )
            )
            in_channels = channel
        self.layer1, self.layer2, self.layer3, self.layer4 = layers
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.feature_adapter = ResidualFeatureAdapter(512, feature_hidden_dim, 0.01)
        self.bottleneck = nn.Sequential(
            nn.Linear(512, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(bottleneck_dim, num_classes)
        self.output_num = int(bottleneck_dim)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def prepare_spectrum(self, x):
        x = self.robust_norm(x)
        x = self.frequency_warp(x)
        return self.spectral_adapter(x)

    def forward_from_spectrum(self, spectrum, use_feature_adapter=True):
        x = self.stem(spectrum)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        raw_feature = self.pool(x).flatten(1)
        if use_feature_adapter:
            raw_feature = self.feature_adapter(raw_feature)
        feature = self.bottleneck(raw_feature)
        return feature, self.classifier(feature)

    def forward_parts(self, x, use_feature_adapter=True):
        return self.forward_from_spectrum(
            self.prepare_spectrum(x), use_feature_adapter=use_feature_adapter
        )

    def forward(self, x):
        return self.forward_parts(x)[1]

    def gate_summary(self):
        return {
            "warp_gate": self.frequency_warp.gate_value,
            "spectral_gate": self.spectral_adapter.gate_value,
            "feature_gate": self.feature_adapter.gate_value,
        }
