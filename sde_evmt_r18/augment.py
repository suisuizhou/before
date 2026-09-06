import torch
import torch.nn.functional as F


def cfg_get(cfg, path, default=None):
    value = cfg
    for key in path.split("."):
        if isinstance(value, dict):
            if key not in value:
                return default
            value = value[key]
        elif hasattr(value, key):
            value = getattr(value, key)
        else:
            return default
    return value


def _as_spectrum(x):
    if x.ndim == 2:
        return x[:, None, :], True
    if x.ndim == 3 and x.shape[1] == 1:
        return x, False
    raise ValueError(f"expected [B,L] or [B,1,L], got {tuple(x.shape)}")


def _restore(x, squeezed):
    return x.squeeze(1) if squeezed else x


class SourceAugmenter:
    def __init__(self, cfg, generator=None):
        self.cfg = cfg
        self.generator = generator

    def _rand(self, shape, reference):
        return torch.rand(
            shape, dtype=reference.dtype, device=reference.device, generator=self.generator
        )

    def _smooth_control(self, batch, knots, reference):
        control = self._rand((batch, 1, int(knots)), reference) * 2.0 - 1.0
        return F.interpolate(
            control, size=reference.shape[-1], mode="linear", align_corners=True
        )

    def style(self, x):
        batch = x.shape[0]
        probability = float(cfg_get(self.cfg, "augmentation.style_prob", 0.5))
        strength_min = float(cfg_get(self.cfg, "augmentation.style_strength_min", 0.05))
        strength_max = float(cfg_get(self.cfg, "augmentation.style_strength_max", 0.2))
        knots = int(cfg_get(self.cfg, "augmentation.style_knots", 16))
        strength = strength_min + (strength_max - strength_min) * self._rand((batch, 1, 1), x)
        mask = (1.0 + strength * self._smooth_control(batch, knots, x)).clamp_min(0.1)
        apply = (self._rand((batch, 1, 1), x) < probability).to(x.dtype)
        return x * (1.0 + apply * (mask - 1.0))

    def warp(self, x):
        batch, channels, length = x.shape
        probability = float(cfg_get(self.cfg, "augmentation.warp_prob", 0.5))
        knots = int(cfg_get(self.cfg, "augmentation.warp_knots", 16))
        warp_min = float(cfg_get(self.cfg, "augmentation.warp_max_min", 0.3))
        warp_max = float(cfg_get(self.cfg, "augmentation.warp_max_max", 2.0))
        strength = warp_min + (warp_max - warp_min) * self._rand((batch, 1, 1), x)
        displacement = strength * self._smooth_control(batch, knots, x)
        apply = (self._rand((batch, 1, 1), x) < probability).to(x.dtype)
        base = torch.arange(length, device=x.device, dtype=x.dtype)[None, None]
        position = (base + apply * displacement).clamp(0, length - 1)
        left = position.floor().long().expand(batch, channels, length)
        right = (left + 1).clamp_max(length - 1)
        weight = (position - position.floor()).expand(batch, channels, length)
        return x.gather(2, left) * (1.0 - weight) + x.gather(2, right) * weight

    def noise(self, x):
        batch, _, length = x.shape
        probability = float(cfg_get(self.cfg, "augmentation.noise_prob", 0.5))
        sample_std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        family = torch.randint(
            0, 3, (batch, 1, 1), device=x.device, generator=self.generator
        )
        snr_min = float(cfg_get(self.cfg, "augmentation.gaussian_snr_min", 25.0))
        snr_max = float(cfg_get(self.cfg, "augmentation.gaussian_snr_max", 40.0))
        snr = snr_min + (snr_max - snr_min) * self._rand((batch, 1, 1), x)
        signal_rms = x.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        gaussian = x + torch.randn(
            x.shape, dtype=x.dtype, device=x.device, generator=self.generator
        ) * signal_rms / torch.pow(torch.tensor(10.0, device=x.device), snr / 20.0)
        uniform_min = float(cfg_get(self.cfg, "augmentation.uniform_scale_min", 0.02))
        uniform_max = float(cfg_get(self.cfg, "augmentation.uniform_scale_max", 0.08))
        uniform_scale = uniform_min + (uniform_max - uniform_min) * self._rand((batch, 1, 1), x)
        uniform = x + (self._rand(x.shape, x) * 2.0 - 1.0) * sample_std * uniform_scale
        impulse_min = float(cfg_get(self.cfg, "augmentation.impulse_prob_min", 0.01))
        impulse_max = float(cfg_get(self.cfg, "augmentation.impulse_prob_max", 0.05))
        impulse_p = impulse_min + (impulse_max - impulse_min) * self._rand((batch, 1, 1), x)
        impulse_mask = self._rand((batch, 1, length), x) < impulse_p
        impulse_sign = torch.where(self._rand((batch, 1, length), x) < 0.5, -1.0, 1.0)
        impulse = x + impulse_mask * impulse_sign * sample_std * (
            2.0 + 4.0 * self._rand((batch, 1, 1), x)
        )
        candidate = torch.where(family == 0, gaussian, torch.where(family == 1, uniform, impulse))
        candidate = candidate * (0.9 + 0.2 * self._rand((batch, 1, 1), x))
        apply = self._rand((batch, 1, 1), x) < probability
        return torch.where(apply, candidate, x).clamp_min(0.0)

    def views(self, x):
        spectrum, squeezed = _as_spectrum(x)
        return {
            "clean": x,
            "style": _restore(self.style(spectrum), squeezed),
            "warp": _restore(self.warp(spectrum), squeezed),
            "noise": _restore(self.noise(spectrum), squeezed),
        }
