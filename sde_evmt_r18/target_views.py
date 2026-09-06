import torch
import torch.nn.functional as F

from .augment import cfg_get


def _as_spectrum(x):
    if x.ndim == 2:
        return x[:, None, :], True
    if x.ndim == 3 and x.shape[1] == 1:
        return x, False
    raise ValueError(f"expected [B,L] or [B,1,L], got {tuple(x.shape)}")


def _restore(x, squeezed):
    return x.squeeze(1) if squeezed else x


def _rand(shape, reference, generator):
    return torch.rand(
        shape,
        dtype=reference.dtype,
        device=reference.device,
        generator=generator,
    )


def _smooth_random(batch, length, reference, generator, knots=16):
    control = _rand((batch, 1, int(knots)), reference, generator) * 2.0 - 1.0
    return F.interpolate(control, size=length, mode="linear", align_corners=True)


def _weak_style(x, cfg, generator):
    batch, _, length = x.shape
    strength = float(cfg_get(cfg, "teacher_views.style_strength", 0.05))
    control = _smooth_random(batch, length, x, generator)
    return (x * (1.0 + strength * control)).clamp_min(0.0)


def _weak_warp(x, cfg, generator):
    batch, channels, length = x.shape
    maximum = float(cfg_get(cfg, "teacher_views.warp_max", 0.5))
    displacement = maximum * _smooth_random(batch, length, x, generator)
    base = torch.arange(length, dtype=x.dtype, device=x.device)[None, None]
    position = (base + displacement).clamp(0, length - 1)
    left = position.floor().long().expand(batch, channels, length)
    right = (left + 1).clamp_max(length - 1)
    weight = (position - position.floor()).expand(batch, channels, length)
    return x.gather(2, left) * (1.0 - weight) + x.gather(2, right) * weight


def _weak_noise_gain(x, cfg, generator):
    batch, _, length = x.shape
    snr_min = float(cfg_get(cfg, "teacher_views.noise_snr_min", 35.0))
    snr_max = float(cfg_get(cfg, "teacher_views.noise_snr_max", 45.0))
    snr = snr_min + (snr_max - snr_min) * _rand((batch, 1, 1), x, generator)
    rms = x.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    noise = torch.randn(
        x.shape, dtype=x.dtype, device=x.device, generator=generator
    ) * rms / torch.pow(x.new_tensor(10.0), snr / 20.0)
    gain_min = float(cfg_get(cfg, "teacher_views.gain_min", 0.97))
    gain_max = float(cfg_get(cfg, "teacher_views.gain_max", 1.03))
    gain = gain_min + (gain_max - gain_min) * _rand((batch, 1, 1), x, generator)
    baseline_scale = float(cfg_get(cfg, "teacher_views.baseline_scale", 0.01))
    baseline = baseline_scale * rms * _smooth_random(
        batch, length, x, generator, knots=4
    )
    return (gain * x + noise + baseline).clamp_min(0.0)


def make_teacher_views(x, cfg, generator=None):
    """Return clean, weak style, weak warp and weak noise/gain views."""
    spectrum, squeezed = _as_spectrum(x)
    return [
        x,
        _restore(_weak_style(spectrum, cfg, generator), squeezed),
        _restore(_weak_warp(spectrum, cfg, generator), squeezed),
        _restore(_weak_noise_gain(spectrum, cfg, generator), squeezed),
    ]
