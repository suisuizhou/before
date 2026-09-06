"""Leakage-free ordinary and 0711-robust source training for WTPG."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
import time

import torch
import torch.nn.functional as F
from omegaconf import open_dict
from torch.utils.data import DataLoader

import Dataset
from Lib.hust_source_training import (
    ROBUST_SOURCE_DEFAULTS,
    _carrier_is_identity,
    build_source_optimizer,
    source_loss,
    spectral_style_augment,
    spectral_warp_augment,
)
from Lib.hust_strict_protocol import ensure_hust_adaptation_carrier
from Lib.model import get_model
from Lib.pu4d_common_source import forward_parts
from Lib.pu4d_vanilla_protocol import reset_and_freeze_adaptation_carrier
from Lib.train_utils import seed_torch
from Lib.wtpg_strict_protocol import (
    PROCESSING,
    SPEEDS,
    checkpoint_dir,
    sha256_file,
    strict_load_checkpoint,
    validate_cache,
)


HYPERPARAMETERS = {
    "epochs": 50,
    "batch_size": 128,
    "num_workers": 4,
    "optimizer": "AdamW",
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "label_smoothing": 0.1,
    "seed": 2025,
}

# WTPG spectra are log-amplitudes followed by per-window Z-score.  Multiplying
# those signed standardized values (the generic HUST SSP transform) does not
# faithfully represent a sensor/background change.  This profile therefore
# uses a smooth additive spectral response plus weak bin noise.  Motor-speed
# changes move gear-mesh peaks approximately proportionally along frequency,
# so SDE also includes a bounded global frequency dilation in addition to the
# original small local warp.
WTPG_ROBUST_DEFAULTS = {
    "profile": "wtpg_speed_noise_v1",
    "response_prob": 0.8,
    "response_strength": 0.10,
    "response_knots": 8,
    "noise_std": 0.02,
    "scale_prob": 0.8,
    "scale_min": 0.88,
    "scale_max": 1.12,
    "local_warp_max": 1.0,
    "local_warp_knots": 16,
}


def prepare_wtpg_source_config(cfg) -> int:
    try:
        source = int(cfg.only_source)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("only_source=0..7 is required") from exc
    if source not in SPEEDS:
        raise ValueError(f"invalid WTPG source domain: {source}")
    with open_dict(cfg):
        cfg.seed_run = 2025
        cfg.batch_size = 128
        cfg.num_workers = 4
        cfg.src_epoch = 50
        cfg.Dataset.data_name = "WTPGStrict"
        cfg.Dataset.data_path = "Dataset/WTPG_STRICT_CACHE_V1"
        cfg.Dataset.TL_list = list(SPEEDS)
        cfg.Dataset.TL_Task = [source, (source + 1) % len(SPEEDS)]
        cfg.Dataset.input_kind = "fft"
        cfg.Dataset.norm_kind = "pre_normalized"
        cfg.Model.model_name = "ResNet18_1D_SDE"
        cfg.Model.use_spectral_adapter = True
        cfg.Model.band_num = 256
        cfg.Model.input_len = 512
        cfg.Model.bottleneck = True
        cfg.Model.bottleneck_num = 128
        cfg.Model.model_type = "linear"
        cfg.Opt.name = "adamw"
        cfg.Opt.lr_src = 0.001
        cfg.Opt.weight_decay_src = 0.0001
        cfg.model_name = "ResNet18_1D_SDE2025fft_Linear.pt"
        if not hasattr(cfg, "wtpg_checkpoint_root"):
            cfg.wtpg_checkpoint_root = "TTA_Model_WTPG_STRICT_V1"
    return source


def _source_dataset(cfg, source):
    dataset = Dataset.WTPGStrict.__new__(Dataset.WTPGStrict)
    dataset.root = Path(str(cfg.Dataset.data_path))
    dataset.source = int(source)
    dataset.target = (int(source) + 1) % 8
    return dataset._load(source)


def wtpg_response_noise_augment(
    x, *, strength=0.10, knots=8, noise_std=0.02, prob=0.8
):
    """Add a smooth response/background perturbation in standardized units."""
    if random.random() > float(prob):
        return x
    if min(float(strength), float(noise_std)) < 0:
        raise ValueError("WTPG response/noise strengths must be non-negative")
    original_shape = x.shape
    x_cf = x.unsqueeze(1) if x.ndim == 2 else x
    if x_cf.ndim != 3:
        raise ValueError(f"unexpected WTPG spectral shape: {tuple(x.shape)}")
    batch, _channels, length = x_cf.shape
    ctrl = x_cf.new_empty(batch, 1, max(2, int(knots))).uniform_(-strength, strength)
    response = F.interpolate(ctrl, size=length, mode="linear", align_corners=True)
    perturbed = x_cf + response
    if noise_std:
        perturbed = perturbed + torch.randn_like(perturbed) * float(noise_std)
    return perturbed.squeeze(1) if len(original_shape) == 2 else perturbed


def wtpg_speed_scale_augment(
    x, *, scale_min=0.88, scale_max=1.12, prob=0.8
):
    """Apply class-preserving global frequency dilation for speed variation."""
    if not 0 < float(scale_min) <= float(scale_max):
        raise ValueError("invalid WTPG frequency-scale interval")
    if random.random() > float(prob):
        return x
    original_shape = x.shape
    x_cf = x.unsqueeze(1) if x.ndim == 2 else x
    if x_cf.ndim != 3:
        raise ValueError(f"unexpected WTPG spectral shape: {tuple(x.shape)}")
    batch, _channels, length = x_cf.shape
    scale = x_cf.new_empty(batch, 1, 1).uniform_(scale_min, scale_max)
    output_bins = torch.arange(length, device=x.device, dtype=x.dtype).view(1, 1, -1)
    source_bins = (output_bins / scale).clamp(0.0, float(length - 1))
    x_norm = (2.0 * source_bins / float(length - 1) - 1.0).squeeze(1)
    grid = torch.stack([x_norm, torch.zeros_like(x_norm)], dim=-1).unsqueeze(1)
    scaled = F.grid_sample(
        x_cf.unsqueeze(2), grid, mode="bilinear", padding_mode="border", align_corners=True
    ).squeeze(2)
    return scaled.squeeze(1) if len(original_shape) == 2 else scaled


def _forward_robust(model, x, labels, num_classes, profile="generic_v1"):
    clean_feature, clean_logits = forward_parts(model, x)
    if profile == WTPG_ROBUST_DEFAULTS["profile"]:
        style_x = wtpg_response_noise_augment(
            x, strength=WTPG_ROBUST_DEFAULTS["response_strength"],
            knots=WTPG_ROBUST_DEFAULTS["response_knots"],
            noise_std=WTPG_ROBUST_DEFAULTS["noise_std"],
            prob=WTPG_ROBUST_DEFAULTS["response_prob"],
        )
        warp_x = wtpg_speed_scale_augment(
            x, scale_min=WTPG_ROBUST_DEFAULTS["scale_min"],
            scale_max=WTPG_ROBUST_DEFAULTS["scale_max"],
            prob=WTPG_ROBUST_DEFAULTS["scale_prob"],
        )
        warp_x = spectral_warp_augment(
            warp_x, max_warp=WTPG_ROBUST_DEFAULTS["local_warp_max"],
            knots=WTPG_ROBUST_DEFAULTS["local_warp_knots"],
            prob=WTPG_ROBUST_DEFAULTS["scale_prob"],
        )
        style_warp_x = wtpg_response_noise_augment(
            warp_x, strength=WTPG_ROBUST_DEFAULTS["response_strength"],
            knots=WTPG_ROBUST_DEFAULTS["response_knots"],
            noise_std=WTPG_ROBUST_DEFAULTS["noise_std"],
            prob=WTPG_ROBUST_DEFAULTS["response_prob"],
        )
    elif profile == "generic_v1":
        style_x = spectral_style_augment(
            x, strength=ROBUST_SOURCE_DEFAULTS["ssp_style_strength"],
            knots=ROBUST_SOURCE_DEFAULTS["ssp_style_knots"],
            prob=ROBUST_SOURCE_DEFAULTS["ssp_style_prob"],
        )
        warp_x = spectral_warp_augment(
            x, max_warp=ROBUST_SOURCE_DEFAULTS["sde_warp_max"],
            knots=ROBUST_SOURCE_DEFAULTS["sde_warp_knots"],
            prob=ROBUST_SOURCE_DEFAULTS["sde_warp_prob"],
        )
        style_warp_x = spectral_style_augment(
            warp_x, strength=ROBUST_SOURCE_DEFAULTS["ssp_style_strength"],
            knots=ROBUST_SOURCE_DEFAULTS["ssp_style_knots"],
            prob=ROBUST_SOURCE_DEFAULTS["ssp_style_prob"],
        )
    else:
        raise ValueError(f"unknown WTPG robust profile: {profile}")
    style_feature, style_logits = forward_parts(model, style_x)
    warp_feature, warp_logits = forward_parts(model, warp_x)
    style_warp_feature, style_warp_logits = forward_parts(model, style_warp_x)
    loss, terms = source_loss(
        "robust", clean_logits, labels, num_classes,
        style_logits=style_logits, warp_logits=warp_logits,
        style_warp_logits=style_warp_logits, clean_feature=clean_feature,
        augmented_features=[style_feature, warp_feature, style_warp_feature],
    )
    return clean_logits, loss, terms


def _publish(model, directory, model_name, metadata):
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}.tmp-", dir=directory.parent))
    try:
        path = temporary / model_name
        state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        payload = {
            "contract": {
                "version": 1, "dataset": "WTPGStrict", "route": metadata["route"],
                "source": metadata["source"], "seed": metadata["seed"], "epoch": 50,
            },
            "state_dict": state,
        }
        torch.save(payload, path)
        metadata["checkpoint_sha256"] = sha256_file(path)
        (temporary / "source_training_summary.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + os.linesep, encoding="utf-8"
        )
        os.replace(temporary, directory)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def train_wtpg_source(cfg, source: int, variant: str):
    if variant not in {"ordinary", "robust"}:
        raise ValueError(f"invalid WTPG source route: {variant}")
    robust_profile = str(getattr(cfg, "wtpg_robust_profile", "generic_v1"))
    if variant == "ordinary":
        robust_profile = "not_applicable"
    elif robust_profile not in {"generic_v1", WTPG_ROBUST_DEFAULTS["profile"]}:
        raise ValueError(f"unknown WTPG robust profile: {robust_profile}")
    seed_torch(2025)
    random.seed(2025)
    cache = validate_cache(Path(str(cfg.Dataset.data_path)))
    cache_identity = {
        "manifest_path": str((Path(str(cfg.Dataset.data_path)) / "manifest.json").resolve()),
        "manifest_sha256": cache["manifest_sha256"],
        "content_sha256": cache["content_sha256"],
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_data = _source_dataset(cfg, source)
    loader = DataLoader(source_data, batch_size=128, shuffle=True, num_workers=4, drop_last=False)
    model = get_model(num_classes=5, cfg=cfg, **cfg.Model).to(device)
    carriers = ensure_hust_adaptation_carrier(model, ROBUST_SOURCE_DEFAULTS["sde_warp_knots"])
    reset_and_freeze_adaptation_carrier(model)
    if not _carrier_is_identity(model, carriers):
        raise RuntimeError("WTPG adaptation carrier did not initialize to identity")
    root = Path(str(cfg.wtpg_checkpoint_root))
    directory = checkpoint_dir(root, variant, source, 2025)
    model_name = str(cfg.model_name)
    checkpoint = directory / model_name
    if checkpoint.exists() or (directory / "source_training_summary.json").exists():
        metadata = strict_load_checkpoint(model, checkpoint)
        print("WTPG_SOURCE_RESULT_JSON=" + json.dumps(metadata, sort_keys=True), flush=True)
        return metadata
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite partial WTPG checkpoint directory: {directory}")
    optimizer, _ = build_source_optimizer(model)
    expected_terms = {"clean"} if variant == "ordinary" else {
        "clean", "style", "warp", "style_warp", "symmetric_kl", "feature",
    }
    totals = {name: 0.0 for name in expected_terms}
    samples, optimizer_steps = 0, 0
    started = time.monotonic()
    correct = seen = 0
    for epoch in range(50):
        model.train()
        correct = seen = 0
        for x, labels, _ in loader:
            x, labels = x.to(device), labels.to(device)
            if variant == "ordinary":
                _, logits = forward_parts(model, x)
                loss, terms = source_loss("ordinary", logits, labels, 5)
            else:
                logits, loss, terms = _forward_robust(
                    model, x, labels, 5, profile=robust_profile
                )
            if set(terms) != expected_terms or not bool(torch.isfinite(loss)):
                raise RuntimeError("invalid WTPG source loss")
            batch = int(labels.numel())
            for name, term in terms.items():
                value = float(term.detach())
                if not math.isfinite(value):
                    raise RuntimeError("non-finite WTPG source loss term")
                totals[name] += value * batch
            samples += batch
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            correct += int((logits.argmax(1) == labels).sum())
            seen += batch
        print(f"WTPG {variant} source={source} epoch={epoch + 1}/50 source_acc={100.0 * correct / seen:.4f}%", flush=True)
    if not _carrier_is_identity(model, carriers):
        raise RuntimeError("WTPG source training modified adaptation carrier")
    metadata = {
        "contract_version": 1,
        "dataset": "WTPGStrict",
        "route": variant,
        "robust_profile": robust_profile,
        "robust_augmentation": (
            dict(WTPG_ROBUST_DEFAULTS)
            if robust_profile == WTPG_ROBUST_DEFAULTS["profile"] else
            (dict(ROBUST_SOURCE_DEFAULTS) if robust_profile == "generic_v1" else None)
        ),
        "source": int(source),
        "speed_hz": SPEEDS[int(source)],
        "seed": 2025,
        "epochs": 50,
        "hyperparameters": HYPERPARAMETERS,
        "cache_identity": cache_identity,
        "target_labels_consumed": False,
        "carrier_identity_check": True,
        "carrier_parameters": carriers,
        "tensor_count": len(model.state_dict()),
        "source_accuracy": 100.0 * correct / max(seen, 1),
        "optimizer_steps": optimizer_steps,
        "elapsed_seconds": time.monotonic() - started,
        "checkpoint": model_name,
        "checkpoint_sha256": "pending",
        "loss_terms": {name: totals[name] / samples for name in sorted(totals)},
    }
    _publish(model, directory, model_name, metadata)
    metadata = strict_load_checkpoint(model, checkpoint)
    print("WTPG_SOURCE_RESULT_JSON=" + json.dumps(metadata, sort_keys=True), flush=True)
    return metadata
