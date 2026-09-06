"""Source-only training for the two strict HUST checkpoint families."""

from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
import time
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import open_dict
from torch.utils.data import DataLoader

import Dataset
from Lib.hust_strict_protocol import (
    apply_hust_protocol_split,
    _strict_load_staged_hust_checkpoint,
    ensure_hust_adaptation_carrier,
    hust_checkpoint_dir,
    SOURCE_LOSS_TERM_KEYS,
    sha256_file,
    strict_load_hust_checkpoint,
)
from Lib.model import get_model
from Lib.pu4d_common_source import forward_parts
from Lib.pu4d_vanilla_protocol import (
    choose_dummy_target,
    reset_and_freeze_adaptation_carrier,
)
from Lib.train_utils import seed_torch


FORMAL_SOURCE_HYPERPARAMETERS = {
    "epochs": 50,
    "batch_size": 128,
    "num_workers": 4,
    "optimizer": "AdamW",
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "label_smoothing": 0.1,
    "seed": 2025,
}

ROBUST_SOURCE_DEFAULTS = {
    "ssp_style_prob": 0.7,
    "ssp_style_strength": 0.15,
    "ssp_style_knots": 8,
    "ssp_lambda_style": 0.5,
    "sde_warp_prob": 0.7,
    "sde_warp_knots": 16,
    "sde_warp_max": 2.0,
    "sde_lambda_warp": 0.5,
    "sde_lambda_style_warp": 0.25,
    "sde_lambda_cons": 0.03,
    "sde_lambda_feat": 0.02,
}

# HUSTStrict returns per-window Z-score spectra. A smooth additive response is
# a better model of background/transfer-path variation in those standardized
# units than multiplying signed bins. Load changes also produce only modest
# shaft-speed movement. The 0W/400W manifest extrema require approximately
# 22.63/24.88 .. 24.88/22.63, so the bounded interval is rounded to 0.90..1.10.
HUST_LOAD_ROBUST_DEFAULTS = {
    "profile": "hust_load_noise_v1",
    "response_prob": 0.8,
    "response_strength": 0.08,
    "response_knots": 8,
    "noise_std": 0.015,
    "scale_prob": 0.8,
    "scale_min": 0.90,
    "scale_max": 1.10,
    "local_warp_max": 1.0,
    "local_warp_knots": 16,
}


def prepare_hust_source_config(cfg) -> int:
    """Force the shared, non-tunable formal source-training configuration."""

    only_source = getattr(cfg, "only_source", None)
    if only_source is None:
        raise ValueError("only_source is required for strict HUST source training")
    try:
        source = int(only_source)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid only_source: {only_source!r}") from exc
    selector = str(getattr(cfg, "hust_protocol_split", "bearing"))
    allowed_sources = {
        "bearing": set(range(4)),
        "load": set(range(3)),
        "bearing_load": set(range(12)),
        "bearing_6208_extrema": {9, 11},
        "bearing_6208_loads": {9, 10, 11},
    }.get(selector, set())
    if selector not in {"bearing", "load", "bearing_load", "bearing_6208_extrema", "bearing_6208_loads"}:
        raise ValueError(f"invalid hust_protocol_split: {selector!r}")
    if source not in allowed_sources:
        raise ValueError(f"only_source is outside {selector!r} domains: {source}")

    with open_dict(cfg):
        cfg.only_source = source
        cfg.seed_run = 2025
        cfg.batch_size = 128
        cfg.num_workers = 4
        cfg.src_epoch = 50
        cfg.Dataset.data_name = "HUSTStrict"
        apply_hust_protocol_split(cfg)
        cfg.Dataset.input_kind = "fft"
        cfg.Dataset.norm_kind = "mean-std"
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
        if not hasattr(cfg, "hust_checkpoint_root"):
            cfg.hust_checkpoint_root = "TTA_Model_HUST_STRICT_V2"
    return source


def _channel_first(x: torch.Tensor):
    if x.ndim == 2:
        return x.unsqueeze(1), "squeezed"
    if x.ndim == 3 and x.shape[-1] == 1:
        return x.transpose(1, 2), "transposed"
    if x.ndim == 3:
        return x, "unchanged"
    raise ValueError(f"unexpected spectral input shape: {tuple(x.shape)}")


def _restore(x: torch.Tensor, mode: str):
    if mode == "squeezed":
        return x.squeeze(1)
    if mode == "transposed":
        return x.transpose(1, 2)
    return x


def spectral_style_augment(x, strength=0.15, knots=8, prob=0.7):
    if random.random() > float(prob):
        return x
    x_cf, mode = _channel_first(x)
    batch, _channels, length = x_cf.shape
    knots = max(2, int(knots))
    ctrl = x_cf.new_empty(batch, 1, knots).uniform_(-strength, strength)
    pos = torch.linspace(-1.0, 1.0, knots, device=x.device, dtype=x.dtype).view(1, 1, -1)
    tilt = x_cf.new_empty(batch, 1, 1).uniform_(-0.5 * strength, 0.5 * strength)
    mask = torch.exp(F.interpolate(ctrl + tilt * pos, size=length, mode="linear", align_corners=True))
    return _restore(x_cf * mask, mode)


def spectral_warp_augment(x, max_warp=2.0, knots=16, prob=0.7):
    if random.random() > float(prob):
        return x
    x_cf, mode = _channel_first(x)
    batch, _channels, length = x_cf.shape
    knots = max(2, int(knots))
    ctrl = x_cf.new_empty(batch, 1, knots).uniform_(-max_warp, max_warp)
    pos = torch.linspace(-1.0, 1.0, knots, device=x.device, dtype=x.dtype).view(1, 1, -1)
    tilt = x_cf.new_empty(batch, 1, 1).uniform_(-0.25 * max_warp, 0.25 * max_warp)
    delta = F.interpolate(ctrl + tilt * pos, size=length, mode="linear", align_corners=True)
    base = torch.arange(length, device=x.device, dtype=x.dtype).view(1, 1, -1)
    sample = (base + delta).clamp(0.0, float(length - 1))
    x_norm = (2.0 * sample / float(length - 1) - 1.0).squeeze(1)
    grid = torch.stack([x_norm, torch.zeros_like(x_norm)], dim=-1).unsqueeze(1)
    warped = F.grid_sample(
        x_cf.unsqueeze(2), grid, mode="bilinear", padding_mode="border", align_corners=True
    ).squeeze(2)
    return _restore(warped, mode)


def hust_response_noise_augment(
    x, *, strength=0.08, knots=8, noise_std=0.015, prob=0.8
):
    """Perturb the smooth response/background in standardized FFT units."""

    if min(float(strength), float(noise_std)) < 0:
        raise ValueError("HUST response/noise strengths must be non-negative")
    if random.random() > float(prob):
        return x
    x_cf, mode = _channel_first(x)
    batch, _channels, length = x_cf.shape
    ctrl = x_cf.new_empty(batch, 1, max(2, int(knots))).uniform_(
        -float(strength), float(strength)
    )
    response = F.interpolate(ctrl, size=length, mode="linear", align_corners=True)
    perturbed = x_cf + response
    if noise_std:
        perturbed = perturbed + torch.randn_like(perturbed) * float(noise_std)
    return _restore(perturbed, mode)


def hust_frequency_scale_augment(
    x, *, scale_min=0.90, scale_max=1.10, prob=0.8
):
    """Apply a weak class-preserving frequency dilation for load/RPM shift."""

    if not 0 < float(scale_min) <= float(scale_max):
        raise ValueError("invalid HUST frequency-scale interval")
    if random.random() > float(prob):
        return x
    x_cf, mode = _channel_first(x)
    _batch, _channels, length = x_cf.shape
    scale = x_cf.new_empty(x_cf.shape[0], 1, 1).uniform_(scale_min, scale_max)
    output_bins = torch.arange(length, device=x.device, dtype=x.dtype).view(1, 1, -1)
    source_bins = (output_bins / scale).clamp(0.0, float(length - 1))
    x_norm = (2.0 * source_bins / float(length - 1) - 1.0).squeeze(1)
    grid = torch.stack([x_norm, torch.zeros_like(x_norm)], dim=-1).unsqueeze(1)
    scaled = F.grid_sample(
        x_cf.unsqueeze(2), grid, mode="bilinear", padding_mode="border",
        align_corners=True,
    ).squeeze(2)
    return _restore(scaled, mode)


def symmetric_kl(logits_a, logits_b, temperature=1.0):
    log_a = F.log_softmax(logits_a / temperature, dim=1)
    log_b = F.log_softmax(logits_b / temperature, dim=1)
    return 0.5 * (
        F.kl_div(log_a, log_b.exp(), reduction="batchmean")
        + F.kl_div(log_b, log_a.exp(), reduction="batchmean")
    ) * temperature**2


def feature_consistency_loss(clean_feature, augmented_feature):
    clean = F.normalize(clean_feature.detach(), dim=1)
    augmented = F.normalize(augmented_feature, dim=1)
    return (clean - augmented).pow(2).sum(dim=1).mean()


def source_loss(
    variant: Literal["ordinary", "robust"],
    clean_logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    *,
    style_logits=None,
    warp_logits=None,
    style_warp_logits=None,
    clean_feature=None,
    augmented_features=None,
):
    if variant not in {"ordinary", "robust"}:
        raise ValueError(f"invalid source variant: {variant}")
    if clean_logits.shape[1] != int(num_classes):
        raise ValueError("clean logits class count mismatch")
    clean = F.cross_entropy(clean_logits, labels, label_smoothing=0.1)
    if variant == "ordinary":
        return clean, {"clean": clean}
    if any(value is None for value in (style_logits, warp_logits, style_warp_logits, clean_feature)):
        raise ValueError("robust source loss requires all three augmented views and features")
    augmented_features = list(augmented_features or [])
    if len(augmented_features) != 3:
        raise ValueError("robust source loss requires three augmented features")
    style = F.cross_entropy(style_logits, labels, label_smoothing=0.1)
    warp = F.cross_entropy(warp_logits, labels, label_smoothing=0.1)
    style_warp = F.cross_entropy(style_warp_logits, labels, label_smoothing=0.1)
    view_logits = [style_logits, warp_logits, style_warp_logits]
    consistency = sum(symmetric_kl(clean_logits, view) for view in view_logits) / 3.0
    feature = sum(
        feature_consistency_loss(clean_feature, augmented)
        for augmented in augmented_features
    ) / 3.0
    terms = {
        "clean": clean,
        "style": style,
        "warp": warp,
        "style_warp": style_warp,
        "symmetric_kl": consistency,
        "feature": feature,
    }
    total = (
        clean
        + ROBUST_SOURCE_DEFAULTS["ssp_lambda_style"] * style
        + ROBUST_SOURCE_DEFAULTS["sde_lambda_warp"] * warp
        + ROBUST_SOURCE_DEFAULTS["sde_lambda_style_warp"] * style_warp
        + ROBUST_SOURCE_DEFAULTS["sde_lambda_cons"] * consistency
        + ROBUST_SOURCE_DEFAULTS["sde_lambda_feat"] * feature
    )
    return total, terms


def _carrier_is_identity(model: nn.Module, carrier_names: list[str]) -> bool:
    named = dict(model.named_parameters())
    return bool(carrier_names) and all(
        name in named
        and not named[name].requires_grad
        and int(torch.count_nonzero(named[name].detach())) == 0
        for name in carrier_names
    )


def build_source_optimizer(model: nn.Module):
    """Return the formal optimizer and its explicit non-carrier parameter names."""

    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not named_parameters:
        raise RuntimeError("HUST source model has no trainable non-carrier parameters")
    carrier_tokens = ("band_scale", "band_bias", "warp_ctrl")
    unexpected = [
        name
        for name, _parameter in named_parameters
        if any(token in name for token in carrier_tokens)
    ]
    if unexpected:
        raise RuntimeError(f"adaptation carrier is trainable in source optimizer: {unexpected}")
    optimizer = torch.optim.AdamW(
        [parameter for _name, parameter in named_parameters],
        lr=0.001,
        weight_decay=0.0001,
    )
    return optimizer, [name for name, _parameter in named_parameters]


def _fsync_file(path: Path) -> None:
    with Path(path).open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_fsync(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + os.linesep,
        encoding="utf-8",
    )
    _fsync_file(path)


def _emit_source_result(record: dict) -> None:
    encoded = json.dumps(
        record, allow_nan=False, separators=(",", ":"), sort_keys=True
    )
    print(f"HUST_SOURCE_RESULT_JSON={encoded}", flush=True)


def _publish_checkpoint_contract(
    model: nn.Module,
    directory: Path,
    checkpoint_name: str,
    payload: dict,
    metadata: dict,
) -> dict:
    """Validate a complete sibling directory before one atomic promotion."""

    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{directory.name}.tmp-", dir=directory.parent)
    )
    try:
        checkpoint = temporary / checkpoint_name
        torch.save(payload, checkpoint)
        _fsync_file(checkpoint)
        metadata["checkpoint_sha256"] = sha256_file(checkpoint)
        _write_json_fsync(temporary / "source_training_summary.json", metadata)
        _fsync_directory(temporary)
        _strict_load_staged_hust_checkpoint(
            model,
            checkpoint,
            temporary,
        )
        os.replace(temporary, directory)
        _fsync_directory(directory.parent)
        return metadata
    except BaseException:
        if temporary.parent == directory.parent and temporary.name.startswith(
            f".{directory.name}.tmp-"
        ):
            shutil.rmtree(temporary, ignore_errors=True)
        raise


def _source_dataset(cfg, source: int, dummy_target: int):
    dataset_cls = getattr(Dataset, str(cfg.Dataset.data_name))
    if not hasattr(dataset_cls, "_load"):
        raise RuntimeError("HUSTStrict must expose source-only _load(domain)")
    # HUSTStrict.__init__ validates every cached domain. Source training must not
    # even read target label tensors, so construct the narrow loader state directly.
    dataset = dataset_cls.__new__(dataset_cls)
    dataset.root = Path(str(cfg.Dataset.data_path))
    dataset.source = int(source)
    dataset.target = int(dummy_target)
    return dataset_cls, dataset._load(source)


def _existing_contract(model, checkpoint: Path, expected: dict):
    summary = checkpoint.parent / "source_training_summary.json"
    if not checkpoint.exists() and not summary.exists():
        if checkpoint.parent.exists():
            raise FileExistsError(
                f"HUST checkpoint destination contains unrelated data: {checkpoint.parent}"
            )
        return None
    if not checkpoint.exists() or not summary.exists():
        raise FileExistsError(
            f"pre-existing HUST checkpoint destination is partial: {checkpoint.parent}"
        )
    try:
        metadata = strict_load_hust_checkpoint(model, checkpoint)
    except ValueError as exc:
        raise FileExistsError(
            f"pre-existing HUST checkpoint destination is invalid: {checkpoint.parent}"
        ) from exc
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise FileExistsError(f"existing HUST checkpoint differs in {key}")
    return metadata


def train_hust_source(cfg, source: int, variant: Literal["ordinary", "robust"]) -> dict:
    if variant not in {"ordinary", "robust"}:
        raise ValueError(f"invalid source variant: {variant}")
    domains = [int(value) for value in cfg.Dataset.TL_list]
    source = int(source)
    if source not in domains:
        raise ValueError(f"source {source} is outside HUST domains {domains}")
    seed = 2025
    root = Path(str(getattr(cfg, "hust_checkpoint_root", "TTA_Model_HUST_STRICT_V2")))
    smoke_mode = bool(getattr(cfg, "smoke_mode", False))
    if smoke_mode and not root.name.endswith("_SMOKE"):
        raise ValueError("smoke_mode requires a checkpoint root ending in _SMOKE")
    epochs_run = int(getattr(cfg, "smoke_epochs", 1)) if smoke_mode else 50
    if epochs_run < 1:
        raise ValueError("training epochs must be positive")
    dummy_target = choose_dummy_target(source, domains)
    robust_profile = str(getattr(cfg, "hust_robust_profile", "generic_v1"))
    if variant == "ordinary":
        robust_profile = "not_applicable"
    elif robust_profile not in {"generic_v1", HUST_LOAD_ROBUST_DEFAULTS["profile"]}:
        raise ValueError(f"unknown HUST robust profile: {robust_profile}")
    robust_augmentation = (
        dict(HUST_LOAD_ROBUST_DEFAULTS)
        if robust_profile == HUST_LOAD_ROBUST_DEFAULTS["profile"]
        else (dict(ROBUST_SOURCE_DEFAULTS) if robust_profile == "generic_v1" else None)
    )

    if smoke_mode:
        cache_identity = {
            "mode": "smoke",
            "manifest_path": None,
            "manifest_sha256": None,
            "content_sha256": hashlib.sha256(
                f"smoke:{cfg.Dataset.data_path}".encode()
            ).hexdigest(),
            "tensor_sha256s": {},
        }
    else:
        from Lib.hust_strict_protocol import validate_cache

        cache = validate_cache(Path(str(cfg.Dataset.data_path)))
        if cache.get("version") != 2 or not cache.get("content_sha256"):
            raise ValueError("formal source training requires a version-2 content-bound cache")
        cache_root = Path(str(cfg.Dataset.data_path))
        cache_identity = {
            "mode": "formal",
            "manifest_path": str((cache_root / "manifest.json").resolve()),
            "manifest_sha256": str(cache["manifest_sha256"]),
            "content_sha256": str(cache["content_sha256"]),
            "tensor_sha256s": {
                str((cache_root / str(summary["tensor_file"])).resolve()): str(summary["tensor_sha256"])
                for summary in cache["domains"].values()
            },
        }

    seed_torch(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_cls, source_data = _source_dataset(cfg, source, dummy_target)
    if len(source_data) == 0:
        raise RuntimeError("empty source loader")
    num_classes = int(dataset_cls.num_classes)
    loader_workers = int(getattr(cfg, "num_workers", 4))
    if not smoke_mode and loader_workers != 4:
        raise ValueError("formal HUST source training requires num_workers=4")
    loader = DataLoader(
        source_data,
        batch_size=128,
        shuffle=True,
        num_workers=loader_workers,
        pin_memory=False,
        drop_last=False,
    )
    model = get_model(num_classes=num_classes, cfg=cfg, **cfg.Model).to(device)
    carrier_names = ensure_hust_adaptation_carrier(
        model, ROBUST_SOURCE_DEFAULTS["sde_warp_knots"]
    )
    reset_and_freeze_adaptation_carrier(model)
    if not _carrier_is_identity(model, carrier_names):
        raise RuntimeError("source adaptation carrier did not reset to identity")

    directory = hust_checkpoint_dir(root, variant, source, seed)
    model_name = str(getattr(cfg, "model_name", f"ResNet18_1D_SDE{seed}fft_Linear.pt"))
    checkpoint = directory / model_name
    expected = {
        "route": variant,
        "source": source,
        "seed": seed,
        "epochs": 50,
        "hyperparameters": FORMAL_SOURCE_HYPERPARAMETERS,
        "cache_identity": cache_identity,
        "robust_profile": robust_profile,
        "robust_augmentation": robust_augmentation,
    }
    existing = _existing_contract(model, checkpoint, expected)
    if existing is not None:
        _emit_source_result(existing["source_result"])
        return existing

    optimizer, _optimized_names = build_source_optimizer(model)
    started = time.monotonic()
    optimizer_steps = 0
    correct = 0
    seen = 0
    loss_sample_count = 0
    loss_term_totals = {name: 0.0 for name in SOURCE_LOSS_TERM_KEYS[variant]}
    for _epoch in range(epochs_run):
        model.train()
        correct = 0
        seen = 0
        for x, labels, _indices in loader:
            x, labels = x.to(device), labels.to(device)
            clean_feature, clean_logits = forward_parts(model, x)
            if variant == "ordinary":
                loss, terms = source_loss(variant, clean_logits, labels, num_classes)
            else:
                if robust_profile == HUST_LOAD_ROBUST_DEFAULTS["profile"]:
                    profile = HUST_LOAD_ROBUST_DEFAULTS
                    style_x = hust_response_noise_augment(
                        x, strength=profile["response_strength"],
                        knots=profile["response_knots"], noise_std=profile["noise_std"],
                        prob=profile["response_prob"],
                    )
                    warp_x = hust_frequency_scale_augment(
                        x, scale_min=profile["scale_min"], scale_max=profile["scale_max"],
                        prob=profile["scale_prob"],
                    )
                    warp_x = spectral_warp_augment(
                        warp_x, max_warp=profile["local_warp_max"],
                        knots=profile["local_warp_knots"], prob=profile["scale_prob"],
                    )
                    style_warp_x = hust_response_noise_augment(
                        warp_x, strength=profile["response_strength"],
                        knots=profile["response_knots"], noise_std=profile["noise_std"],
                        prob=profile["response_prob"],
                    )
                else:
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
                style_feature, style_logits = forward_parts(model, style_x)
                warp_feature, warp_logits = forward_parts(model, warp_x)
                style_warp_feature, style_warp_logits = forward_parts(model, style_warp_x)
                loss, terms = source_loss(
                    variant,
                    clean_logits,
                    labels,
                    num_classes,
                    style_logits=style_logits,
                    warp_logits=warp_logits,
                    style_warp_logits=style_warp_logits,
                    clean_feature=clean_feature,
                    augmented_features=[style_feature, warp_feature, style_warp_feature],
                )
            if set(terms) != SOURCE_LOSS_TERM_KEYS[variant]:
                raise RuntimeError(f"invalid {variant} source loss terms: {sorted(terms)}")
            batch_count = int(labels.numel())
            for name, term in terms.items():
                if term.numel() != 1:
                    raise RuntimeError(f"source loss term is not scalar: {name}")
                value = float(term.detach().cpu().item())
                if not math.isfinite(value):
                    raise RuntimeError(f"non-finite source loss term: {name}")
                loss_term_totals[name] += value * batch_count
            if not bool(torch.isfinite(loss.detach()).all()):
                raise RuntimeError("non-finite total source loss")
            loss_sample_count += batch_count
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            correct += int((clean_logits.argmax(1) == labels).sum())
            seen += int(labels.numel())

    elapsed = time.monotonic() - started
    if optimizer_steps == 0 or loss_sample_count == 0:
        raise RuntimeError("empty source loader")
    if not _carrier_is_identity(model, carrier_names):
        raise RuntimeError("source training modified the target adaptation carrier")
    aggregated_terms = {
        name: total / loss_sample_count for name, total in loss_term_totals.items()
    }
    if not all(math.isfinite(value) for value in aggregated_terms.values()):
        raise RuntimeError("non-finite aggregated source loss")
    source_result = {
        "route": variant,
        "source": source,
        "seed": seed,
        "epochs": epochs_run,
        "optimizer_steps": optimizer_steps,
        "target_labels_consumed": False,
        "carrier_identity": True,
        "loss_terms": aggregated_terms,
    }
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    metadata = {
        "contract_version": 1,
        **expected,
        "target_labels_consumed": False,
        "carrier_identity_check": True,
        "carrier_parameters": carrier_names,
        "tensor_count": len(state),
        "source_accuracy": 100.0 * correct / max(seen, 1),
        "elapsed_seconds": elapsed,
        "optimizer_steps": optimizer_steps,
        "execution": {"smoke_mode": smoke_mode, "epochs_run": epochs_run},
        "source_result": source_result,
        "checkpoint": checkpoint.name,
    }
    payload = {
        "contract": {
            "version": 1,
            "route": variant,
            "source": source,
            "seed": seed,
            "epoch": 50,
        },
        "metadata": metadata,
        "state_dict": state,
    }
    published = _publish_checkpoint_contract(
        model=model,
        directory=directory,
        checkpoint_name=checkpoint.name,
        payload=payload,
        metadata=metadata,
    )
    _emit_source_result(published["source_result"])
    return published
