"""Dataset-specific 0711 robust source training for balanced CWRU."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
import time

import torch
from torch.utils.data import DataLoader

from Dataset.CWRU import CWRUCacheDataset
from Lib.hust_source_training import (
    _carrier_is_identity,
    build_source_optimizer,
    hust_frequency_scale_augment,
    hust_response_noise_augment,
    source_loss,
    spectral_warp_augment,
)
from Lib.hust_strict_protocol import ensure_hust_adaptation_carrier
from Lib.model import get_model
from Lib.pu4d_common_source import forward_parts
from Lib.pu4d_vanilla_protocol import reset_and_freeze_adaptation_carrier
from Lib.train_utils import seed_torch


CWRU_ROBUST_DEFAULTS = {
    "profile": "cwru_load_noise_v1",
    "response_prob": 0.8,
    "response_strength": 0.08,
    "response_knots": 8,
    "noise_std": 0.015,
    "scale_prob": 0.8,
    "scale_min": 0.96,
    "scale_max": 1.04,
    "local_warp_max": 1.0,
    "local_warp_knots": 16,
}

CWRU_SOURCE_HYPERPARAMETERS = {
    "epochs": 50,
    "batch_size": 128,
    "num_workers": 4,
    "optimizer": "AdamW",
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "label_smoothing": 0.1,
    "seed": 2025,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _robust_views(x: torch.Tensor):
    profile = CWRU_ROBUST_DEFAULTS
    style = hust_response_noise_augment(
        x,
        strength=profile["response_strength"],
        knots=profile["response_knots"],
        noise_std=profile["noise_std"],
        prob=profile["response_prob"],
    )
    warp = hust_frequency_scale_augment(
        x,
        scale_min=profile["scale_min"],
        scale_max=profile["scale_max"],
        prob=profile["scale_prob"],
    )
    warp = spectral_warp_augment(
        warp,
        max_warp=profile["local_warp_max"],
        knots=profile["local_warp_knots"],
        prob=profile["scale_prob"],
    )
    combined = hust_response_noise_augment(
        warp,
        strength=profile["response_strength"],
        knots=profile["response_knots"],
        noise_std=profile["noise_std"],
        prob=profile["response_prob"],
    )
    return style, warp, combined


def train_cwru_robust_source(cfg, source: int) -> dict:
    source = int(source)
    if source not in {0, 1, 3}:
        raise ValueError("redesigned CWRU sources are 0, 1, or 3")
    seed_torch(2025)
    random.seed(2025)
    cache_root = Path(str(cfg.Dataset.data_path))
    cache_file = cache_root / f"domain_{source}.pt"
    metadata_file = cache_root / "metadata.json"
    if not cache_file.is_file() or not metadata_file.is_file():
        raise FileNotFoundError(f"incomplete CWRU cache: {cache_root}")
    dataset = CWRUCacheDataset(cache_file)
    loader = DataLoader(
        dataset, batch_size=128, shuffle=True, num_workers=4, drop_last=False
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(num_classes=10, cfg=cfg, **cfg.Model).to(device)
    carrier_names = ensure_hust_adaptation_carrier(model, 16)
    reset_and_freeze_adaptation_carrier(model)
    if not _carrier_is_identity(model, carrier_names):
        raise RuntimeError("CWRU adaptation carrier is not identity")

    root = Path(str(getattr(cfg, "cwru_checkpoint_root", "TTA_Model_CWRU_REDESIGN_V1")))
    directory = root / "robust" / f"source_{source}" / "seed_2025"
    model_name = str(cfg.model_name)
    checkpoint = directory / model_name
    summary = directory / "source_training_summary.json"
    if checkpoint.exists() and summary.exists():
        record = json.loads(summary.read_text(encoding="utf-8"))
        if _sha256(checkpoint) != record["checkpoint_sha256"]:
            raise ValueError("CWRU robust checkpoint hash mismatch")
        print("CWRU_SOURCE_RESULT_JSON=" + json.dumps(record, sort_keys=True), flush=True)
        return record
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite partial checkpoint: {directory}")

    optimizer, _optimized_names = build_source_optimizer(model)
    totals = {
        name: 0.0
        for name in {"clean", "style", "warp", "style_warp", "symmetric_kl", "feature"}
    }
    samples = optimizer_steps = 0
    started = time.monotonic()
    correct = seen = 0
    for epoch in range(50):
        model.train()
        correct = seen = 0
        for x, labels, _indices in loader:
            x, labels = x.to(device), labels.to(device)
            clean_feature, clean_logits = forward_parts(model, x)
            style_x, warp_x, combined_x = _robust_views(x)
            style_feature, style_logits = forward_parts(model, style_x)
            warp_feature, warp_logits = forward_parts(model, warp_x)
            combined_feature, combined_logits = forward_parts(model, combined_x)
            loss, terms = source_loss(
                "robust",
                clean_logits,
                labels,
                10,
                style_logits=style_logits,
                warp_logits=warp_logits,
                style_warp_logits=combined_logits,
                clean_feature=clean_feature,
                augmented_features=[style_feature, warp_feature, combined_feature],
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite CWRU robust source loss")
            batch = int(labels.numel())
            for name, term in terms.items():
                value = float(term.detach())
                if not math.isfinite(value):
                    raise RuntimeError(f"non-finite CWRU source term: {name}")
                totals[name] += value * batch
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            samples += batch
            correct += int((clean_logits.argmax(1) == labels).sum())
            seen += batch
        print(
            f"CWRU robust source={source} epoch={epoch + 1}/50 "
            f"source_acc={100.0 * correct / max(seen, 1):.4f}%",
            flush=True,
        )

    if not _carrier_is_identity(model, carrier_names):
        raise RuntimeError("CWRU robust source training modified adaptation carrier")
    record = {
        "contract_version": 1,
        "dataset": "CWRU",
        "route": "robust",
        "source": source,
        "seed": 2025,
        "hyperparameters": CWRU_SOURCE_HYPERPARAMETERS,
        "robust_profile": CWRU_ROBUST_DEFAULTS["profile"],
        "robust_augmentation": CWRU_ROBUST_DEFAULTS,
        "cache_identity": {
            "metadata_sha256": _sha256(metadata_file),
            "source_tensor_sha256": _sha256(cache_file),
        },
        "target_labels_consumed": False,
        "carrier_identity_check": True,
        "carrier_parameters": carrier_names,
        "source_accuracy": 100.0 * correct / max(seen, 1),
        "optimizer_steps": optimizer_steps,
        "elapsed_seconds": time.monotonic() - started,
        "loss_terms": {name: value / samples for name, value in sorted(totals.items())},
        "checkpoint": model_name,
        "checkpoint_sha256": "pending",
    }
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}.tmp-", dir=directory.parent))
    try:
        staged = temporary / model_name
        torch.save(
            {
                "contract": {"dataset": "CWRU", "route": "robust", "source": source, "seed": 2025},
                "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            },
            staged,
        )
        record["checkpoint_sha256"] = _sha256(staged)
        (temporary / "source_training_summary.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + os.linesep,
            encoding="utf-8",
        )
        os.replace(temporary, directory)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print("CWRU_SOURCE_RESULT_JSON=" + json.dumps(record, sort_keys=True), flush=True)
    return record
