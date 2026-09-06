"""Versioned WTPG speed-domain cache and checkpoint contracts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

import torch
import torch.nn as nn


SPEEDS = {0: 20, 1: 25, 2: 30, 3: 35, 4: 40, 5: 45, 6: 50, 7: 55}
SPEED_TO_DOMAIN = {speed: domain for domain, speed in SPEEDS.items()}
LABEL_MAP = {
    "Healthy": 0,
    "Broken tooth": 1,
    "Wear gear": 2,
    "Gear root crack": 3,
    "Missing tooth": 4,
}
CLASS_PREFIX = {0: "N", 1: "B", 2: "W", 3: "R", 4: "M"}
PROCESSING = {
    "sampling_rate_hz": 48_000,
    "window_size": 2_048,
    "window_stride": 2_048,
    "fft_kind": "rfft_log1p_without_dc",
    "spectrum_length": 512,
    "normalization": "per_sample_zscore",
    "channel_idx": 0,
    "samples_per_recording": 300,
    "samples_per_class": 600,
}
CACHE_KEYS = {
    "data", "label", "speed_hz", "recording_id", "window_offset",
}
FILENAME_RE = re.compile(r"^([NBWRM])([12])_(20|25|30|35|40|45|50|55)\.MAT$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_dir(root: Path, variant: str, source: int, seed: int) -> Path:
    if variant not in {"ordinary", "robust"}:
        raise ValueError(f"invalid WTPG source variant: {variant}")
    if int(source) not in SPEEDS:
        raise ValueError(f"invalid WTPG source domain: {source}")
    return Path(root) / variant / f"source_{int(source)}" / f"seed_{int(seed)}"


def resolve_checkpoint(root: Path, variant: str, source: int, seed: int, model_name: str) -> Path:
    directory = checkpoint_dir(root, variant, source, seed)
    path = directory / str(model_name)
    if not path.is_file():
        raise FileNotFoundError(f"WTPG source checkpoint not found: {path}")
    return path


def strict_load_checkpoint(model: nn.Module, path: Path) -> dict:
    """Verify cache, route, hashes and every tensor before loading a source model."""
    path = Path(path)
    summary_path = path.parent / "source_training_summary.json"
    try:
        metadata = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid WTPG checkpoint summary: {summary_path}") from exc
    required = {
        "contract_version", "dataset", "route", "source", "speed_hz", "seed",
        "epochs", "hyperparameters", "cache_identity", "target_labels_consumed",
        "carrier_identity_check", "carrier_parameters", "tensor_count",
        "source_accuracy", "optimizer_steps", "elapsed_seconds", "checkpoint",
        "checkpoint_sha256", "loss_terms",
    }
    optional = {"robust_profile", "robust_augmentation"}
    if (
        not isinstance(metadata, dict)
        or not required.issubset(metadata)
        or set(metadata) - required - optional
    ):
        raise ValueError("invalid WTPG checkpoint metadata fields")
    if "robust_profile" in metadata:
        profile = metadata["robust_profile"]
        augmentation = metadata.get("robust_augmentation")
        route = metadata.get("route")
        valid_ordinary_marker = (
            route == "ordinary" and profile == "not_applicable" and augmentation is None
        )
        valid_robust_marker = route == "robust" and isinstance(profile, str)
        if not (valid_ordinary_marker or valid_robust_marker):
            raise ValueError("invalid WTPG robust-source profile metadata")
        if profile == "wtpg_speed_noise_v1" and not isinstance(augmentation, dict):
            raise ValueError("missing WTPG robust augmentation metadata")
    if metadata["contract_version"] != 1 or metadata["dataset"] != "WTPGStrict":
        raise ValueError("unsupported WTPG checkpoint contract")
    variant = str(metadata["route"])
    source = int(metadata["source"])
    seed = int(metadata["seed"])
    expected_dir = checkpoint_dir(path.parent.parent.parent.parent, variant, source, seed)
    if path.parent != expected_dir or metadata["speed_hz"] != SPEEDS[source]:
        raise ValueError("WTPG checkpoint route/domain mismatch")
    if seed != 2025 or metadata["epochs"] != 50 or metadata["target_labels_consumed"] is not False:
        raise ValueError("invalid WTPG source-training protocol")
    if metadata["checkpoint"] != path.name or sha256_file(path) != metadata["checkpoint_sha256"]:
        raise ValueError("WTPG checkpoint hash/name mismatch")
    identity = metadata["cache_identity"]
    cache = validate_cache(Path(identity["manifest_path"]).parent)
    if identity != {
        "manifest_path": str((Path(identity["manifest_path"]).parent / "manifest.json").resolve()),
        "manifest_sha256": cache["manifest_sha256"],
        "content_sha256": cache["content_sha256"],
    }:
        raise ValueError("WTPG checkpoint/cache identity mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"contract", "state_dict"}:
        raise ValueError("invalid WTPG checkpoint payload")
    if payload["contract"] != {
        "version": 1, "dataset": "WTPGStrict", "route": variant,
        "source": source, "seed": seed, "epoch": 50,
    }:
        raise ValueError("WTPG embedded checkpoint contract mismatch")
    state = payload["state_dict"]
    if not isinstance(state, dict) or len(state) != metadata["tensor_count"]:
        raise ValueError("invalid WTPG checkpoint state")
    backbone = model[0] if isinstance(model, (nn.Sequential, nn.ModuleList)) else model
    if not hasattr(backbone, "warp_ctrl"):
        device = next(backbone.parameters()).device
        backbone.register_parameter("warp_ctrl", nn.Parameter(torch.zeros(1, 1, 16, device=device)))
    actual_carriers = [
        name for name, _ in model.named_parameters()
        if any(token in name for token in ("band_scale", "band_bias", "warp_ctrl"))
    ]
    if actual_carriers != metadata["carrier_parameters"] or not metadata["carrier_identity_check"]:
        raise ValueError("WTPG adaptation-carrier contract mismatch")
    if any(name not in state or bool(torch.count_nonzero(state[name])) for name in actual_carriers):
        raise ValueError("WTPG checkpoint carrier is not identity")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError("WTPG checkpoint does not match model") from exc
    return metadata


def validate_cache(root: Path) -> dict:
    root = Path(root)
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid WTPG cache manifest: {manifest_path}") from exc
    if manifest.get("version") != 1:
        raise ValueError("unsupported WTPG cache version")
    if manifest.get("split") != "motor_speed" or manifest.get("label_map") != LABEL_MAP:
        raise ValueError("invalid WTPG domain/label contract")
    if manifest.get("processing") != PROCESSING or manifest.get("seed") != 2025:
        raise ValueError("invalid WTPG preprocessing contract")
    expected_map = {str(key): f"{value}Hz" for key, value in SPEEDS.items()}
    if manifest.get("domain_map") != expected_map:
        raise ValueError("invalid WTPG speed-domain map")
    source_files = manifest.get("source_files")
    if not isinstance(source_files, list) or len(source_files) != 80:
        raise ValueError("WTPG cache must bind all 80 source recordings")
    source_paths = [row.get("path") for row in source_files]
    if source_paths != sorted(source_paths) or len(set(source_paths)) != 80:
        raise ValueError("invalid WTPG source-file inventory")
    for recording_id, row in enumerate(source_files):
        required = {
            "recording_id", "path", "sha256", "label", "class_name", "replicate",
            "speed_hz", "sample_rate_hz", "data_shape", "window_count",
        }
        if not isinstance(row, dict) or set(row) != required:
            raise ValueError("invalid WTPG source-file metadata")
        if row["recording_id"] != recording_id:
            raise ValueError("invalid WTPG recording id")
        if re.fullmatch(r"[0-9a-f]{64}", str(row["sha256"])) is None:
            raise ValueError("invalid WTPG source hash")
        if row["label"] not in LABEL_MAP.values() or row["speed_hz"] not in SPEED_TO_DOMAIN:
            raise ValueError("invalid WTPG source class/speed")
        if row["sample_rate_hz"] != 48_000 or row["window_count"] < 300:
            raise ValueError("invalid WTPG recording sampling metadata")
    domains = manifest.get("domains")
    if not isinstance(domains, dict) or set(domains) != set(expected_map):
        raise ValueError("invalid WTPG domain summaries")
    identity = {}
    for domain_text, speed_text in expected_map.items():
        summary = domains[domain_text]
        tensor_name = f"domain_{domain_text}.pt"
        tensor_path = root / tensor_name
        if summary.get("tensor_file") != tensor_name or not tensor_path.is_file():
            raise ValueError(f"missing WTPG domain tensor: {tensor_path}")
        actual_hash = sha256_file(tensor_path)
        if summary.get("tensor_sha256") != actual_hash:
            raise ValueError(f"WTPG tensor hash mismatch: {tensor_path}")
        obj = torch.load(tensor_path, map_location="cpu", weights_only=True)
        if not isinstance(obj, dict) or set(obj) != CACHE_KEYS:
            raise ValueError("invalid WTPG cache tensor keys")
        n = int(obj["label"].numel())
        if tuple(obj["data"].shape) != (n, 1, 512) or n != 3_000:
            raise ValueError("invalid WTPG domain tensor shape")
        if any(tuple(obj[key].shape) != (n,) for key in CACHE_KEYS - {"data"}):
            raise ValueError("invalid WTPG metadata tensor shape")
        if obj["data"].dtype != torch.float32 or not bool(torch.isfinite(obj["data"]).all()):
            raise ValueError("invalid WTPG spectral tensor")
        if any(obj[key].dtype != torch.int64 for key in CACHE_KEYS - {"data"}):
            raise ValueError("invalid WTPG metadata dtype")
        speed = SPEEDS[int(domain_text)]
        if not bool((obj["speed_hz"] == speed).all()):
            raise ValueError("WTPG sample assigned to wrong speed domain")
        counts = {str(label): int((obj["label"] == label).sum()) for label in range(5)}
        if counts != {str(label): 600 for label in range(5)}:
            raise ValueError("WTPG domain is not class balanced")
        if summary.get("shape") != [3_000, 1, 512] or summary.get("class_counts") != counts:
            raise ValueError("WTPG manifest/tensor mismatch")
        recording_ids = obj["recording_id"]
        offsets = obj["window_offset"]
        if not bool(((recording_ids >= 0) & (recording_ids < 80)).all()):
            raise ValueError("WTPG recording id out of bounds")
        if not bool(((offsets >= 0) & (offsets % 2_048 == 0)).all()):
            raise ValueError("invalid WTPG window offset")
        ids = recording_ids.tolist()
        if any(source_files[index]["speed_hz"] != speed for index in ids):
            raise ValueError("WTPG recording crosses speed domains")
        expected_labels = torch.tensor([source_files[index]["label"] for index in ids])
        if not torch.equal(obj["label"], expected_labels):
            raise ValueError("WTPG sample label/recording mismatch")
        if len(set(zip(ids, offsets.tolist()))) != n:
            raise ValueError("duplicate WTPG recording window")
        identity[domain_text] = {"tensor_file": tensor_name, "tensor_sha256": actual_hash}
    expected_content = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if manifest.get("content_sha256") != expected_content:
        raise ValueError("WTPG cache content identity mismatch")
    if not isinstance(manifest.get("created_at"), str) or not math.isfinite(float(manifest.get("build_seconds", -1))):
        raise ValueError("invalid WTPG cache build metadata")
    return {**manifest, "manifest_sha256": sha256_file(manifest_path)}
