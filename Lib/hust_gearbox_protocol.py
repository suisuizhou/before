"""Contracts for the HUST gearbox speed-domain benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


SPEEDS = {0: 20, 1: 25, 2: 30, 3: 35, 4: 40, 5: -1}
SPEED_NAMES = {**{k: f"{v}Hz" for k, v in SPEEDS.items() if v > 0}, 5: "0-40-0Hz"}
LABEL_MAP = {"H": 0, "B": 1, "M": 2}
PROCESSING = {
    "sampling_rate_hz": 25_600,
    "window_size": 2_048,
    "window_stride": 2_048,
    "spectrum_length": 512,
    "normalization": "rfft_log1p_without_dc_per_sample_zscore",
    "signal": "three_axis_acceleration_magnitude",
    "samples_per_recording": 128,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_cache(root: Path) -> dict:
    root = Path(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != 1 or manifest.get("dataset") != "HUSTGearboxStrict":
        raise ValueError("invalid HUST gearbox cache contract")
    if manifest.get("domain_map") != {str(k): v for k, v in SPEED_NAMES.items()}:
        raise ValueError("invalid HUST gearbox domain map")
    if manifest.get("label_map") != LABEL_MAP or manifest.get("processing") != PROCESSING:
        raise ValueError("invalid HUST gearbox label/processing contract")
    for domain in SPEEDS:
        path = root / f"domain_{domain}.pt"
        obj = torch.load(path, map_location="cpu", weights_only=True)
        n = int(obj["label"].numel())
        if tuple(obj["data"].shape) != (n, 1, 512) or n != 1_920:
            raise ValueError(f"invalid HUST gearbox domain {domain} shape")
        counts = [int((obj["label"] == label).sum()) for label in range(3)]
        if counts != [640, 640, 640] or not bool(torch.isfinite(obj["data"]).all()):
            raise ValueError(f"invalid HUST gearbox domain {domain} balance/data")
    return {**manifest, "manifest_sha256": sha256_file(manifest_path)}

