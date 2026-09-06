#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Balanced cached CWRU 12-kHz drive-end dataset for cross-domain TTA.

The raw 40 MAT files are first converted with ``build_cwru_cache.py`` into
four balanced domain caches.  Each cache sample is already a 512-bin FFT
amplitude spectrum with per-window Z-score normalization and shape [1, 512].
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Sequence
import json

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset


CLASS_NAMES = (
    "Normal",
    "IR007", "B007", "OR007_6",
    "IR014", "B014", "OR014_6",
    "IR021", "B021", "OR021_6",
)
DOMAIN_RPM: Dict[int, float] = {0: 1797.0, 1: 1772.0, 2: 1750.0, 3: 1730.0}

# label order is fixed across all four domains.
CWRU_FILE_MAP: Dict[int, Dict[int, str]] = {
    0: {0:"97.mat", 1:"105.mat", 2:"118.mat", 3:"130.mat", 4:"169.mat", 5:"185.mat", 6:"197.mat", 7:"209.mat", 8:"222.mat", 9:"234.mat"},
    1: {0:"98.mat", 1:"106.mat", 2:"119.mat", 3:"131.mat", 4:"170.mat", 5:"186.mat", 6:"198.mat", 7:"210.mat", 8:"223.mat", 9:"235.mat"},
    2: {0:"99.mat", 1:"107.mat", 2:"120.mat", 3:"132.mat", 4:"171.mat", 5:"187.mat", 6:"199.mat", 7:"211.mat", 8:"224.mat", 9:"236.mat"},
    3: {0:"100.mat", 1:"108.mat", 2:"121.mat", 3:"133.mat", 4:"172.mat", 5:"188.mat", 6:"200.mat", 7:"212.mat", 8:"225.mat", 9:"237.mat"},
}


def common_window_count(lengths: Iterable[int], window_size: int = 1024) -> int:
    counts = [int(v) // int(window_size) for v in lengths]
    if not counts:
        raise ValueError("lengths must not be empty")
    result = min(counts)
    if result < 1:
        raise ValueError(f"At least one complete window is required, counts={counts}")
    return int(result)


def fft_window(window: np.ndarray, fft_size: int = 1024, spectrum_length: int = 512) -> np.ndarray:
    x = np.asarray(window, dtype=np.float64).reshape(-1)
    if x.size != int(fft_size):
        raise ValueError(f"Expected {fft_size} samples, got {x.size}")
    spectrum = np.abs(np.fft.fft(x, n=int(fft_size))) / float(fft_size)
    spectrum = spectrum[: int(spectrum_length)].astype(np.float32, copy=False)
    mean = float(spectrum.mean())
    std = float(spectrum.std())
    spectrum = (spectrum - mean) / max(std, 1e-8)
    return spectrum.astype(np.float32, copy=False)


def _find_signal_key(mat: dict, suffix: str) -> str:
    keys = sorted(k for k in mat.keys() if not k.startswith("__") and k.endswith(suffix))
    if not keys:
        raise KeyError(f"No MAT variable ending with {suffix!r}; available={sorted(k for k in mat if not k.startswith('__'))}")
    if len(keys) > 1:
        # Prefer the standard X###_DE_time / X###RPM style key if present.
        exactish = [k for k in keys if "DE" in k] if suffix == "_DE_time" else keys
        keys = exactish or keys
    return keys[0]


def load_de_signal(mat_path: Path | str) -> tuple[np.ndarray, float | None]:
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError("scipy is required to build the CWRU cache") from exc
    path = Path(mat_path)
    mat = loadmat(path)
    signal_key = _find_signal_key(mat, "_DE_time")
    signal = np.asarray(mat[signal_key]).reshape(-1).astype(np.float64, copy=False)
    rpm_keys = [k for k in mat if not k.startswith("__") and k.upper().endswith("RPM")]
    rpm = None
    if rpm_keys:
        arr = np.asarray(mat[sorted(rpm_keys)[0]]).reshape(-1)
        if arr.size:
            rpm = float(arr[0])
    return signal, rpm


def build_balanced_cache(
    raw_dir: Path | str,
    cache_dir: Path | str,
    window_size: int = 1024,
    spectrum_length: int = 512,
) -> dict:
    raw = Path(raw_dir)
    out = Path(cache_dir)
    out.mkdir(parents=True, exist_ok=True)

    missing = [raw / CWRU_FILE_MAP[d][k] for d in range(4) for k in range(10) if not (raw / CWRU_FILE_MAP[d][k]).is_file()]
    if missing:
        raise FileNotFoundError("Missing CWRU MAT files: " + ", ".join(str(p) for p in missing[:10]))

    loaded: dict[tuple[int,int], tuple[np.ndarray,float|None]] = {}
    lengths = []
    for domain in range(4):
        for label in range(10):
            signal, rpm = load_de_signal(raw / CWRU_FILE_MAP[domain][label])
            loaded[(domain, label)] = (signal, rpm)
            lengths.append(signal.size)

    per_class = common_window_count(lengths, window_size)
    metadata = {
        "raw_dir": str(raw.resolve()),
        "window_size": int(window_size),
        "fft_size": int(window_size),
        "spectrum_length": int(spectrum_length),
        "per_domain_per_class_windows": int(per_class),
        "num_classes": 10,
        "class_names": list(CLASS_NAMES),
        "domain_nominal_rpm": DOMAIN_RPM,
        "file_map": CWRU_FILE_MAP,
        "domains": {},
    }

    for domain in range(4):
        x_list, y_list = [], []
        actual_rpm = {}
        for label in range(10):
            signal, rpm = loaded[(domain, label)]
            actual_rpm[str(label)] = rpm
            for idx in range(per_class):
                start = idx * int(window_size)
                segment = signal[start:start + int(window_size)]
                x_list.append(fft_window(segment, int(window_size), int(spectrum_length)))
                y_list.append(label)
        x = torch.from_numpy(np.stack(x_list)).unsqueeze(1).contiguous()
        y = torch.tensor(y_list, dtype=torch.long)
        payload = {
            "x": x,
            "y": y,
            "domain": int(domain),
            "nominal_rpm": float(DOMAIN_RPM[domain]),
            "class_names": list(CLASS_NAMES),
            "files": {str(k): CWRU_FILE_MAP[domain][k] for k in range(10)},
            "actual_rpm_by_label": actual_rpm,
            "window_size": int(window_size),
            "fft_size": int(window_size),
            "spectrum_length": int(spectrum_length),
            "per_class_windows": int(per_class),
        }
        target = out / f"domain_{domain}.pt"
        torch.save(payload, target)
        metadata["domains"][str(domain)] = {
            "cache": str(target.resolve()),
            "samples": int(x.size(0)),
            "per_class": int(per_class),
            "actual_rpm_by_label": actual_rpm,
        }

    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


class CWRUCacheDataset(TorchDataset):
    def __init__(self, cache_file: Path | str):
        payload = torch.load(Path(cache_file), map_location="cpu")
        self.x = payload["x"].float().contiguous()
        self.y = payload["y"].long().contiguous()
        if self.x.ndim == 2:
            self.x = self.x.unsqueeze(1)
        if self.x.ndim != 3 or self.x.shape[1:] != (1, 512):
            raise ValueError(f"Expected cache x shape [N,1,512], got {tuple(self.x.shape)}")
        if self.y.ndim != 1 or self.y.numel() != self.x.size(0):
            raise ValueError("Cache labels must have shape [N]")

    def __len__(self):
        return int(self.y.numel())

    def __getitem__(self, index: int):
        idx = int(index)
        return self.x[idx], self.y[idx], idx


class CWRU:
    num_classes = 10
    inputchannel = 1

    def __init__(self, data_path, TL_Task=(0, 1), **kwargs):
        path = Path(str(data_path))
        if not (path / "domain_0.pt").is_file():
            sibling = path.parent / "CWRU_CACHE"
            if (sibling / "domain_0.pt").is_file():
                path = sibling
        self.data_path = path
        task = list(TL_Task)
        if len(task) != 2:
            raise ValueError(f"TL_Task must contain [source,target], got {TL_Task}")
        self.source = int(task[0])
        self.target = int(task[1])
        if self.source not in range(4) or self.target not in range(4):
            raise ValueError(f"CWRU domains must be 0..3, got {TL_Task}")

    def _domain(self, domain: int) -> CWRUCacheDataset:
        target = self.data_path / f"domain_{int(domain)}.pt"
        if not target.is_file():
            raise FileNotFoundError(
                f"CWRU cache missing: {target}. Run `python build_cwru_cache.py` first."
            )
        return CWRUCacheDataset(target)

    def data_generator(self):
        return self._domain(self.source), self._domain(self.target)
