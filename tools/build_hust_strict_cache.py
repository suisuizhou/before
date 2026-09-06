#!/usr/bin/env python3
"""Build the deterministic version-2 strict HUST cache from raw MAT files."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Literal

import numpy as np
from scipy.io import loadmat
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Lib.hust_strict_protocol import (  # noqa: E402
    BEARING_TO_DOMAIN,
    CACHE_PROCESSING,
    DOMAIN_TO_BEARING,
    LABEL_MAP,
    parse_hust_filename,
    sha256_file,
    validate_cache,
)


WINDOW_SIZE = 2_048
WINDOW_STRIDE = 1_024
SPECTRUM_LENGTH = 512
PROCESSING = CACHE_PROCESSING


def fft_window(signal: np.ndarray) -> np.ndarray:
    """Return the strict one-sided retained FFT representation."""

    window = np.asarray(signal)
    if window.ndim != 1 or window.shape[0] != WINDOW_SIZE:
        raise ValueError(f"FFT window must be a 1-D array of length {WINDOW_SIZE}")
    if not np.isfinite(window).all():
        raise ValueError("FFT window contains non-finite values")
    spectrum = np.abs(np.fft.fft(window, n=WINDOW_SIZE))[:SPECTRUM_LENGTH]
    return (spectrum / float(WINDOW_SIZE)).astype(np.float32, copy=False)[None, :]


def balanced_window_indices(
    available: dict[str, int], seed: int
) -> dict[str, np.ndarray]:
    """Select the same number of windows for every sorted recording key."""

    if not available or any(int(count) <= 0 for count in available.values()):
        raise ValueError("every balancing cell must contain at least one window")
    balance_count = min(int(count) for count in available.values())
    selected = {}
    for recording_index, key in enumerate(sorted(available)):
        generator = np.random.default_rng(int(seed) + recording_index)
        indices = generator.choice(int(available[key]), size=balance_count, replace=False)
        selected[key] = np.sort(indices.astype(np.int64, copy=False))
    return selected


def _discover(raw_root: Path, split: Literal["bearing", "load"]):
    raw_root = Path(raw_root)
    if not raw_root.is_dir():
        raise ValueError(f"raw HUST root is not a directory: {raw_root}")
    rows = []
    paths = sorted(
        raw_root.rglob("*.mat"),
        key=lambda item: item.relative_to(raw_root).as_posix(),
    )
    for path in paths:
        recording = parse_hust_filename(path.name)
        if split == "bearing" and recording.bearing not in BEARING_TO_DOMAIN:
            continue
        rows.append((path, recording))
    if not rows:
        raise ValueError("no supported HUST recordings found")

    expected_bearings = (
        set(DOMAIN_TO_BEARING.values())
        if split == "bearing"
        else set(range(6204, 6209))
    )
    present = {(row.bearing, row.label, row.load_w) for _, row in rows}
    missing = [
        (bearing, label, load)
        for bearing in sorted(expected_bearings)
        for label in LABEL_MAP.values()
        for load in (0, 200, 400)
        if (bearing, label, load) not in present
        and not (
            split == "load"
            and bearing == 6204
            and label in {LABEL_MAP["B"], LABEL_MAP["IB"]}
        )
    ]
    if missing:
        raise ValueError(f"missing HUST bearing/class/load cells: {missing[:5]}")
    if len(present) != len(rows):
        raise ValueError("duplicate HUST bearing/class/load recordings")
    return rows


def _load_recording(path: Path) -> tuple[np.ndarray, float]:
    try:
        mat = loadmat(path)
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read MAT file: {path}") from exc
    if "data" not in mat or "fs" not in mat:
        raise ValueError(f"MAT file must contain data and fs: {path}")
    signal = np.asarray(mat["data"])
    if signal.ndim == 2 and 1 in signal.shape:
        signal = signal.reshape(-1)
    if signal.ndim != 1 or not np.issubdtype(signal.dtype, np.number):
        raise ValueError(f"MAT data must be a numeric vector: {path}")
    if np.iscomplexobj(signal):
        raise ValueError(f"MAT data must be real-valued: {path}")
    signal = signal.astype(np.float32, copy=False)
    if not np.isfinite(signal).all():
        raise ValueError(f"MAT data contains non-finite values: {path}")
    shaft = np.asarray(mat["fs"])
    if shaft.size != 1:
        raise ValueError(f"MAT fs must be scalar: {path}")
    if not np.issubdtype(shaft.dtype, np.number) or np.iscomplexobj(shaft):
        raise ValueError(f"MAT fs must be real-valued: {path}")
    shaft_hz = float(shaft.reshape(-1)[0])
    if not np.isfinite(shaft_hz) or shaft_hz <= 0:
        raise ValueError(f"MAT fs must be finite and positive: {path}")
    if signal.size < WINDOW_SIZE:
        raise ValueError(f"MAT data is shorter than one window: {path}")
    return signal, shaft_hz


def _window_count(signal: np.ndarray) -> int:
    return 1 + (int(signal.size) - WINDOW_SIZE) // WINDOW_STRIDE


def _inspect_rows(raw_root: Path, rows):
    source_files = []
    loaded = []
    for recording_id, (path, recording) in enumerate(rows):
        signal, shaft_hz = _load_recording(path)
        source_files.append(
            {
                "recording_id": recording_id,
                "path": path.relative_to(raw_root).as_posix(),
                "sha256": sha256_file(path),
                "fault": recording.fault,
                "label": recording.label,
                "bearing": recording.bearing,
                "load_w": recording.load_w,
                "shaft_hz": shaft_hz,
                "window_count": _window_count(signal),
            }
        )
        loaded.append((path, recording, recording_id, signal, shaft_hz))
    return source_files, loaded


def _identity(seed: int, split: str, source_files: list[dict]) -> dict:
    return {
        "version": 2,
        "split": split,
        "seed": int(seed),
        "processing": PROCESSING,
        "label_map": LABEL_MAP,
        "source_files": source_files,
    }


def _append_sample(store, signal, index, recording, shaft_hz, recording_id):
    offset = int(index) * WINDOW_STRIDE
    store["data"].append(fft_window(signal[offset : offset + WINDOW_SIZE]))
    store["label"].append(recording.label)
    store["shaft_hz"].append(shaft_hz)
    store["load_w"].append(recording.load_w)
    store["recording_id"].append(recording_id)
    store["window_offset"].append(offset)


def _empty_store():
    keys = (
        "data",
        "label",
        "shaft_hz",
        "load_w",
        "recording_id",
        "window_offset",
    )
    return {key: [] for key in keys}


def _to_tensor_object(store) -> dict[str, torch.Tensor]:
    return {
        "data": torch.from_numpy(np.stack(store["data"])).float(),
        "label": torch.tensor(store["label"], dtype=torch.long),
        "shaft_hz": torch.tensor(store["shaft_hz"], dtype=torch.float32),
        "load_w": torch.tensor(store["load_w"], dtype=torch.long),
        "recording_id": torch.tensor(store["recording_id"], dtype=torch.long),
        "window_offset": torch.tensor(store["window_offset"], dtype=torch.long),
    }


def _summarize(obj: dict[str, torch.Tensor]) -> dict:
    class_counts = Counter(obj["label"].tolist())
    load_counts = Counter(obj["load_w"].tolist())
    return {
        "shape": list(obj["data"].shape),
        "class_counts": {str(key): class_counts[key] for key in LABEL_MAP.values()},
        "load_counts": {str(key): load_counts[key] for key in (0, 200, 400)},
    }


def _prepare(loaded, split: Literal["bearing", "load"], seed: int):
    stores = defaultdict(_empty_store)
    if split == "bearing":
        available = {path.name: _window_count(signal) for path, _, _, signal, _ in loaded}
        choices = balanced_window_indices(available, seed)
        balance_count = min(available.values())
        for path, recording, recording_id, signal, shaft_hz in loaded:
            domain = BEARING_TO_DOMAIN[recording.bearing]
            for index in choices[path.name]:
                _append_sample(stores[domain], signal, index, recording, shaft_hz, recording_id)
        domain_map = {str(domain): str(bearing) for domain, bearing in DOMAIN_TO_BEARING.items()}
    else:
        load_to_domain = {0: 0, 200: 1, 400: 2}
        candidates = defaultdict(list)
        for path, recording, recording_id, signal, shaft_hz in loaded:
            key = f"{recording.load_w}:{recording.label}"
            candidates[key].extend(
                (recording, recording_id, signal, shaft_hz, index)
                for index in range(_window_count(signal))
            )
        choices = balanced_window_indices(
            {key: len(value) for key, value in candidates.items()}, seed
        )
        balance_count = min(len(value) for value in candidates.values())
        for key in sorted(candidates):
            for candidate_index in choices[key]:
                recording, recording_id, signal, shaft_hz, index = candidates[key][
                    candidate_index
                ]
                domain = load_to_domain[recording.load_w]
                _append_sample(stores[domain], signal, index, recording, shaft_hz, recording_id)
        domain_map = {"0": "0W", "1": "200W", "2": "400W"}

    objects = {domain: _to_tensor_object(stores[domain]) for domain in range(len(domain_map))}
    return objects, domain_map, int(balance_count)


def build_cache(
    raw_root: Path,
    output_root: Path,
    seed: int,
    split: Literal["bearing", "load"] = "bearing",
) -> dict:
    """Build and atomically promote a deterministic strict HUST cache."""

    if split not in {"bearing", "load"}:
        raise ValueError(f"unsupported HUST split: {split}")
    raw_root = Path(raw_root).resolve()
    output_root = Path(output_root)
    rows = _discover(raw_root, split)
    source_files, loaded = _inspect_rows(raw_root, rows)
    requested_identity = _identity(seed, split, source_files)

    if output_root.exists():
        try:
            existing = validate_cache(output_root)
        except ValueError as exc:
            raise FileExistsError(
                f"refusing to replace non-identical output directory: {output_root}"
            ) from exc
        if all(existing.get(key) == value for key, value in requested_identity.items()):
            print(f"existing cache manifest SHA256: {existing['manifest_sha256']}")
            return existing
        raise FileExistsError(
            f"refusing to replace non-identical output directory: {output_root}"
        )

    objects, domain_map, balance_count = _prepare(loaded, split, int(seed))
    manifest = {
        **requested_identity,
        "domain_map": domain_map,
        "balance_count": balance_count,
        "recording_count": len(rows),
        "domains": {str(domain): _summarize(obj) for domain, obj in objects.items()},
    }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        for domain, obj in objects.items():
            torch.save(obj, temp_root / f"domain_{domain}.pt")
        tensor_identity = {}
        for domain in sorted(objects):
            tensor_path = temp_root / f"domain_{domain}.pt"
            digest = sha256_file(tensor_path)
            manifest["domains"][str(domain)].update(
                tensor_file=tensor_path.name,
                tensor_sha256=digest,
            )
            tensor_identity[str(domain)] = {
                "tensor_file": tensor_path.name,
                "tensor_sha256": digest,
            }
        manifest["content_sha256"] = hashlib.sha256(
            json.dumps(
                {"version": 2, "split": split, "domains": tensor_identity},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        (temp_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        validated = validate_cache(temp_root)
        os.replace(temp_root, output_root)
        return validated
    finally:
        if temp_root.exists():
            shutil.rmtree(temp_root)


def _dry_run(raw_root: Path, seed: int, split: Literal["bearing", "load"]) -> None:
    raw_root = Path(raw_root).resolve()
    rows = _discover(raw_root, split)
    loaded_counts = {}
    for path, _ in rows:
        signal, _ = _load_recording(path)
        loaded_counts[path.name] = _window_count(signal)
    if split == "bearing":
        balance_count = min(loaded_counts.values())
        print(f"primary recordings: {len(rows)}")
        print("domains: 4")
    else:
        groups = Counter()
        for path, recording in rows:
            groups[(recording.load_w, recording.label)] += loaded_counts[path.name]
        balance_count = min(groups.values())
        print(f"load-audit recordings: {len(rows)}")
        print("domains: 3")
    print(f"classes: {len(LABEL_MAP)}")
    print("loads: 3")
    print(f"global balance count: {balance_count}")
    print(f"seed: {seed}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--split", choices=("bearing", "load"), default="bearing")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        _dry_run(args.raw_root, args.seed, args.split)
    else:
        manifest = build_cache(args.raw_root, args.output, args.seed, args.split)
        print(f"cache manifest SHA256: {manifest['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
