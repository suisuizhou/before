"""Shared contract for strict HUST caches and source checkpoints."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import pickle
import re

import torch
import torch.nn as nn


HUST_PROTOCOL_SPLITS = {
    "bearing": ("Dataset/HUST_STRICT_CACHE_V2", [0, 1, 2, 3]),
    "load": ("Dataset/HUST_STRICT_LOAD_CACHE_V2", [0, 1, 2]),
    "bearing_load": ("Dataset/HUST_STRICT_BEARING_LOAD_CACHE_V2", list(range(12))),
    "bearing_6208_extrema": ("Dataset/HUST_STRICT_BEARING_LOAD_CACHE_V2", [9, 11]),
    # Same bearing (6208), all three released load conditions. Atomic cache
    # ids 9/10/11 correspond to 0/200/400 W; bearing and speed stay fixed.
    "bearing_6208_loads": ("Dataset/HUST_STRICT_BEARING_LOAD_CACHE_V2", [9, 10, 11]),
}

SOURCE_LOSS_TERM_KEYS = {
    "ordinary": {"clean"},
    "robust": {"clean", "style", "warp", "style_warp", "symmetric_kl", "feature"},
}


def apply_hust_protocol_split(cfg) -> str:
    """Apply the closed primary/supplementary dataset selector."""
    selector = str(getattr(cfg, "hust_protocol_split", "bearing"))
    if selector not in HUST_PROTOCOL_SPLITS:
        raise ValueError(f"invalid hust_protocol_split: {selector!r}")
    path, domains = HUST_PROTOCOL_SPLITS[selector]
    cfg.Dataset.data_path = path
    cfg.Dataset.TL_list = list(domains)
    return selector


LABEL_MAP = {"N": 0, "I": 1, "O": 2, "B": 3, "IB": 4, "IO": 5, "OB": 6}
DOMAIN_TO_BEARING = {0: 6205, 1: 6206, 2: 6207, 3: 6208}
BEARING_TO_DOMAIN = {value: key for key, value in DOMAIN_TO_BEARING.items()}
LOAD_CODE_TO_WATTS = {"00": 0, "02": 200, "04": 400}
FILENAME_RE = re.compile(r"^(IB|IO|OB|N|I|O|B)([4-8])(00|02|04)\.mat$")

CACHE_KEYS = {
    "data",
    "label",
    "shaft_hz",
    "load_w",
    "recording_id",
    "window_offset",
}
CACHE_PROCESSING = {
    "sampling_rate_hz": 51_200,
    "window_size": 2_048,
    "window_stride": 1_024,
    "fft_size": 2_048,
    "fft_scale": 2_048,
    "spectrum_length": 512,
    "frequency_resolution_hz": 25.0,
}


@dataclass(frozen=True)
class HUSTRecording:
    fault: str
    label: int
    bearing: int
    load_w: int


def parse_hust_filename(name: str) -> HUSTRecording:
    """Parse a released constant-speed HUST filename, rejecting all others."""

    match = FILENAME_RE.fullmatch(Path(name).name)
    if match is None:
        raise ValueError(f"unsupported HUST filename: {name}")
    fault, bearing_digit, load_code = match.groups()
    return HUSTRecording(
        fault=fault,
        label=LABEL_MAP[fault],
        bearing=6200 + int(bearing_digit),
        load_w=LOAD_CODE_TO_WATTS[load_code],
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_variant(variant: str) -> str:
    variant = str(variant)
    if variant not in {"ordinary", "robust"}:
        raise ValueError(f"invalid source_variant: {variant}")
    return variant


def hust_checkpoint_dir(root: Path, variant: str, source: int, seed: int) -> Path:
    """Return the isolated source-checkpoint directory for one HUST route."""

    return (
        Path(root)
        / _checkpoint_variant(variant)
        / f"source_{int(source)}"
        / f"seed_{int(seed)}"
    )


def resolve_hust_checkpoint(
    root: Path,
    variant: str,
    source: int,
    seed: int,
    model_name: str,
) -> Path:
    """Resolve an existing HUST checkpoint without crossing route families."""

    directory = hust_checkpoint_dir(root, variant, source, seed)
    best = directory / ("best_source_" + str(model_name))
    final = directory / str(model_name)
    if best.exists():
        return best
    if final.exists():
        return final
    raise FileNotFoundError(
        f"Neither best nor final HUST source checkpoint exists: {best} / {final}"
    )


def ensure_hust_adaptation_carrier(model: nn.Module, warp_knots: int = 16) -> list[str]:
    """Ensure a HUST source model carries zero-valued target adaptation tensors."""

    backbone = model[0] if isinstance(model, (nn.Sequential, nn.ModuleList)) else model
    if not hasattr(backbone, "band_scale") or backbone.band_scale is None:
        raise ValueError("ResNet18_1D_SDE spectral adapter carrier is disabled")
    if not hasattr(backbone, "band_bias") or backbone.band_bias is None:
        raise ValueError("ResNet18_1D_SDE spectral bias carrier is disabled")
    if not hasattr(backbone, "warp_ctrl"):
        device = next(backbone.parameters()).device
        backbone.register_parameter(
            "warp_ctrl", nn.Parameter(torch.zeros(1, 1, int(warp_knots), device=device))
        )
    return [
        name
        for name, _parameter in model.named_parameters()
        if any(token in name for token in ("band_scale", "band_bias", "warp_ctrl"))
    ]


def _load_hust_checkpoint_contract(
    model: nn.Module,
    path: Path,
    *,
    expected_directory: Path | None = None,
) -> dict:
    """Verify the adjacent source contract and strictly load every model tensor."""

    path = Path(path)
    summary_path = path.parent / "source_training_summary.json"
    metadata_text = summary_path.read_text(encoding="utf-8")
    try:
        metadata = json.loads(metadata_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid HUST source metadata: {summary_path}") from exc

    required = {
        "contract_version",
        "route",
        "source",
        "seed",
        "epochs",
        "hyperparameters",
        "cache_identity",
        "target_labels_consumed",
        "carrier_identity_check",
        "carrier_parameters",
        "tensor_count",
        "source_accuracy",
        "elapsed_seconds",
        "optimizer_steps",
        "execution",
        "source_result",
        "checkpoint",
        "checkpoint_sha256",
    }
    profile_fields = {"robust_profile", "robust_augmentation"}
    _require(
        frozenset(metadata) in {frozenset(required), frozenset(required | profile_fields)},
        "invalid HUST source metadata fields",
    )
    _require(metadata["contract_version"] == 1, "unsupported checkpoint metadata")
    route = _checkpoint_variant(metadata["route"])
    if profile_fields.issubset(metadata):
        profile = metadata["robust_profile"]
        _require(isinstance(profile, str) and bool(profile), "invalid robust profile")
        _require(
            (
                route == "ordinary"
                and profile == "not_applicable"
                and metadata["robust_augmentation"] is None
            )
            or (
                route == "robust"
                and isinstance(metadata["robust_augmentation"], dict)
            ),
            "robust augmentation metadata mismatch",
        )
    _require(metadata["target_labels_consumed"] is False, "target labels were consumed")
    _require(metadata["carrier_identity_check"] is True, "carrier is not identity")
    _require(metadata["epochs"] == 50, "checkpoint is not the formal epoch-50 carrier")
    _require(metadata["seed"] == 2025, "checkpoint seed mismatch")
    _require(
        metadata["hyperparameters"]
        == {
            "epochs": 50,
            "batch_size": 128,
            "num_workers": 4,
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "label_smoothing": 0.1,
            "seed": 2025,
        },
        "source hyperparameter metadata mismatch",
    )
    cache_identity = metadata["cache_identity"]
    _require(
        isinstance(cache_identity, dict)
        and set(cache_identity)
        == {"mode", "manifest_path", "manifest_sha256", "content_sha256", "tensor_sha256s"}
        and cache_identity["mode"] in {"formal", "smoke"}
        and re.fullmatch(r"[0-9a-f]{64}", str(cache_identity["content_sha256"])) is not None,
        "source cache identity metadata mismatch",
    )
    if cache_identity["mode"] == "formal":
        manifest_path = Path(str(cache_identity["manifest_path"]))
        cache = validate_cache(manifest_path.parent)
        tensor_hashes = {
            str((manifest_path.parent / str(summary["tensor_file"])).resolve()): str(summary["tensor_sha256"])
            for summary in cache["domains"].values()
        }
        _require(
            cache.get("version") == 2
            and cache.get("manifest_sha256") == cache_identity["manifest_sha256"]
            and cache.get("content_sha256") == cache_identity["content_sha256"]
            and tensor_hashes == cache_identity["tensor_sha256s"],
            "source cache content identity mismatch",
        )
    else:
        _require(
            cache_identity["manifest_path"] is None
            and cache_identity["manifest_sha256"] is None
            and cache_identity["tensor_sha256s"] == {},
            "smoke cache identity metadata mismatch",
        )
    _require(metadata["checkpoint"] == path.name, "checkpoint filename mismatch")
    _validate_source_result(metadata, route)
    _require(sha256_file(path) == metadata["checkpoint_sha256"], "checkpoint hash mismatch")

    expected_dir = (
        Path(expected_directory)
        if expected_directory is not None
        else hust_checkpoint_dir(
            path.parent.parent.parent.parent,
            route,
            metadata["source"],
            metadata["seed"],
        )
    )
    _require(path.parent == expected_dir, "checkpoint route metadata mismatch")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except OSError:
        raise
    except (EOFError, IndexError, RuntimeError, ValueError, pickle.UnpicklingError) as exc:
        raise ValueError(f"invalid HUST source checkpoint: {path}") from exc
    _require(isinstance(payload, dict), "checkpoint payload must be a mapping")
    _require(
        set(payload) == {"contract", "metadata", "state_dict"},
        "checkpoint payload contract mismatch",
    )
    _require(
        payload["contract"]
        == {
            "version": 1,
            "route": route,
            "source": int(metadata["source"]),
            "seed": 2025,
            "epoch": 50,
        },
        "checkpoint embedded metadata mismatch",
    )
    bound_metadata = dict(metadata)
    bound_metadata.pop("checkpoint_sha256")
    _require(payload["metadata"] == bound_metadata, "checkpoint metadata mismatch")
    state = payload["state_dict"]
    _require(isinstance(state, dict), "checkpoint state_dict must be a mapping")
    _require(len(state) == int(metadata["tensor_count"]), "checkpoint tensor count mismatch")
    actual_carriers = ensure_hust_adaptation_carrier(model)
    _require(
        metadata["carrier_parameters"] == actual_carriers,
        "carrier parameter metadata mismatch",
    )
    for name in metadata["carrier_parameters"]:
        _require(name in state, f"missing carrier tensor: {name}")
        _require(torch.is_tensor(state[name]), f"invalid carrier tensor: {name}")
        _require(bool(torch.count_nonzero(state[name]) == 0), f"non-identity carrier: {name}")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError("checkpoint does not strictly match the model") from exc
    return metadata


def _validate_source_result(metadata: dict, route: str) -> None:
    result = metadata["source_result"]
    required = {
        "route",
        "source",
        "seed",
        "epochs",
        "optimizer_steps",
        "target_labels_consumed",
        "carrier_identity",
        "loss_terms",
    }
    _require(
        isinstance(result, dict) and set(result) == required,
        "invalid source loss record",
    )
    _require(result["route"] == route, "source loss route mismatch")
    _require(result["source"] == metadata["source"], "source loss domain mismatch")
    _require(result["seed"] == metadata["seed"], "source loss seed mismatch")
    _require(
        isinstance(metadata["execution"], dict)
        and set(metadata["execution"]) == {"smoke_mode", "epochs_run"},
        "invalid source execution metadata",
    )
    _require(
        result["epochs"] == metadata["execution"]["epochs_run"]
        and isinstance(result["epochs"], int)
        and not isinstance(result["epochs"], bool)
        and result["epochs"] > 0,
        "source loss epoch mismatch",
    )
    _require(
        result["optimizer_steps"] == metadata["optimizer_steps"]
        and isinstance(result["optimizer_steps"], int)
        and not isinstance(result["optimizer_steps"], bool)
        and result["optimizer_steps"] > 0,
        "source loss optimizer-step mismatch",
    )
    _require(
        result["target_labels_consumed"] is False,
        "source loss consumed target labels",
    )
    _require(result["carrier_identity"] is True, "source loss carrier is not identity")
    terms = result["loss_terms"]
    _require(
        isinstance(terms, dict) and set(terms) == SOURCE_LOSS_TERM_KEYS[route],
        "invalid source loss terms",
    )
    _require(
        all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in terms.values()
        ),
        "non-finite source loss term",
    )


def strict_load_hust_checkpoint(model: nn.Module, path: Path) -> dict:
    """Verify route location, metadata, hash, carrier, and every model tensor."""

    return _load_hust_checkpoint_contract(model, path)


def _strict_load_staged_hust_checkpoint(
    model: nn.Module, path: Path, staging_directory: Path
) -> dict:
    """Apply the same strict checks before an unpublished directory is promoted."""

    return _load_hust_checkpoint_contract(
        model,
        path,
        expected_directory=staging_directory,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_cache(root: Path) -> dict:
    """Validate every cache tensor and return the manifest plus its exact SHA."""

    root = Path(root)
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid HUST cache manifest: {manifest_path}") from exc

    version = manifest.get("version")
    _require(version in {1, 2}, "unsupported HUST cache version")
    _require(manifest.get("label_map") == LABEL_MAP, "HUST label map mismatch")
    _require(manifest.get("processing") == CACHE_PROCESSING, "processing contract mismatch")
    _require(isinstance(manifest.get("seed"), int), "invalid preprocessing seed")
    _require(
        isinstance(manifest.get("balance_count"), int) and manifest["balance_count"] > 0,
        "invalid balance count",
    )
    split = manifest.get("split")
    _require(split in {"bearing", "load", "bearing_load"}, "invalid cache split")
    domain_map = manifest.get("domain_map")
    if split == "bearing":
        expected_domain_map = {"0": "6205", "1": "6206", "2": "6207", "3": "6208"}
    elif split == "load":
        expected_domain_map = {"0": "0W", "1": "200W", "2": "400W"}
    else:
        expected_domain_map = {
            str((bearing - 6205) * 3 + load_index): f"{bearing}@{load}W"
            for bearing in range(6205, 6209)
            for load_index, load in enumerate((0, 200, 400))
        }
    _require(domain_map == expected_domain_map, "invalid domain map")
    domains = manifest.get("domains")
    _require(isinstance(domains, dict), "missing domain summaries")
    _require(set(domains) == set(domain_map), "domain summary mismatch")

    if version == 2:
        tensor_identity = {}
        expected_files = set()
        for domain_text in sorted(domain_map, key=int):
            summary = domains[domain_text]
            tensor_file = f"domain_{int(domain_text)}.pt"
            _require(
                isinstance(summary, dict)
                and summary.get("tensor_file") == tensor_file
                and re.fullmatch(r"[0-9a-f]{64}", str(summary.get("tensor_sha256")))
                is not None,
                "invalid HUST domain tensor metadata",
            )
            tensor_path = root / tensor_file
            _require(tensor_path.is_file(), f"missing HUST domain tensor: {tensor_path}")
            actual_hash = sha256_file(tensor_path)
            _require(
                actual_hash == summary["tensor_sha256"],
                f"HUST domain tensor hash mismatch: {tensor_path}",
            )
            expected_files.add(tensor_file)
            tensor_identity[domain_text] = {
                "tensor_file": tensor_file,
                "tensor_sha256": actual_hash,
            }
        actual_files = {path.name for path in root.glob("domain_*.pt")}
        _require(actual_files == expected_files, "HUST domain tensor inventory mismatch")
        expected_content = hashlib.sha256(
            json.dumps(
                {"version": 2, "split": split, "domains": tensor_identity},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        _require(
            manifest.get("content_sha256") == expected_content,
            "HUST cache content hash mismatch",
        )

    source_files = manifest.get("source_files")
    expected_recording_count = 84 if split in {"bearing", "bearing_load"} else 99
    _require(
        isinstance(source_files, list)
        and len(source_files) == expected_recording_count,
        "unexpected source recording count",
    )
    _require(
        manifest.get("recording_count") == len(source_files),
        "manifest recording count mismatch",
    )
    expected_source_keys = {
        "recording_id",
        "path",
        "sha256",
        "fault",
        "label",
        "bearing",
        "load_w",
        "shaft_hz",
        "window_count",
    }
    source_cells = set()
    source_paths = []
    for recording_id, row in enumerate(source_files):
        _require(
            isinstance(row, dict) and set(row) == expected_source_keys,
            "invalid source metadata",
        )
        _require(row["recording_id"] == recording_id, "source recording_id mismatch")
        _require(isinstance(row["path"], str), "invalid source path")
        _require(
            re.fullmatch(r"[0-9a-f]{64}", str(row["sha256"])) is not None,
            "invalid source-file hash",
        )
        try:
            parsed = parse_hust_filename(Path(row["path"]).name)
        except ValueError as exc:
            raise ValueError("invalid source metadata filename") from exc
        _require(
            (
                row["fault"],
                row["label"],
                row["bearing"],
                row["load_w"],
            )
            == (parsed.fault, parsed.label, parsed.bearing, parsed.load_w),
            "source metadata does not match filename",
        )
        _require(
            isinstance(row["window_count"], int) and row["window_count"] > 0,
            "invalid source window count",
        )
        _require(
            isinstance(row["shaft_hz"], (int, float))
            and not isinstance(row["shaft_hz"], bool)
            and math.isfinite(row["shaft_hz"])
            and row["shaft_hz"] > 0,
            "invalid source shaft_hz",
        )
        source_paths.append(row["path"])
        source_cells.add((row["bearing"], row["label"], row["load_w"]))
    _require(source_paths == sorted(source_paths), "source metadata is not sorted")
    _require(len(set(source_paths)) == len(source_paths), "duplicate source path")
    expected_bearings = (
        range(6205, 6209)
        if split in {"bearing", "bearing_load"}
        else range(6204, 6209)
    )
    expected_cells = {
        (bearing, label, load)
        for bearing in expected_bearings
        for label in LABEL_MAP.values()
        for load in (0, 200, 400)
        if not (
            split in {"load", "bearing_load"}
            and bearing == 6204
            and label in {LABEL_MAP["B"], LABEL_MAP["IB"]}
        )
    }
    _require(source_cells == expected_cells, "incomplete source recording cells")

    for domain_text in domain_map:
        try:
            domain = int(domain_text)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid domain key: {domain_text!r}") from exc
        tensor_path = root / f"domain_{domain}.pt"
        try:
            obj = torch.load(tensor_path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"invalid HUST domain tensor: {tensor_path}") from exc

        _require(isinstance(obj, dict) and set(obj) == CACHE_KEYS, "cache key mismatch")
        _require(all(torch.is_tensor(obj[key]) for key in CACHE_KEYS), "non-tensor cache field")
        n = int(obj["label"].numel())
        _require(n > 0, "empty cache domain")
        _require(tuple(obj["data"].shape) == (n, 1, 512), "data shape mismatch")
        for key in CACHE_KEYS - {"data"}:
            _require(tuple(obj[key].shape) == (n,), f"{key} shape mismatch")
        _require(obj["data"].dtype == torch.float32, "data dtype mismatch")
        _require(obj["shaft_hz"].dtype == torch.float32, "shaft_hz dtype mismatch")
        for key in ("label", "load_w", "recording_id", "window_offset"):
            _require(obj[key].dtype == torch.int64, f"{key} dtype mismatch")
        _require(bool(torch.isfinite(obj["data"]).all()), "non-finite cache data")
        _require(bool(torch.isfinite(obj["shaft_hz"]).all()), "non-finite shaft_hz")
        _require(bool((obj["shaft_hz"] > 0).all()), "non-positive shaft_hz")
        _require(
            set(obj["label"].tolist()).issubset(set(LABEL_MAP.values())),
            "invalid cache label",
        )
        _require(
            set(obj["load_w"].tolist()).issubset({0, 200, 400}),
            "invalid cache load",
        )
        _require(domains[domain_text].get("shape") == [n, 1, 512], "manifest shape mismatch")
        actual_class_counts = {
            str(label): int((obj["label"] == label).sum()) for label in LABEL_MAP.values()
        }
        actual_load_counts = {
            str(load): int((obj["load_w"] == load).sum()) for load in (0, 200, 400)
        }
        _require(
            domains[domain_text].get("class_counts") == actual_class_counts,
            "manifest class counts mismatch",
        )
        _require(
            domains[domain_text].get("load_counts") == actual_load_counts,
            "manifest load counts mismatch",
        )
        class_values = list(actual_class_counts.values())
        _require(
            all(
                value
                == manifest["balance_count"] * (3 if split == "bearing" else 1)
                for value in class_values
            ),
            "cache class balance mismatch",
        )
        if split == "bearing":
            _require(
                all(
                    value == manifest["balance_count"] * 7
                    for value in actual_load_counts.values()
                ),
                "cache load balance mismatch",
            )
            for label in LABEL_MAP.values():
                for load in (0, 200, 400):
                    _require(
                        int(((obj["label"] == label) & (obj["load_w"] == load)).sum())
                        == manifest["balance_count"],
                        "cache bearing/class/load balance mismatch",
                    )
        else:
            domain_load = int(domain_map[domain_text].split("@")[-1].removesuffix("W"))
            _require(
                bool((obj["load_w"] == domain_load).all()),
                "cache load-domain membership mismatch",
            )

        recording_ids = obj["recording_id"]
        _require(
            bool((recording_ids >= 0).all())
            and bool((recording_ids < len(source_files)).all()),
            "recording_id out of bounds",
        )
        offsets = obj["window_offset"]
        _require(
            bool((offsets >= 0).all())
            and bool((offsets % CACHE_PROCESSING["window_stride"] == 0).all()),
            "invalid window offset alignment",
        )
        ids = recording_ids.tolist()
        expected_labels = torch.tensor(
            [source_files[index]["label"] for index in ids], dtype=torch.int64
        )
        expected_loads = torch.tensor(
            [source_files[index]["load_w"] for index in ids], dtype=torch.int64
        )
        expected_shafts = torch.tensor(
            [source_files[index]["shaft_hz"] for index in ids], dtype=torch.float32
        )
        max_offsets = torch.tensor(
            [
                source_files[index]["window_count"]
                * CACHE_PROCESSING["window_stride"]
                for index in ids
            ],
            dtype=torch.int64,
        )
        _require(torch.equal(obj["label"], expected_labels), "recording label mismatch")
        _require(torch.equal(obj["load_w"], expected_loads), "recording load mismatch")
        _require(torch.equal(obj["shaft_hz"], expected_shafts), "recording shaft_hz mismatch")
        _require(bool((offsets < max_offsets).all()), "window offset out of range")
        pairs = list(zip(ids, offsets.tolist()))
        _require(len(set(pairs)) == len(pairs), "duplicate recording window")

        if split == "bearing":
            domain_bearing = int(domain_map[domain_text])
            _require(
                all(source_files[index]["bearing"] == domain_bearing for index in ids),
                "recording bearing-domain membership mismatch",
            )
            per_recording_counts = torch.bincount(
                recording_ids, minlength=len(source_files)
            )
            domain_recordings = [
                row["recording_id"]
                for row in source_files
                if row["bearing"] == domain_bearing
            ]
            _require(
                all(
                    int(per_recording_counts[index]) == manifest["balance_count"]
                    for index in domain_recordings
                ),
                "per-recording balance mismatch",
            )
        else:
            domain_load = int(domain_map[domain_text].split("@")[-1].removesuffix("W"))
            _require(
                all(source_files[index]["load_w"] == domain_load for index in ids),
                "recording load-domain membership mismatch",
            )
            if split == "bearing_load":
                domain_bearing = int(domain_map[domain_text].split("@")[0])
                _require(
                    all(source_files[index]["bearing"] == domain_bearing for index in ids),
                    "recording bearing-load-domain membership mismatch",
                )

    return {**manifest, "manifest_sha256": sha256_file(manifest_path)}
