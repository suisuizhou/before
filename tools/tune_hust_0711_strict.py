#!/usr/bin/env python3
"""Strict HUST experiment orchestration with deterministic resume semantics."""
from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from threading import Condition
import time
from typing import Any

import yaml


ALL_TASKS = tuple((s, t) for s in range(4) for t in range(4) if s != t)
DEV_TASKS = ((0, 1), (1, 2), (2, 3), (3, 0))
ROUTES = {"dtcc_ordinary", "0711_robust", "0711_common"}
_DECLARED_TARGET_JOB_FIELDS = frozenset({
    "kind", "stage", "route", "variant", "source", "target", "task",
    "source_seed", "stream_seed", "overrides", "candidate_id", "config_sha256",
    "beginning_only", "load_split", "expected_result_kind",
    "expected_result_contract", "evidence_status", "artifacts",
    "cache_manifest_path", "cache_manifest_sha256", "runner_script_path",
    "runner_script_sha256", "experiment_config_path", "experiment_config_sha256",
    "source_checkpoint_path", "source_checkpoint_sha256", "source_summary_path",
    "source_summary_sha256", "cache_content_sha256", "cache_tensor_sha256s",
    "freeze_config_path", "freeze_config_sha256", "freeze_sha256",
    "freeze_proof_path", "freeze_proof_file_sha256", "freeze_proof_sha256",
    "freeze_frozen_at",
})
_PARSER_TARGET_STATE_FIELDS = frozenset({
    "schema_version", "result_kind", "beginning", "before", "strict_online",
    "post_stream", "macro_precision", "macro_recall", "macro_f1",
    "post_macro_precision", "post_macro_recall", "post_macro_f1",
    "confusion_matrix", "offline_confusion_matrix", "samples", "batches", "passes",
    "class_coverage", "finite_losses", "trainable_parameters", "trainable_allowlist",
    "pre_update_scoring", "metadata_evidence_used", "runtime_seconds",
    "peak_memory_mb", "mean_batch_ms", "memory_size", "memory_class_coverage",
    "certain_ratio", "uncertain_ratio", "offline_purity", "certain_purity",
    "evidence_applicable", "evidence_active_ratio",
})
TASK_STATUSES = {"pending", "running", "succeeded", "failed"}
EXPECTED_GROUPS = (
    ("learning_rate", ({"Opt.lr_tar": 0.008}, {"Opt.lr_tar": 0.015}, {"Opt.lr_tar": 0.024}, {"Opt.lr_tar": 0.036})),
    ("adapter_lr_scale", ({"TTA0711.adapter_lr_scale": 0.5}, {"TTA0711.adapter_lr_scale": 1.0}, {"TTA0711.adapter_lr_scale": 2.0})),
    ("warp_lr_scale", ({"TTA0711.warp_lr_scale": 0.05}, {"TTA0711.warp_lr_scale": 0.10}, {"TTA0711.warp_lr_scale": 0.20})),
    ("ema_beta", ({"TTA0711.ema_beta": 0.990}, {"TTA0711.ema_beta": 0.995}, {"TTA0711.ema_beta": 0.999})),
    ("warmup_batches", ({"TTA0711.warmup_batches": 0}, {"TTA0711.warmup_batches": 5}, {"TTA0711.warmup_batches": 10})),
    ("aux_ramp_batches", ({"TTA0711.aux_ramp_batches": 10}, {"TTA0711.aux_ramp_batches": 20}, {"TTA0711.aux_ramp_batches": 40})),
    ("minimum_reliability", ({"TTA0711.min_reliability": 0.10}, {"TTA0711.min_reliability": 0.20}, {"TTA0711.min_reliability": 0.30})),
    ("loss_profile", ({"TTA0711.lambda_mt": 0.01, "TTA0711.lambda_pcl": 0.01, "TTA0711.lambda_ncl": 0.005}, {"TTA0711.lambda_mt": 0.02, "TTA0711.lambda_pcl": 0.02, "TTA0711.lambda_ncl": 0.01}, {"TTA0711.lambda_mt": 0.04, "TTA0711.lambda_pcl": 0.04, "TTA0711.lambda_ncl": 0.02})),
    ("memory_per_class", ({"TTA0711.memory_per_class": 32}, {"TTA0711.memory_per_class": 64}, {"TTA0711.memory_per_class": 128})),
    ("contrastive_temperature", ({"TTA0711.pcl_temperature": 0.10, "TTA0711.ncl_temperature": 0.10}, {"TTA0711.pcl_temperature": 0.20, "TTA0711.ncl_temperature": 0.20}, {"TTA0711.pcl_temperature": 0.30, "TTA0711.ncl_temperature": 0.30})),
)
NVIDIA_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
    "--format=csv,noheader,nounits",
]
_PLAN_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _planned_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    stat = path.stat()
    key = (str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns))
    if key not in _PLAN_HASH_CACHE:
        _PLAN_HASH_CACHE[key] = _sha256_file(path)
    return _PLAN_HASH_CACHE[key]


def atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("config must be a mapping")
    validate_config(value)
    return value


def validate_config(config: Mapping[str, object]) -> None:
    required = {"version", "protocol", "routes", "checkpoints", "tasks", "budget", "selection", "beginning_audit", "fixed_overrides", "baseline_overrides", "search"}
    if set(config) != required:
        raise ValueError(f"top-level schema mismatch: expected {sorted(required)}")
    if config["version"] != 1:
        raise ValueError("config version must be 1")
    protocol = config["protocol"]
    if not isinstance(protocol, Mapping):
        raise ValueError("protocol must be a mapping")
    immutable = {"dataset": "HUSTStrict", "data_path": "Dataset/HUST_STRICT_CACHE_V2", "load_data_path": "Dataset/HUST_STRICT_LOAD_CACHE_V2", "model": "ResNet18_1D_SDE", "batch_size": 128, "num_workers": 4, "source_seed": 2025, "stream_seed": 2025, "passes": 1, "gpu_policy": "auto_idle", "max_gpu_utilization": 10, "max_gpu_memory_fraction": 0.10}
    if dict(protocol) != immutable:
        raise ValueError("protocol invariants changed")
    if dict(config["routes"]) != {"source_ordinary": "main_src_dtcc_hust_strict.py", "source_robust": "main_src_0711_hust_strict.py", "dtcc_ordinary": "main_tta_dtcc_hust_strict.py", "0711_robust": "main_tta_0711_hust_strict.py", "0711_common": "main_tta_0711_hust_strict.py"}:
        raise ValueError("route paths changed")
    if dict(config["checkpoints"]) != {"root": "TTA_Model_HUST_STRICT_V2", "load_root": "TTA_Model_HUST_STRICT_LOAD_V2"}:
        raise ValueError("checkpoint roots changed")
    tasks = config["tasks"]
    if not isinstance(tasks, Mapping):
        raise ValueError("tasks must be a mapping")
    development = tuple(tuple(x) for x in tasks.get("development", []))
    heldout = tuple(tuple(x) for x in tasks.get("heldout", []))
    if development != DEV_TASKS or set(development + heldout) != set(ALL_TASKS) or set(development) & set(heldout):
        raise ValueError("development/heldout partition is not the fixed 12-task split")
    selection = config["selection"]
    if selection != {"minimum_mean_gain": 0.30, "maximum_task_regression": 1.00, "stability_stream_seed": 2026}:
        raise ValueError("selection guards changed")
    if dict(config["budget"]) != {"tuning_hours": 24, "estimated_task_minutes": 16, "reserve_minutes": 30}:
        raise ValueError("tuning budget changed")
    if dict(config["beginning_audit"]) != {"preferred_min": 25.0, "preferred_max": 70.0, "acceptable_count": 9, "absolute_min": 15.0, "absolute_max": 80.0}:
        raise ValueError("Beginning audit thresholds changed")
    groups = config["search"].get("groups", [])
    observed_groups = tuple((group.get("name"), tuple(group.get("values", []))) for group in groups)
    if observed_groups != EXPECTED_GROUPS:
        raise ValueError("search differs from the ten declared coordinate groups")
    fixed = config["fixed_overrides"]
    expected_fixed = {"batch_size": 128, "TTA0711.passes": 1, "TTA0711.sampling_rate_hz": 51200, "TTA0711.fft_size": 2048, "TTA0711.spectrum_length": 512, "TTA0711.physical_harmonics": 8, "TTA0711.outer_sideband_orders": [0, 1], "TTA0711.inner_sideband_orders": [0, 1, 2], "TTA0711.ball_sideband_orders": [0, 1, 2], "TTA0711.max_mask_ratio": 0.18, "TTA0711.adapter_delta": 0.10, "TTA0711.max_warp": 2.0, "TTA0711.warp_knots": 16, "TTA0711.min_pcl_classes": 3, "TTA0711.min_ncl_classes": 5, "TTA0711.min_ncl_entries": 20}
    if dict(fixed) != expected_fixed:
        raise ValueError("fixed strict runner invariants changed")
    expected_baseline = {"Opt.lr_tar": 0.015, "TTA0711.adapter_lr_scale": 1.0, "TTA0711.warp_lr_scale": 0.10, "TTA0711.ema_beta": 0.995, "TTA0711.warmup_batches": 5, "TTA0711.aux_ramp_batches": 20, "TTA0711.min_reliability": 0.20, "TTA0711.lambda_mt": 0.02, "TTA0711.lambda_pcl": 0.02, "TTA0711.lambda_ncl": 0.01, "TTA0711.memory_per_class": 64, "TTA0711.pcl_temperature": 0.20, "TTA0711.ncl_temperature": 0.20}
    if dict(config["baseline_overrides"]) != expected_baseline or dict(config["search"]["initial_candidate"]) != {"name": "frozen_pu4d_recommendation", "overrides": {"Opt.lr_tar": 0.024, "TTA0711.warp_lr_scale": 0.10}}:
        raise ValueError("baseline/initial candidate changed")


def discover_idle_gpus(rows: str | None = None, *, max_utilization: float = 10, max_memory_fraction: float = 0.10) -> list[int]:
    if rows is None:
        rows = subprocess.run(NVIDIA_QUERY, check=True, text=True, capture_output=True).stdout
    eligible = []
    for line in rows.splitlines():
        if not line.strip():
            continue
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 5:
            raise ValueError(f"malformed nvidia-smi row: {line!r}")
        index, _name, total, used, utilization = fields
        total_value = float(total)
        if total_value <= 0:
            continue
        if float(utilization) <= max_utilization and float(used) / total_value <= max_memory_fraction:
            eligible.append(int(index))
    return sorted(eligible)


def _gpu_is_idle(gpu: int, max_utilization: float, max_memory_fraction: float) -> bool:
    completed = subprocess.run([*NVIDIA_QUERY, f"--id={gpu}"], check=True, text=True, capture_output=True)
    return gpu in discover_idle_gpus(completed.stdout, max_utilization=max_utilization, max_memory_fraction=max_memory_fraction)


def normalize_overrides(overrides: Mapping[str, object]) -> dict[str, object]:
    return {key: overrides[key] for key in sorted(overrides)}


def candidate_id(overrides: Mapping[str, object], source_seed: int, stream_seed: int | None = None) -> str:
    """Configuration identity; evaluation seed/task/route intentionally do not participate."""
    payload = {"overrides": normalize_overrides(overrides), "source_seed": int(source_seed)}
    return _sha256_bytes(_canonical(payload))[:12]


def validate_candidate_overrides(
    config: Mapping[str, object], identifier: str, overrides: Mapping[str, object]
) -> dict[str, object]:
    """Validate a frozen candidate against the closed HUST search space."""
    normalized = normalize_overrides(overrides)
    if identifier == "untuned":
        if normalized not in ({}, normalize_overrides(config["baseline_overrides"])):
            raise ValueError("untuned fallback overrides changed")
        return normalized
    expected_id = candidate_id(normalized, int(config["protocol"]["source_seed"]))
    if str(identifier) != expected_id:
        raise ValueError("candidate_id does not match normalized overrides")
    allowed: dict[str, set[bytes]] = {
        key: {_canonical(value)}
        for key, value in dict(config["baseline_overrides"]).items()
    }
    for group in config["search"]["groups"]:
        for values in group["values"]:
            for key, value in values.items():
                allowed.setdefault(key, set()).add(_canonical(value))
    for key, value in dict(config["search"]["initial_candidate"]["overrides"]).items():
        allowed.setdefault(key, set()).add(_canonical(value))
    if set(normalized) - set(allowed) or any(
        _canonical(value) not in allowed.get(key, set())
        for key, value in normalized.items()
    ):
        raise ValueError("candidate overrides are outside the validated search space")
    loss_tuple = tuple(
        normalized.get(key, config["baseline_overrides"][key])
        for key in ("TTA0711.lambda_mt", "TTA0711.lambda_pcl", "TTA0711.lambda_ncl")
    )
    allowed_loss = {
        tuple(value[key] for key in ("TTA0711.lambda_mt", "TTA0711.lambda_pcl", "TTA0711.lambda_ncl"))
        for value in config["search"]["groups"][7]["values"]
    }
    temperatures = (
        normalized.get("TTA0711.pcl_temperature", config["baseline_overrides"]["TTA0711.pcl_temperature"]),
        normalized.get("TTA0711.ncl_temperature", config["baseline_overrides"]["TTA0711.ncl_temperature"]),
    )
    allowed_temperatures = {
        (value["TTA0711.pcl_temperature"], value["TTA0711.ncl_temperature"])
        for value in config["search"]["groups"][9]["values"]
    }
    if loss_tuple not in allowed_loss or temperatures not in allowed_temperatures:
        raise ValueError("candidate compound overrides are outside the validated search space")
    return normalized


def expand_coordinate_group(anchor: Mapping[str, object], values: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    # The anchor is always evaluated first, even if absent from a group's values.
    candidates = [normalize_overrides(anchor)]
    seen = {_canonical(candidates[0])}
    for value in values:
        merged = normalize_overrides({**anchor, **value})
        marker = _canonical(merged)
        if marker not in seen:
            candidates.append(merged)
            seen.add(marker)
    return candidates


def _hydra_value(value: object) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), separators=(",", ":"))
    return str(value)


def _override(key: str, value: object) -> str:
    return f"++{key}={_hydra_value(value)}"


def build_command(config: Mapping[str, object], route: str, overrides: Mapping[str, object], task: tuple[int, int], stream_seed: int, gpu: int) -> list[str]:
    validate_config(config)
    if route not in ROUTES:
        raise ValueError(f"unknown target route: {route}")
    if tuple(task) not in ALL_TASKS:
        raise ValueError(f"invalid directed task: {task}")
    routes, protocol = config["routes"], config["protocol"]
    source, target = task
    command = [sys.executable, routes[route], "Model=ResNet18_1D_SDE", "Dataset=HUSTStrict", "gpu_id=0", "process_wandb=False", "++hust_protocol_split=bearing", f"++Dataset.data_path={protocol['data_path']}", f"++seed_runs=[{protocol['source_seed']}]", f"++only_task=[{source},{target}]", "batch_size=128", "num_workers=4"]
    if route == "dtcc_ordinary":
        command += [f"++stream_seed={int(stream_seed)}", f"++hust_checkpoint_root={config['checkpoints']['root']}"]
        return command
    variant = "robust" if route == "0711_robust" else "ordinary"
    command += [f"++source_variant={variant}", f"++TTA0711.stream_seed={int(stream_seed)}", f"++hust_checkpoint_root={config['checkpoints']['root']}"]
    effective = {**config["fixed_overrides"], **config["baseline_overrides"], **normalize_overrides(overrides)}
    command.extend(_override(key, value) for key, value in effective.items())
    return command


def build_stage_command(config: Mapping[str, object], kind: str, *, source: int | None = None, task: tuple[int, int] | None = None, variant: str = "ordinary", load_split: bool = False, gpu: int = 0) -> list[str]:
    """Build source, Beginning-only, or supplementary commands."""
    routes, protocol, checkpoints = config["routes"], config["protocol"], config["checkpoints"]
    data_path = protocol["load_data_path"] if load_split else protocol["data_path"]
    root = checkpoints["load_root"] if load_split else checkpoints["root"]
    if kind == "source":
        if source is None or variant not in {"ordinary", "robust"}:
            raise ValueError("source command requires a source and valid variant")
        split = "load" if load_split else "bearing"
        return [sys.executable, routes[f"source_{variant}"], "Model=ResNet18_1D_SDE", "Dataset=HUSTStrict", f"only_source={source}", "gpu_id=0", "process_wandb=False", f"++hust_protocol_split={split}", f"++Dataset.data_path={data_path}", f"++hust_checkpoint_root={root}", "num_workers=4"]
    if kind == "beginning":
        route = "0711_robust" if variant == "robust" else "dtcc_ordinary"
        command = build_command(config, route, {}, task, 2025, gpu)
        command.append("++beginning_only=True")
        if load_split:
            command = [arg.replace(str(protocol["data_path"]), str(data_path)).replace(str(checkpoints["root"]), str(root)) for arg in command]
            command = [arg.replace("++hust_protocol_split=bearing", "++hust_protocol_split=load") for arg in command]
        return command
    raise ValueError(f"unknown stage command kind: {kind}")


def _planned_target(
    config: Mapping[str, object], stage: str, route: str, task: tuple[int, int],
    stream_seed: int, overrides: Mapping[str, object], *, candidate: str | None = None,
    beginning_only: bool = False, load_split: bool = False,
) -> dict[str, object]:
    variant = "robust" if route == "0711_robust" else "ordinary"
    effective_id = candidate or candidate_id(overrides, int(config["protocol"]["source_seed"]))
    effective_config = {
        "route": route, "variant": variant,
        "protocol": {
            "dataset": config["protocol"]["dataset"],
            "model": config["protocol"]["model"],
            "data_path": config["protocol"]["load_data_path" if load_split else "data_path"],
            "checkpoint_root": config["checkpoints"]["load_root" if load_split else "root"],
            "batch_size": 128,
            "num_workers": 4,
            "passes": 1,
        },
        "fixed": dict(config["fixed_overrides"]),
        "baseline": dict(config["baseline_overrides"]),
        "candidate": normalize_overrides(overrides),
    }
    return {
        "kind": "target", "stage": stage, "route": route, "variant": variant,
        "source": int(task[0]), "target": int(task[1]), "task": tuple(task),
        "source_seed": int(config["protocol"]["source_seed"]),
        "stream_seed": int(stream_seed), "overrides": normalize_overrides(overrides),
        "candidate_id": effective_id, "config_sha256": _sha256_bytes(_canonical(effective_config)),
        "beginning_only": bool(beginning_only), "load_split": bool(load_split),
        "expected_result_kind": "beginning" if beginning_only else "target",
    }


def _planned_source(config: Mapping[str, object], stage: str, variant: str, source: int, *, load_split: bool = False) -> dict[str, object]:
    return {
        "kind": "source", "stage": stage, "route": f"source_{variant}",
        "variant": variant, "source": int(source), "source_seed": 2025,
        "candidate_id": f"source-{variant}-{source}-2025", "overrides": {},
        "config_sha256": _sha256_bytes(_canonical({
            "variant": variant,
            "source": source,
            "seed": 2025,
            "load_split": load_split,
            "num_workers": 4,
            "cache_path": config["protocol"]["load_data_path" if load_split else "data_path"],
        })),
        "load_split": bool(load_split), "expected_result_kind": "source",
    }


def _declare_job_evidence(config: Mapping[str, object], job: Mapping[str, object], *, resolve: bool = True) -> dict[str, object]:
    """Declare every input/output artifact; missing files remain explicit for dry planning."""
    value = dict(job)
    load_split = bool(value.get("load_split"))
    data_root = Path(str(config["protocol"]["load_data_path"] if load_split else config["protocol"]["data_path"]))
    checkpoint_root = Path(str(config["checkpoints"]["load_root"] if load_split else config["checkpoints"]["root"]))
    cache_manifest = data_root / "manifest.json"
    cache_content_sha256 = None
    cache_tensor_sha256s: dict[str, str | None] = {
        str((data_root / f"domain_{domain}.pt").resolve()): None
        for domain in range(3 if load_split else 4)
    }
    cache_manifest_sha256 = _planned_hash(cache_manifest) if resolve else None
    if resolve and cache_manifest.is_file():
        try:
            from Lib.hust_strict_protocol import validate_cache
        except ModuleNotFoundError:
            from hust_strict_protocol import validate_cache
        cache = validate_cache(data_root)
        if cache.get("version") != 2 or not cache.get("content_sha256"):
            raise ValueError("formal jobs require a version-2 content-bound cache")
        cache_manifest_sha256 = str(cache["manifest_sha256"])
        cache_content_sha256 = str(cache["content_sha256"])
        cache_tensor_sha256s = {
            str((data_root / str(summary["tensor_file"])).resolve()): str(summary["tensor_sha256"])
            for summary in cache["domains"].values()
        }
    checkpoint_dir = checkpoint_root / str(value["variant"]) / f"source_{value['source']}" / "seed_2025"
    best_checkpoint = checkpoint_dir / "best_source_ResNet18_1D_SDE2025fft_Linear.pt"
    final_checkpoint = checkpoint_dir / "ResNet18_1D_SDE2025fft_Linear.pt"
    checkpoint = best_checkpoint if best_checkpoint.is_file() else final_checkpoint
    summary = checkpoint_dir / "source_training_summary.json"
    runner = Path(str(config["routes"][str(value["route"])]))
    experiment_config = Path("Configs/Experiments/HUST0711_strict_tuning.yaml")
    value.update(
        cache_manifest_path=str(cache_manifest),
        cache_manifest_sha256=cache_manifest_sha256,
        cache_content_sha256=cache_content_sha256,
        cache_tensor_sha256s=cache_tensor_sha256s,
        source_checkpoint_path=str(checkpoint),
        source_summary_path=str(summary),
        source_checkpoint_sha256=_planned_hash(checkpoint) if resolve else None,
        source_summary_sha256=_planned_hash(summary) if resolve else None,
        expected_result_contract=str(value["expected_result_kind"]),
        runner_script_path=str(runner),
        runner_script_sha256=_planned_hash(runner) if resolve else None,
        experiment_config_path=str(experiment_config),
        experiment_config_sha256=_planned_hash(experiment_config) if resolve else None,
        freeze_config_path=value.get("_freeze_config_path"),
        freeze_config_sha256=value.get("_freeze_config_sha256"),
        freeze_sha256=value.get("_freeze_sha256"),
        freeze_proof_path=value.get("_freeze_proof_path"),
        freeze_proof_file_sha256=value.get("_freeze_proof_file_sha256"),
        freeze_proof_sha256=value.get("_freeze_proof_sha256"),
        freeze_frozen_at=value.get("_freeze_frozen_at"),
    )
    required_inputs = [cache_manifest, *map(Path, cache_tensor_sha256s), runner, experiment_config]
    if value["kind"] != "source":
        required_inputs += [checkpoint, summary]
    if value.get("freeze_config_path"):
        required_inputs += [Path(str(value["freeze_config_path"])), Path(str(value["freeze_proof_path"]))]
    value["evidence_status"] = "resolved" if resolve and all(path.is_file() for path in required_inputs) else "unresolved"
    value["artifacts"] = [str(path) for path in required_inputs]
    return value


def _declared(config: Mapping[str, object], jobs: Sequence[Mapping[str, object]], *, resolve: bool = True) -> list[dict[str, object]]:
    return [_declare_job_evidence(config, job, resolve=resolve) for job in jobs]


def plan_stage_jobs(config: Mapping[str, object], stage: str, run_dir: Path, *, resolve_artifacts: bool = True) -> list[dict[str, object]]:
    """Return the exact immutable jobs belonging to one formal stage."""
    validate_config(config)
    if stage == "source":
        return _declared(config, [_planned_source(config, "source", variant, source) for variant in ("ordinary", "robust") for source in range(4)], resolve=resolve_artifacts)
    if stage == "beginning":
        return _declared(config, [_planned_target(config, "beginning", "0711_robust" if variant == "robust" else "dtcc_ordinary", task, 2025, {}, candidate=f"beginning-{variant}", beginning_only=True) for variant in ("ordinary", "robust") for task in ALL_TASKS], resolve=resolve_artifacts)
    if stage == "baseline":
        jobs = []
        for route in ("dtcc_ordinary", "0711_robust", "0711_common"):
            candidate = "untuned" if route == "0711_robust" else route
            jobs.extend(_planned_target(config, "baseline", route, task, 2025, {}, candidate=candidate) for task in ALL_TASKS)
        return _declared(config, jobs, resolve=resolve_artifacts)
    if stage == "tune":
        jobs = []
        anchor = dict(config["baseline_overrides"])
        initial = dict(config["search"]["initial_candidate"]["overrides"])
        for group_index, group in enumerate(config["search"]["groups"]):
            values = list(group["values"])
            if group_index == 0:
                values.append(initial)
            for overrides in expand_coordinate_group(anchor, values):
                cid = candidate_id(overrides, 2025)
                jobs.extend(_planned_target(config, f"tune_group_{group_index:02d}_{group['name']}", "0711_robust", task, 2025, overrides, candidate=cid) for task in DEV_TASKS)
        return _declared(config, jobs, resolve=resolve_artifacts)
    if stage == "final":
        freeze_path = Path(run_dir) / "best_config.yaml"
        frozen = load_frozen_candidate(freeze_path)
        frozen_id, overrides = str(frozen["candidate_id"]), dict(frozen["overrides"])
        validate_candidate_overrides(config, frozen_id, overrides)
        proof_path = Path(str(frozen["proof_path"]))
        binding = {
            "_freeze_config_path": str(freeze_path.resolve()),
            "_freeze_config_sha256": _sha256_file(freeze_path),
            "_freeze_sha256": str(frozen["freeze_sha256"]),
            "_freeze_proof_path": str(proof_path.resolve()),
            "_freeze_proof_file_sha256": _sha256_file(proof_path),
            "_freeze_proof_sha256": str(frozen["proof_sha256"]),
            "_freeze_frozen_at": float(frozen["frozen_at"]),
        }
        jobs = [{**_planned_target(config, "heldout", "0711_robust", task, 2025, overrides, candidate=frozen_id), **binding} for task in tuple(tuple(x) for x in config["tasks"]["heldout"])]
        stability_candidates = (
            [("untuned", overrides, binding)]
            if frozen_id == "untuned"
            else [
                ("untuned", dict(config["baseline_overrides"]), {}),
                (frozen_id, overrides, binding),
            ]
        )
        nondevelopment_tasks = tuple(tuple(x) for x in config["tasks"]["heldout"])
        for cid, values, candidate_binding in stability_candidates:
            jobs.extend({**_planned_target(config, "stability", "0711_robust", task, 2026, values, candidate=cid), **candidate_binding} for task in nondevelopment_tasks)
        return _declared(config, jobs, resolve=resolve_artifacts)
    if stage == "load-audit":
        domains = (0, 1, 2)
        tasks = tuple((s, t) for s in domains for t in domains if s != t)
        jobs = [_planned_source(config, "load-audit-source", variant, source, load_split=True) for variant in ("ordinary", "robust") for source in domains]
        jobs += [_planned_target(config, "load-audit-beginning", "0711_robust" if variant == "robust" else "dtcc_ordinary", task, 2025, {}, candidate=f"load-beginning-{variant}", beginning_only=True, load_split=True) for variant in ("ordinary", "robust") for task in tasks]
        return _declared(config, jobs, resolve=resolve_artifacts)
    if stage == "report":
        return []
    if stage == "all":
        # Tune/final are state-dependent and are planned by the executor only after prior gates.
        return plan_stage_jobs(config, "source", run_dir, resolve_artifacts=resolve_artifacts) + plan_stage_jobs(config, "beginning", run_dir, resolve_artifacts=resolve_artifacts) + plan_stage_jobs(config, "baseline", run_dir, resolve_artifacts=resolve_artifacts)
    raise ValueError(f"unknown stage: {stage}")


def _default_run(command: list[str], log_path: Path, env: Mapping[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        return subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=dict(env), check=False).returncode


def _parse_runner_log(path: Path, route: str) -> dict[str, object]:
    try:
        from tools.summarize_hust_dtcc_0711 import parse_runner_log
    except ModuleNotFoundError:  # Direct execution places tools/, not the repo root, on sys.path.
        from summarize_hust_dtcc_0711 import parse_runner_log
    return parse_runner_log(path, route)


def _target_command_for_job(config: Mapping[str, object], job: Mapping[str, object], gpu: int) -> list[str]:
    command = build_command(config, str(job["route"]), dict(job.get("overrides", {})), tuple(job["task"]), int(job.get("stream_seed", 2025)), gpu)
    if job.get("load_split"):
        command = [arg.replace(str(config["protocol"]["data_path"]), str(config["protocol"]["load_data_path"])).replace(str(config["checkpoints"]["root"]), str(config["checkpoints"]["load_root"])) for arg in command]
        command = [arg.replace("++hust_protocol_split=bearing", "++hust_protocol_split=load") for arg in command]
    if job.get("beginning_only"):
        command.append("++beginning_only=True")
    if job.get("candidate_id") and job.get("config_sha256"):
        command += [f"++hust_candidate_id={job['candidate_id']}", f"++hust_config_sha256={job['config_sha256']}"]
    return command


def _job_identity(config: Mapping[str, object], job: Mapping[str, object]) -> tuple[str, str]:
    route = str(job["route"])
    cid = str(job.get("candidate_id") or candidate_id(job.get("overrides", {}), int(config["protocol"]["source_seed"])))
    stage = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(job.get("stage", "task")))
    if job.get("kind") == "source":
        return cid, f"{stage}_{route}_source{int(job['source'])}_seed2025"
    task = tuple(job["task"])
    seed = int(job.get("stream_seed", 2025))
    return cid, f"{stage}_{route}_{cid}_{task[0]}to{task[1]}_seed{seed}"


def _artifact_hashes(paths: Sequence[object]) -> dict[str, str]:
    hashes = {}
    for value in paths:
        path = Path(str(value)).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"required artifact missing: {path}")
        hashes[str(path)] = _sha256_file(path)
    return hashes


def _valid_success(
    record: Mapping[str, object], command: list[str], artifacts: Sequence[object],
    job: Mapping[str, object] | None = None,
) -> bool:
    if job is not None and job.get("kind") == "target":
        if not _DECLARED_TARGET_JOB_FIELDS <= set(job) or not _DECLARED_TARGET_JOB_FIELDS <= set(record):
            return False
        if any(_canonical(record[key]) != _canonical(job[key]) for key in _DECLARED_TARGET_JOB_FIELDS):
            return False
        try:
            from tools.summarize_hust_dtcc_0711 import _validate_formal_state
        except ModuleNotFoundError:
            from summarize_hust_dtcc_0711 import _validate_formal_state
        try:
            _validate_formal_state(record)
        except (KeyError, OSError, TypeError, ValueError):
            return False
    if record.get("status") != "succeeded" or record.get("returncode") != 0:
        return False
    if record.get("command_sha256") != _sha256_bytes(_canonical(command)):
        return False
    path = Path(str(record.get("log_path", "")))
    if not path.is_file() or record.get("log_sha256") != _sha256_file(path):
        return False
    command_path = Path(str(record.get("command_path", "")))
    if not command_path.is_file() or record.get("command_log_sha256") != _sha256_file(command_path):
        return False
    try:
        if record.get("artifact_hashes", {}) != _artifact_hashes(artifacts):
            return False
    except FileNotFoundError:
        return False
    try:
        parsed = _parse_runner_log(path, str(record["route"])) if not record.get("result_identity") else _parse_runner_log_with_expected(path, str(record["route"]), record["result_identity"])
    except (OSError, ValueError):
        return False
    return parsed == record.get("metrics")


def _source_output(job: Mapping[str, object]) -> tuple[dict[str, object], dict[str, str]]:
    checkpoint = Path(str(job["source_checkpoint_path"])).resolve()
    summary_path = Path(str(job["source_summary_path"])).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {"route": job["variant"], "source": int(job["source"]), "seed": 2025}
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError("source summary identity mismatch")
    checkpoint_hash = _sha256_file(checkpoint)
    if summary.get("checkpoint_sha256") != checkpoint_hash or summary.get("target_labels_consumed") is not False:
        raise ValueError("source checkpoint/summary contract mismatch")
    cache_identity = summary.get("cache_identity")
    expected_tensors = {
        str(Path(str(path)).resolve()): str(digest)
        for path, digest in dict(job.get("cache_tensor_sha256s", {})).items()
    }
    if (
        not isinstance(cache_identity, Mapping)
        or cache_identity.get("mode") != "formal"
        or str(Path(str(cache_identity.get("manifest_path"))).resolve())
        != str(Path(str(job.get("cache_manifest_path"))).resolve())
        or cache_identity.get("manifest_sha256") != job.get("cache_manifest_sha256")
        or cache_identity.get("content_sha256") != job.get("cache_content_sha256")
        or {
            str(Path(str(path)).resolve()): str(digest)
            for path, digest in dict(cache_identity.get("tensor_sha256s", {})).items()
        } != expected_tensors
    ):
        raise ValueError("source checkpoint cache content identity mismatch")
    return summary, {str(checkpoint): checkpoint_hash, str(summary_path): _sha256_file(summary_path)}


def _valid_source_success(record: Mapping[str, object], command: list[str], artifacts: Sequence[object], job: Mapping[str, object]) -> bool:
    if record.get("status") != "succeeded" or record.get("returncode") != 0 or record.get("command_sha256") != _sha256_bytes(_canonical(command)):
        return False
    log_path = Path(str(record.get("log_path", "")))
    command_path = Path(str(record.get("command_path", "")))
    if (not log_path.is_file() or record.get("log_sha256") != _sha256_file(log_path)
            or not command_path.is_file() or record.get("command_log_sha256") != _sha256_file(command_path)):
        return False
    try:
        if record.get("artifact_hashes") != _artifact_hashes(artifacts):
            return False
        summary, outputs = _source_output(job)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return record.get("metrics") == summary and record.get("output_artifact_hashes") == outputs


def _execute_source_one(run_dir: Path, config: Mapping[str, object], job: Mapping[str, object], gpu: int, run_process, *, retry_failed: bool) -> dict[str, object]:
    cid, stem = _job_identity(config, job)
    command = build_stage_command(config, "source", source=int(job["source"]), variant=str(job["variant"]), load_split=bool(job.get("load_split")), gpu=gpu)
    command += [f"++hust_candidate_id={cid}", f"++hust_config_sha256={job['config_sha256']}"]
    artifacts = list(job["artifacts"])
    state_path = Path(run_dir) / "state" / f"{stem}.json"
    existing = json.loads(state_path.read_text()) if state_path.is_file() else None
    if existing and _valid_source_success(existing, command, artifacts, job):
        return existing
    if existing and existing.get("status") == "failed" and not retry_failed:
        return existing
    if existing and existing.get("status") == "running" and int(existing.get("attempt", 0)) >= 2 and not retry_failed:
        exhausted = {**existing, "status": "failed", "failure_class": "stale_resume_exhausted", "ended_at": time.time()}
        atomic_write_json(state_path, exhausted)
        return exhausted
    attempt = int(existing.get("attempt", 0)) + 1 if existing else 1
    environment = os.environ.copy()
    environment.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu))
    automatic_retry_available = True
    while True:
        log_path = (Path(run_dir) / "logs" / f"{stem}.attempt-{attempt}.log").resolve()
        command_path = Path(run_dir) / "commands" / f"{stem}.attempt-{attempt}.txt"
        if log_path.exists() or command_path.exists():
            raise FileExistsError(f"immutable source attempt exists: {log_path}")
        command_path.parent.mkdir(parents=True, exist_ok=True)
        command_path.write_text(shlex.join(command) + "\n", encoding="utf-8")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        running = {**dict(job), "candidate_id": cid, "command": command, "command_sha256": _sha256_bytes(_canonical(command)), "command_path": str(command_path.resolve()), "command_log_sha256": _sha256_file(command_path), "artifact_hashes": _artifact_hashes(artifacts), "gpu": gpu, "status": "running", "attempt": attempt, "started_at": time.time(), "ended_at": None, "returncode": None, "log_path": str(log_path), "metrics": None}
        atomic_write_json(state_path, running)
        returncode = int(run_process(command, log_path, environment))
        parse_error = None
        summary = outputs = None
        if returncode == 0:
            try:
                summary, outputs = _source_output(job)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                parse_error = str(exc)
        if summary is not None:
            checkpoint_path = str(Path(str(job["source_checkpoint_path"])).resolve())
            summary_path = str(Path(str(job["source_summary_path"])).resolve())
            result = {**running, "status": "succeeded", "returncode": 0, "ended_at": time.time(), "metrics": summary, "output_artifact_hashes": outputs, "source_checkpoint_sha256": outputs[checkpoint_path], "source_summary_sha256": outputs[summary_path], "log_sha256": _sha256_file(log_path)}
            atomic_write_json(state_path, result)
            return result
        text = log_path.read_text(errors="ignore") if log_path.exists() else ""
        transient = returncode != 0 and any(marker in text.casefold() for marker in ("cuda initialization error", "cuda driver initialization failed", "cuda error: system not yet initialized"))
        if not transient or not automatic_retry_available:
            result = {**running, "status": "failed", "returncode": returncode, "ended_at": time.time(), "failure_class": "transient" if transient else "permanent", "parse_error": parse_error}
            atomic_write_json(state_path, result)
            return result
        automatic_retry_available = False
        attempt += 1


def _parse_runner_log_with_expected(path: Path, route: str, expected: Mapping[str, object]) -> dict[str, object]:
    try:
        from tools.summarize_hust_dtcc_0711 import parse_runner_log
    except ModuleNotFoundError:
        from summarize_hust_dtcc_0711 import parse_runner_log
    return parse_runner_log(path, route, expected=expected)


def _resolve_target_evidence(config: Mapping[str, object], job: Mapping[str, object]) -> dict[str, object]:
    """Rebuild and validate target inputs at launch, after deferred sources may exist."""
    resolved = _declare_job_evidence(config, job)
    try:
        if resolved.get("evidence_status") != "resolved":
            raise ValueError("required artifacts are unresolved")
        _summary, outputs = _source_output(resolved)
        checkpoint = str(Path(str(resolved["source_checkpoint_path"])).resolve())
        summary = str(Path(str(resolved["source_summary_path"])).resolve())
        if (resolved.get("source_checkpoint_sha256") != outputs[checkpoint]
                or resolved.get("source_summary_sha256") != outputs[summary]):
            raise ValueError("declared source hashes do not match validated artifacts")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"target source evidence unavailable or invalid for "
            f"{resolved.get('variant')} source {resolved.get('source')}: {exc}"
        ) from exc
    return resolved


def _execute_one(run_dir: Path, config: Mapping[str, object], job: Mapping[str, object], gpu: int, run_process: Callable[[list[str], Path, Mapping[str, str]], int], *, retry_failed: bool = False) -> dict[str, object]:
    cid, stem = _job_identity(config, job)
    state_path = run_dir / "state" / f"{stem}.json"
    existing = json.loads(state_path.read_text()) if state_path.is_file() else None
    if existing and existing.get("status") == "failed" and not retry_failed:
        return existing
    if existing and existing.get("status") == "running" and int(existing.get("attempt", 0)) >= 2 and not retry_failed:
        exhausted = {**existing, "status": "failed", "failure_class": "stale_resume_exhausted", "ended_at": time.time()}
        atomic_write_json(state_path, exhausted)
        return exhausted
    if job.get("kind") == "target":
        job = _resolve_target_evidence(config, job)
        resolved_cid, resolved_stem = _job_identity(config, job)
        if (resolved_cid, resolved_stem) != (cid, stem):
            raise ValueError("resolved target evidence changed the job identity")
    route, task = str(job["route"]), tuple(job["task"])
    seed, overrides = int(job.get("stream_seed", 2025)), dict(job.get("overrides", {}))
    artifacts = list(job.get("artifacts", []))
    command = _target_command_for_job(config, {**job, "candidate_id": cid}, gpu)
    if existing:
        if _valid_success(existing, command, artifacts, job):
            return existing
        attempt = int(existing.get("attempt", 0)) + 1
    else:
        attempt = 1
    environment = os.environ.copy()
    environment.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu))
    result_identity = {
        "candidate_id": cid, "route": route, "task": list(task), "source": int(task[0]),
        "target": int(task[1]), "source_seed": int(config["protocol"]["source_seed"]),
        "stream_seed": seed, "variant": str(job.get("variant", "robust" if route == "0711_robust" else "ordinary")),
        "result_kind": str(job.get("expected_result_kind", "target")),
    }
    if job.get("config_sha256"):
        result_identity["config_sha256"] = job["config_sha256"]
    if job.get("source_checkpoint_sha256"):
        result_identity["source_checkpoint_sha256"] = job["source_checkpoint_sha256"]
    declared_job = {key: value for key, value in job.items() if key in _DECLARED_TARGET_JOB_FIELDS}
    running = {
        **declared_job,
        "candidate_id": cid,
        "stage": str(job.get("stage", "task")),
        "route": route,
        "variant": str(job.get("variant", "robust" if route == "0711_robust" else "ordinary")),
        "task": list(task),
        "source": int(task[0]),
        "target": int(task[1]),
        "source_seed": int(config["protocol"]["source_seed"]),
        "stream_seed": seed,
        "overrides": normalize_overrides(overrides),
        "config_sha256": job.get("config_sha256"),
        "command": command,
        "command_sha256": _sha256_bytes(_canonical(command)),
        "command_path": None,
        "command_log_sha256": None,
        "artifact_hashes": _artifact_hashes(artifacts),
        "result_identity": result_identity if job.get("candidate_id") else None,
        "gpu": gpu,
        "status": "running",
        "attempt": attempt,
        "started_at": time.time(),
        "ended_at": None,
        "returncode": None,
        "log_path": None,
        "log_sha256": None,
        "metrics": None,
        "strict_online": None,
        "runtime_seconds": None,
        "failure_class": None,
        "parse_error": None,
    }
    automatic_retry_available = True
    current_attempt = attempt
    while True:
        log_path = (run_dir / "logs" / f"{stem}.attempt-{current_attempt}.log").resolve()
        command_path = run_dir / "commands" / f"{stem}.attempt-{current_attempt}.txt"
        if log_path.exists() or command_path.exists():
            raise FileExistsError(f"immutable attempt artifact already exists: {log_path}")
        command_path.parent.mkdir(parents=True, exist_ok=True)
        command_path.write_text(shlex.join(command) + "\n", encoding="utf-8")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        running.update(attempt=current_attempt, log_path=str(log_path), command_path=str(command_path.resolve()), command_log_sha256=_sha256_file(command_path), started_at=time.time())
        atomic_write_json(state_path, running)
        returncode = int(run_process(command, log_path, environment))
        metrics = None
        parse_error = None
        if returncode == 0:
            try:
                metrics = _parse_runner_log(log_path, route) if running["result_identity"] is None else _parse_runner_log_with_expected(log_path, route, running["result_identity"])
            except (OSError, ValueError) as exc:
                parse_error = str(exc)
        if returncode == 0 and metrics is not None:
            parsed_state = {key: value for key, value in metrics.items() if key in _PARSER_TARGET_STATE_FIELDS}
            result = {**running, **parsed_state, "status": "succeeded", "ended_at": time.time(), "returncode": 0, "metrics": metrics, "log_sha256": _sha256_file(log_path)}
            atomic_write_json(state_path, result)
            return result
        text = log_path.read_text(errors="ignore") if log_path.exists() else ""
        transient = returncode != 0 and any(marker in text.casefold() for marker in ("cuda initialization error", "cuda driver initialization failed", "cuda error: system not yet initialized"))
        if not transient or not automatic_retry_available:
            result = {**running, "attempt": current_attempt, "status": "failed", "ended_at": time.time(), "returncode": returncode, "failure_class": "transient" if transient else "permanent", "parse_error": parse_error}
            atomic_write_json(state_path, result)
            return result
        automatic_retry_available = False
        current_attempt += 1


def _execution_phases(jobs: Sequence[Mapping[str, object]]) -> tuple[list[list[tuple[int, Mapping[str, object]]]], bool]:
    indexed = list(enumerate(jobs))
    sources = [
        item for item in indexed
        if item[1].get("kind") == "source"
        and item[1].get("stage") == "load-audit-source"
        and item[1].get("load_split") is True
    ]
    targets = [
        item for item in indexed
        if item[1].get("kind") == "target"
        and item[1].get("stage") == "load-audit-beginning"
        and item[1].get("load_split") is True
    ]
    if not sources or not targets:
        return [indexed], False
    expected_sources = {(variant, source) for variant in ("ordinary", "robust") for source in range(3)}
    expected_targets = {
        (variant, source, target)
        for variant in ("ordinary", "robust")
        for source in range(3)
        for target in range(3)
        if source != target
    }
    source_inventory = [(str(job.get("variant")), int(job.get("source", -1))) for _index, job in sources]
    target_inventory = [
        (str(job.get("variant")), int(job.get("source", -1)), int(job.get("target", -1)))
        for _index, job in targets
    ]
    if (len(sources) != 6 or set(source_inventory) != expected_sources
            or len(targets) != 12 or set(target_inventory) != expected_targets
            or len(sources) + len(targets) != len(indexed)):
        raise ValueError("load-audit dependency inventory must contain exactly 6 sources and 12 targets")
    return [sources, targets], True


def execute_parallel_tasks(run_dir: Path, config: Mapping[str, object], jobs: Sequence[Mapping[str, object]], gpus: Sequence[int], *, run_process: Callable[[list[str], Path, Mapping[str, str]], int] | None = None, deadline: float | None = None, retry_failed: bool = False) -> list[dict[str, object]]:
    validate_config(config)
    gpus = list(dict.fromkeys(int(gpu) for gpu in gpus))
    if not gpus:
        raise RuntimeError("no eligible GPU")
    if deadline is not None:
        reserve = float(config["budget"]["reserve_minutes"]) * 60
        estimate = float(config["budget"]["estimated_task_minutes"]) * 60
        if time.time() + reserve + estimate > deadline:
            raise TimeoutError("24-hour wall budget reserve reached")
    active_run = run_process or _default_run
    run_dir = Path(run_dir)

    phases, load_audit_dependency = _execution_phases(jobs)
    formal_run = run_process is None

    completed_records: dict[int, dict[str, object]] = {}
    for phase_index, phase in enumerate(phases):
        pending = deque(phase)
        changed = Condition()
        stop_requested = [False]
        active_launch_checks = [0]

        def stop_locked() -> None:
            stop_requested[0] = True
            changed.notify_all()

        def check_launch_budget() -> None:
            if deadline is None:
                return
            reserve = float(config["budget"]["reserve_minutes"]) * 60
            estimate = float(config["budget"]["estimated_task_minutes"]) * 60
            if time.time() + reserve + estimate > deadline:
                raise TimeoutError("24-hour wall budget reserve reached before launch")

        def gpu_worker(gpu: int) -> None:
            while True:
                with changed:
                    if stop_requested[0] or not pending:
                        return
                    active_launch_checks[0] += 1

                try:
                    check_launch_budget()
                    eligible = not formal_run or _gpu_is_idle(
                        gpu,
                        float(config["protocol"]["max_gpu_utilization"]),
                        float(config["protocol"]["max_gpu_memory_fraction"]),
                    )
                except BaseException:
                    with changed:
                        active_launch_checks[0] -= 1
                        stop_locked()
                    raise

                with changed:
                    active_launch_checks[0] -= 1
                    changed.notify_all()
                    while active_launch_checks[0] and not stop_requested[0]:
                        changed.wait()
                    if stop_requested[0] or not pending:
                        return
                    if not eligible:
                        changed.wait(timeout=5)
                        continue
                    try:
                        check_launch_budget()
                    except BaseException:
                        stop_locked()
                        raise
                    original_index, job = pending.popleft()
                    changed.notify_all()

                try:
                    if job.get("kind") == "source":
                        record = _execute_source_one(
                            run_dir, config, job, gpu, active_run, retry_failed=retry_failed
                        )
                    else:
                        record = _execute_one(
                            run_dir, config, job, gpu, active_run, retry_failed=retry_failed
                        )
                except BaseException:
                    with changed:
                        stop_locked()
                    raise
                with changed:
                    completed_records[original_index] = record
                    changed.notify_all()

        with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            list(executor.map(gpu_worker, gpus))
        phase_records = [completed_records[original_index] for original_index, _job in phase]
        if load_audit_dependency and phase_index == 0:
            failed = [
                _job_identity(config, record)[1]
                for record in phase_records
                if record.get("status") != "succeeded"
            ]
            if failed:
                raise RuntimeError(f"load-audit source phase failed; targets were not launched: {failed}")
    return [completed_records[index] for index in range(len(jobs))]


def resume_pipeline(run_dir: Path, config: Mapping[str, object], jobs: Sequence[Mapping[str, object]], gpus: Sequence[int], *, run_process=None, retry_failed: bool = False) -> list[dict[str, object]]:
    return execute_parallel_tasks(run_dir, config, jobs, gpus, run_process=run_process, retry_failed=retry_failed)


def rank_candidates(
    rows: Sequence[Mapping[str, object]], required_tasks: set[tuple[int, int]],
    baseline: Mapping[tuple[int, int], float], minimum_gain: float = 0.30,
    maximum_regression: float = 1.00, *, required_seeds: set[int] | None = None,
    expected_stages: Mapping[int, str] | None = None, expected_route: str = "0711_robust",
    expected_variant: str = "robust",
    eligibility_seeds: set[int] | None = None,
) -> list[dict[str, object]]:
    seeds = required_seeds or {2025}
    if expected_stages is None:
        inferred_stages: dict[int, str] = {}
        for seed in seeds:
            stages = {
                row.get("stage")
                for row in rows
                if row.get("stream_seed") == seed
            }
            if len(stages) != 1 or not all(isinstance(stage, str) and stage for stage in stages):
                return []
            inferred_stages[seed] = str(next(iter(stages)))
        expected_stages = inferred_stages
    def baseline_score(task, seed):
        return float(baseline[(task, seed)] if (task, seed) in baseline else baseline[task])
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["candidate_id"]), []).append(row)
    ranked = []
    for cid, candidate_rows in grouped.items():
        expected_keys = {(task, seed) for task in required_tasks for seed in seeds}
        valid_identity = all(
            row.get("status") == "succeeded"
            and row.get("route") == expected_route
            and row.get("variant") == expected_variant
            and type(row.get("source_seed")) is int
            and row.get("source_seed") == 2025
            and isinstance(row.get("config_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", str(row["config_sha256"])) is not None
            and int(row.get("stream_seed", -1)) in seeds
            and str(row.get("stage")) == expected_stages.get(int(row.get("stream_seed", -1)))
            and tuple(row.get("task", ())) in required_tasks
            and int(row.get("source", -1)) == int(row["task"][0])
            and int(row.get("target", -1)) == int(row["task"][1])
            for row in candidate_rows
        )
        keys = [(tuple(row["task"]), int(row["stream_seed"])) for row in candidate_rows] if valid_identity else []
        if not valid_identity or len(keys) != len(expected_keys) or len(keys) != len(set(keys)) or set(keys) != expected_keys:
            continue
        config_hashes = {str(row["config_sha256"]) for row in candidate_rows}
        overrides = {_sha256_bytes(_canonical(normalize_overrides(row.get("overrides", {})))) for row in candidate_rows}
        if len(config_hashes) != 1 or len(overrides) != 1:
            continue
        successful = list(candidate_rows)
        by_key = {key: row for key, row in zip(keys, successful, strict=True)}
        scores_by_task = {task: sum(float(by_key[(task, seed)]["strict_online"]) for seed in seeds) / len(seeds) for task in required_tasks}
        mean = sum(scores_by_task.values()) / len(scores_by_task)
        deltas = [float(by_key[(task, seed)]["strict_online"]) - baseline_score(task, seed) for task in required_tasks for seed in seeds] if baseline else list(scores_by_task.values())
        if baseline:
            rejected = False
            for seed in (eligibility_seeds if eligibility_seeds is not None else seeds):
                candidate_seed_mean = sum(float(by_key[(task, seed)]["strict_online"]) for task in required_tasks) / len(required_tasks)
                baseline_seed_mean = sum(baseline_score(task, seed) for task in required_tasks) / len(required_tasks)
                seed_deltas = [float(by_key[(task, seed)]["strict_online"]) - baseline_score(task, seed) for task in required_tasks]
                if candidate_seed_mean - baseline_seed_mean < minimum_gain or min(seed_deltas) < -maximum_regression:
                    rejected = True
                    break
            if rejected:
                continue
        runtime = sum(float(by_key[(task, seed)].get("runtime_seconds", 0.0)) for task in required_tasks for seed in seeds)
        ranked.append({"candidate_id": cid, "mean_strict_online": mean, "minimum_task_delta": min(deltas), "runtime_seconds": runtime, "overrides": dict(successful[0].get("overrides", {})), "stages": {int(seed): expected_stages[int(seed)] for seed in seeds}})
    return sorted(ranked, key=lambda row: (-row["mean_strict_online"], -row["minimum_task_delta"], row["runtime_seconds"], row["candidate_id"]))


def rank_coordinate_candidates_globally(
    rows: Sequence[Mapping[str, object]],
    required_tasks: set[tuple[int, int]],
    baseline: Mapping[tuple[int, int], float],
    *,
    minimum_gain: float,
    maximum_regression: float,
) -> list[dict[str, object]]:
    """Rank every complete eligible coordinate candidate once, across all groups."""
    by_stage: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        stage = str(row.get("stage", ""))
        if stage.startswith("tune_group_"):
            by_stage.setdefault(stage, []).append(row)
    observations: dict[str, list[dict[str, object]]] = {}
    for stage, stage_rows in sorted(by_stage.items()):
        for ranked in rank_candidates(
            stage_rows,
            required_tasks,
            baseline,
            minimum_gain,
            maximum_regression,
            expected_stages={2025: stage},
            eligibility_seeds={2025},
        ):
            observations.setdefault(str(ranked["candidate_id"]), []).append(ranked)
    deduplicated = []
    for identifier, entries in observations.items():
        signatures = {
            _canonical({
                "mean": row["mean_strict_online"],
                "minimum_delta": row["minimum_task_delta"],
                "overrides": normalize_overrides(row["overrides"]),
            })
            for row in entries
        }
        if len(signatures) != 1:
            continue
        chosen = min(entries, key=lambda row: str(dict(row["stages"])[2025]))
        deduplicated.append(chosen)
    return sorted(
        deduplicated,
        key=lambda row: (
            -row["mean_strict_online"],
            -row["minimum_task_delta"],
            row["runtime_seconds"],
            row["candidate_id"],
        ),
    )


def _write_yaml_atomic(path: Path, value: Mapping[str, object]) -> None:
    serialized = yaml.safe_dump(dict(value), sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _formal_evidence_reference(
    run_dir: Path, config: Mapping[str, object], row: Mapping[str, object]
) -> dict[str, object]:
    _cid, stem = _job_identity(config, row)
    state_path = Path(run_dir) / "state" / f"{stem}.json"
    if not state_path.is_file():
        raise ValueError(f"authoritative candidate state missing: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    identity = (
        "kind", "stage", "route", "variant", "candidate_id", "source",
        "source_seed", "stream_seed", "config_sha256", "status",
    )
    if any(state.get(key) != row.get(key) for key in identity) or state.get("status") != "succeeded":
        raise ValueError(f"authoritative candidate state identity mismatch: {state_path}")
    if state.get("kind") == "target" and tuple(state.get("task", ())) != tuple(row.get("task", ())):
        raise ValueError(f"authoritative candidate task mismatch: {state_path}")
    command = state.get("command")
    expected_command = _target_command_for_job(
        config, state, int(state.get("gpu", 0))
    )
    if (
        not isinstance(command, list)
        or command != expected_command
        or state.get("command_sha256") != _sha256_bytes(_canonical(command))
    ):
        raise ValueError(f"authoritative candidate command mismatch: {state_path}")
    references = {
        "state_path": str(state_path.resolve()),
        "state_sha256": _sha256_file(state_path),
        "stage": str(state["stage"]),
        "candidate_id": str(state["candidate_id"]),
        "stream_seed": int(state.get("stream_seed", 2025)),
        "task": list(state.get("task", [])),
        "config_sha256": str(state["config_sha256"]),
        "command_path": str(state["command_path"]),
        "command_sha256": str(state["command_log_sha256"]),
        "log_path": str(state["log_path"]),
        "log_sha256": str(state["log_sha256"]),
    }
    for path_key, hash_key in (
        ("command_path", "command_sha256"),
        ("log_path", "log_sha256"),
    ):
        artifact = Path(str(references[path_key]))
        if not artifact.is_file() or _sha256_file(artifact) != references[hash_key]:
            raise ValueError(f"authoritative candidate {path_key} mismatch: {artifact}")
    return references


def _assert_no_final_artifacts(run_dir: Path) -> None:
    forbidden = []
    for path in (Path(run_dir) / "state").glob("*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("stage") in {"heldout", "stability"}:
            forbidden.append(str(path))
    for directory in ("commands", "logs"):
        for path in (Path(run_dir) / directory).glob("*"):
            if path.name.startswith(("heldout_", "stability_")):
                forbidden.append(str(path))
    if forbidden:
        raise ValueError(f"heldout/final artifacts predate freeze: {sorted(forbidden)}")


def _hash_manifest_artifacts(manifest: Mapping[str, object]) -> None:
    def walk(value):
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if isinstance(nested, str) and re.fullmatch(r"[0-9a-f]{64}", nested):
                    path = Path(str(key))
                    if path.suffix in {".pt", ".json", ".yaml"} or "/" in str(key):
                        if not path.is_file() or _sha256_file(path) != nested:
                            raise ValueError(f"source manifest artifact mismatch: {path}")
                else:
                    walk(nested)
    walk(manifest)


def _validate_source_checkpoint_manifest(
    manifest: Mapping[str, object],
    config: Mapping[str, object],
    cache_root: Path,
    cache: Mapping[str, object],
) -> None:
    """Bind all formal source artifacts to route, source, cache, and root."""
    expected_keys = {
        f"source-{variant}-{source}-2025"
        for variant in ("ordinary", "robust")
        for source in range(4)
    }
    if set(manifest) != expected_keys:
        raise ValueError("authoritative source checkpoint identities are incomplete")
    tensor_hashes = {
        str((cache_root / str(summary["tensor_file"])).resolve()): str(summary["tensor_sha256"])
        for summary in cache["domains"].values()
    }
    expected_cache_identity = {
        "mode": "formal",
        "manifest_path": str((cache_root / "manifest.json").resolve()),
        "manifest_sha256": str(cache["manifest_sha256"]),
        "content_sha256": str(cache["content_sha256"]),
        "tensor_sha256s": tensor_hashes,
    }
    checkpoint_root = Path(str(config["checkpoints"]["root"])).resolve()
    for variant in ("ordinary", "robust"):
        for source in range(4):
            key = f"source-{variant}-{source}-2025"
            outputs = {
                str(Path(str(path)).resolve()): str(digest)
                for path, digest in dict(manifest[key]).items()
            }
            checkpoint_paths = [Path(path) for path in outputs if Path(path).suffix == ".pt"]
            summary_paths = [
                Path(path) for path in outputs
                if Path(path).name == "source_training_summary.json"
            ]
            expected_parent = checkpoint_root / variant / f"source_{source}" / "seed_2025"
            if (
                len(outputs) != 2
                or len(checkpoint_paths) != 1
                or len(summary_paths) != 1
                or checkpoint_paths[0].parent != expected_parent
                or summary_paths[0].parent != expected_parent
            ):
                raise ValueError(f"authoritative source artifact paths mismatch: {key}")
            checkpoint_path, summary_path = checkpoint_paths[0], summary_paths[0]
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if (
                summary.get("route") != variant
                or summary.get("source") != source
                or summary.get("seed") != 2025
                or summary.get("checkpoint") != checkpoint_path.name
                or summary.get("checkpoint_sha256") != outputs[str(checkpoint_path)]
                or summary.get("cache_identity") != expected_cache_identity
            ):
                raise ValueError(f"authoritative source metadata mismatch: {key}")


def write_search_proof(
    run_dir: Path,
    config: Mapping[str, object],
    *,
    coordinate_rows: Sequence[Mapping[str, object]],
    stability_rows: Sequence[Mapping[str, object]],
    global_ranking: Sequence[Mapping[str, object]],
    final_ranking: Sequence[Mapping[str, object]],
    finalists: Sequence[Mapping[str, object]],
    selected: Mapping[str, object],
    checkpoint_manifest: Mapping[str, object],
) -> Path:
    """Write the externally bound, non-circular authority used by best_config."""
    try:
        from Lib.hust_strict_protocol import validate_cache
    except ModuleNotFoundError:
        from hust_strict_protocol import validate_cache
    run_dir = Path(run_dir)
    history_path = run_dir / "search_history.json"
    if not history_path.is_file():
        raise ValueError("authoritative search history must exist before freeze")
    _assert_no_final_artifacts(run_dir)
    validate_candidate_overrides(
        config, str(selected["candidate_id"]), dict(selected.get("overrides", {}))
    )
    cache_root = Path(str(config["protocol"]["data_path"]))
    cache = validate_cache(cache_root)
    if cache.get("version") != 2 or not cache.get("content_sha256"):
        raise ValueError("formal freeze requires a version-2 content-bound cache")
    config_path = Path("Configs/Experiments/HUST0711_strict_tuning.yaml")
    if not config_path.is_file() or load_config(config_path) != dict(config):
        raise ValueError("formal experiment config does not match the loaded config")
    _hash_manifest_artifacts(checkpoint_manifest)
    _validate_source_checkpoint_manifest(
        checkpoint_manifest, config, cache_root, cache
    )
    coordinate_evidence = [
        _formal_evidence_reference(run_dir, config, row) for row in coordinate_rows
    ]
    stability_evidence = [
        _formal_evidence_reference(run_dir, config, row) for row in stability_rows
    ]
    baseline_rows = [
        row
        for row in _state_records(run_dir)
        if row.get("kind") == "target"
        and row.get("stage") == "baseline"
        and row.get("route") == "0711_robust"
        and row.get("candidate_id") == "untuned"
        and int(row.get("stream_seed", -1)) == 2025
        and tuple(row.get("task", ())) in DEV_TASKS
        and row.get("status") == "succeeded"
    ]
    if len(baseline_rows) != len(DEV_TASKS):
        raise ValueError("authoritative proof requires four untuned seed-2025 dev states")
    baseline_evidence = [
        _formal_evidence_reference(run_dir, config, row) for row in baseline_rows
    ]
    selected_id = str(selected["candidate_id"])
    selected_rank = next(
        (
            index + 1
            for index, row in enumerate(final_ranking)
            if str(row["candidate_id"]) == selected_id
        ),
        None,
    )
    if selected_id != "untuned" and selected_rank is None:
        raise ValueError("selected candidate is absent from matched final ranking")
    proof_core = {
        "schema_version": 2,
        "created_at": time.time(),
        "experiment_config_path": str(config_path.resolve()),
        "experiment_config_sha256": _sha256_file(config_path),
        "cache_manifest_path": str((cache_root / "manifest.json").resolve()),
        "cache_manifest_sha256": str(cache["manifest_sha256"]),
        "cache_content_sha256": str(cache["content_sha256"]),
        "cache_tensor_sha256s": {
            str((cache_root / str(summary["tensor_file"])).resolve()): str(summary["tensor_sha256"])
            for summary in cache["domains"].values()
        },
        "checkpoint_manifest": dict(checkpoint_manifest),
        "checkpoint_manifest_sha256": _sha256_bytes(_canonical(checkpoint_manifest)),
        "search_history_path": str(history_path.resolve()),
        "search_history_sha256": _sha256_file(history_path),
        "coordinate_evidence": coordinate_evidence,
        "stability_evidence": stability_evidence,
        "baseline_evidence": baseline_evidence,
        "global_ranking": list(global_ranking),
        "final_ranking": list(final_ranking),
        "finalists": [str(row["candidate_id"]) for row in finalists],
        "selected": {
            "candidate_id": selected_id,
            "overrides": normalize_overrides(selected.get("overrides", {})),
            "rank": selected_rank,
            "untuned_fallback": selected_id == "untuned",
        },
        "pre_freeze_absence": {"heldout_states": [], "final_states": [], "commands": [], "logs": []},
    }
    proof = {**proof_core, "proof_sha256": _sha256_bytes(_canonical(proof_core))}
    proof_path = run_dir / "search_proof.json"
    if proof_path.exists():
        existing = verify_search_proof(proof_path)
        existing_semantics = {
            key: value
            for key, value in existing.items()
            if key not in {"created_at", "proof_sha256"}
        }
        intended_semantics = {
            key: value
            for key, value in proof_core.items()
            if key != "created_at"
        }
        if _canonical(existing_semantics) != _canonical(intended_semantics):
            raise ValueError("existing authoritative search proof does not match resumed tuning evidence")
        return proof_path
    atomic_write_json(proof_path, proof)
    return proof_path


def verify_search_proof(path: Path) -> dict[str, object]:
    """Recompute every external hash and semantic binding in a freeze proof."""
    path = Path(path)
    proof = json.loads(path.read_text(encoding="utf-8"))
    proof_hash = proof.pop("proof_sha256", None)
    if proof_hash != _sha256_bytes(_canonical(proof)) or proof.get("schema_version") != 2:
        raise ValueError("authoritative search proof hash mismatch")
    for path_key, hash_key in (
        ("experiment_config_path", "experiment_config_sha256"),
        ("cache_manifest_path", "cache_manifest_sha256"),
        ("search_history_path", "search_history_sha256"),
    ):
        artifact = Path(str(proof[path_key]))
        if not artifact.is_file() or _sha256_file(artifact) != proof[hash_key]:
            raise ValueError(f"authoritative proof artifact mismatch: {artifact}")
    config = load_config(Path(str(proof["experiment_config_path"])))
    if Path(str(config["protocol"]["data_path"])).resolve() != Path(str(proof["cache_manifest_path"])).parent.resolve():
        raise ValueError("authoritative proof cache/config path mismatch")
    for raw_path, digest in dict(proof["cache_tensor_sha256s"]).items():
        tensor = Path(raw_path)
        if not tensor.is_file() or _sha256_file(tensor) != digest:
            raise ValueError(f"authoritative cache tensor mismatch: {tensor}")
    try:
        from Lib.hust_strict_protocol import validate_cache
    except ModuleNotFoundError:
        from hust_strict_protocol import validate_cache
    cache = validate_cache(Path(str(proof["cache_manifest_path"])).parent)
    if cache.get("version") != 2 or cache.get("content_sha256") != proof["cache_content_sha256"]:
        raise ValueError("authoritative cache content identity mismatch")
    _hash_manifest_artifacts(dict(proof["checkpoint_manifest"]))
    _validate_source_checkpoint_manifest(
        dict(proof["checkpoint_manifest"]),
        config,
        Path(str(proof["cache_manifest_path"])).parent,
        cache,
    )
    manifest_outputs = [
        dict(outputs)
        for outputs in dict(proof["checkpoint_manifest"]).values()
        if isinstance(outputs, Mapping)
    ]
    checkpoint_paths = [
        str(Path(path).resolve())
        for outputs in manifest_outputs
        for path in outputs
        if Path(path).suffix == ".pt"
    ]
    if (
        len(dict(proof["checkpoint_manifest"])) != 8
        or len(manifest_outputs) != 8
        or any(
            len(outputs) != 2
            or sum(Path(path).suffix == ".pt" for path in outputs) != 1
            or sum(Path(path).suffix == ".json" for path in outputs) != 1
            for outputs in manifest_outputs
        )
        or len(checkpoint_paths) != 8
        or len(set(checkpoint_paths)) != 8
        or _sha256_bytes(_canonical(proof["checkpoint_manifest"])) != proof["checkpoint_manifest_sha256"]
    ):
        raise ValueError("authoritative checkpoint manifest identity mismatch")
    evidence_states: dict[str, list[dict[str, object]]] = {
        "baseline": [], "coordinate": [], "stability": [],
    }
    for family, evidence_rows in (
        ("baseline", proof.get("baseline_evidence", [])),
        ("coordinate", proof["coordinate_evidence"]),
        ("stability", proof["stability_evidence"]),
    ):
        for evidence in evidence_rows:
            for path_key, hash_key in (
                ("state_path", "state_sha256"),
                ("command_path", "command_sha256"),
                ("log_path", "log_sha256"),
            ):
                artifact = Path(str(evidence[path_key]))
                if not artifact.is_file() or _sha256_file(artifact) != evidence[hash_key]:
                    raise ValueError(f"authoritative candidate evidence mismatch: {artifact}")
            state = json.loads(Path(str(evidence["state_path"])).read_text(encoding="utf-8"))
            expected_plan = _planned_target(
                config,
                str(state.get("stage")),
                str(state.get("route")),
                tuple(state.get("task", ())),
                int(state.get("stream_seed", -1)),
                dict(state.get("overrides", {})),
                candidate=str(state.get("candidate_id")),
                beginning_only=bool(state.get("beginning_only", False)),
                load_split=bool(state.get("load_split", False)),
            )
            if (
                state.get("status") != "succeeded"
                or state.get("candidate_id") != evidence["candidate_id"]
                or state.get("stage") != evidence["stage"]
                or int(state.get("stream_seed", -1)) != evidence["stream_seed"]
                or list(state.get("task", [])) != evidence["task"]
                or state.get("config_sha256") != evidence["config_sha256"]
                or state.get("config_sha256") != expected_plan["config_sha256"]
            ):
                raise ValueError("authoritative candidate evidence identity mismatch")
            evidence_states[family].append(state)
    baseline_rows = evidence_states["baseline"]
    if (
        len(baseline_rows) != 4
        or {tuple(row["task"]) for row in baseline_rows} != set(DEV_TASKS)
        or any(
            row.get("stage") != "baseline"
            or row.get("candidate_id") != "untuned"
            or row.get("route") != "0711_robust"
            or int(row.get("stream_seed", -1)) != 2025
            for row in baseline_rows
        )
    ):
        raise ValueError("authoritative seed-2025 baseline evidence is incomplete")
    baseline_2025 = {
        tuple(row["task"]): float(row["strict_online"]) for row in baseline_rows
    }
    coordinate_rows = evidence_states["coordinate"]
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in coordinate_rows:
        groups.setdefault((str(row.get("stage")), str(row.get("candidate_id"))), []).append(row)
    if not groups or any(
        not stage.startswith("tune_group_")
        or len(rows) != 4
        or {tuple(row["task"]) for row in rows} != set(DEV_TASKS)
        or any(int(row.get("stream_seed", -1)) != 2025 for row in rows)
        for (stage, _candidate), rows in groups.items()
    ):
        raise ValueError("authoritative coordinate evidence is incomplete")
    history = json.loads(Path(str(proof["search_history_path"])).read_text(encoding="utf-8"))
    anchor = dict(config["baseline_overrides"])
    expected_history = []
    initial = dict(config["search"]["initial_candidate"]["overrides"])
    expected_stage_names = set()
    for group_index, group in enumerate(config["search"]["groups"]):
        stage = f"tune_group_{group_index:02d}_{group['name']}"
        expected_stage_names.add(stage)
        values = list(group["values"])
        if group_index == 0:
            values.append(initial)
        candidates = expand_coordinate_group(anchor, values)
        expected_ids = {candidate_id(values, 2025) for values in candidates}
        stage_rows = [row for row in coordinate_rows if row.get("stage") == stage]
        observed_ids = {str(row.get("candidate_id")) for row in stage_rows}
        if (
            observed_ids != expected_ids
            or len(stage_rows) != len(expected_ids) * len(DEV_TASKS)
        ):
            raise ValueError(f"authoritative coordinate inventory mismatch: {stage}")
        ranked_group = rank_candidates(
            stage_rows,
            set(DEV_TASKS),
            baseline_2025,
            float(config["selection"]["minimum_mean_gain"]),
            float(config["selection"]["maximum_task_regression"]),
            expected_stages={2025: stage},
            eligibility_seeds={2025},
        )
        winner = (
            ranked_group[0]
            if ranked_group
            else {
                "candidate_id": "untuned",
                "overrides": dict(config["baseline_overrides"]),
                "fallback": True,
            }
        )
        expected_history.append({
            "group_index": group_index,
            "group": group["name"],
            "candidate_ids": sorted(expected_ids),
            "winner": winner,
        })
        anchor = dict(winner["overrides"])
    if (
        {str(row.get("stage")) for row in coordinate_rows} != expected_stage_names
        or _canonical(history.get("groups")) != _canonical(expected_history)
    ):
        raise ValueError("authoritative coordinate search history is incomplete")
    global_ranking = rank_coordinate_candidates_globally(
        coordinate_rows,
        set(DEV_TASKS),
        baseline_2025,
        minimum_gain=float(config["selection"]["minimum_mean_gain"]),
        maximum_regression=float(config["selection"]["maximum_task_regression"]),
    )
    if _canonical(global_ranking) != _canonical(proof["global_ranking"]):
        raise ValueError("authoritative global coordinate ranking mismatch")
    finalists = global_ranking[:3] if len(global_ranking) >= 3 else global_ranking[:2]
    if [str(row["candidate_id"]) for row in finalists] != list(proof["finalists"]):
        raise ValueError("authoritative global finalist selection mismatch")
    stability_rows = evidence_states["stability"]
    expected_stability_candidates = {"untuned", *proof["finalists"]}
    finalist_overrides = {
        str(row["candidate_id"]): normalize_overrides(row["overrides"])
        for row in finalists
    }
    finalist_overrides["untuned"] = normalize_overrides(config["baseline_overrides"])
    if any(
        len([row for row in stability_rows if row.get("candidate_id") == candidate]) != 4
        or {
            tuple(row["task"])
            for row in stability_rows
            if row.get("candidate_id") == candidate
        } != set(DEV_TASKS)
        for candidate in expected_stability_candidates
    ) or {
        str(row.get("candidate_id")) for row in stability_rows
    } != expected_stability_candidates or any(
        row.get("stage") != "tune_stability"
        or int(row.get("stream_seed", -1)) != 2026
        or normalize_overrides(row.get("overrides", {}))
        != finalist_overrides.get(str(row.get("candidate_id")))
        for row in stability_rows
    ):
        raise ValueError("authoritative seed-2026 finalist evidence is incomplete")
    baseline_2026 = {
        tuple(row["task"]): float(row["strict_online"])
        for row in stability_rows
        if row.get("candidate_id") == "untuned"
    }
    matched_baseline = {(task, 2025): score for task, score in baseline_2025.items()}
    matched_baseline.update({(task, 2026): score for task, score in baseline_2026.items()})
    final_ranking = []
    for finalist in finalists:
        candidate = str(finalist["candidate_id"])
        stage_2025 = str(dict(finalist["stages"]).get(2025, dict(finalist["stages"]).get("2025", "")))
        candidate_rows = [
            row for row in coordinate_rows
            if row.get("candidate_id") == candidate and row.get("stage") == stage_2025
        ] + [
            row for row in stability_rows if row.get("candidate_id") == candidate
        ]
        final_ranking.extend(rank_candidates(
            candidate_rows,
            set(DEV_TASKS),
            matched_baseline,
            float(config["selection"]["minimum_mean_gain"]),
            float(config["selection"]["maximum_task_regression"]),
            required_seeds={2025, 2026},
            expected_stages={2025: stage_2025, 2026: "tune_stability"},
            eligibility_seeds={2025},
        ))
    final_ranking.sort(key=lambda row: (-row["mean_strict_online"], -row["minimum_task_delta"], row["runtime_seconds"], row["candidate_id"]))
    if _canonical(final_ranking) != _canonical(proof["final_ranking"]):
        raise ValueError("authoritative matched final ranking mismatch")
    expected_selected = (
        {
            "candidate_id": str(final_ranking[0]["candidate_id"]),
            "overrides": normalize_overrides(final_ranking[0]["overrides"]),
            "rank": 1,
            "untuned_fallback": False,
        }
        if final_ranking
        else {
            "candidate_id": "untuned",
            "overrides": normalize_overrides(config["baseline_overrides"]),
            "rank": None,
            "untuned_fallback": True,
        }
    )
    if proof["selected"] != expected_selected:
        raise ValueError("authoritative selected candidate mismatch")
    validate_candidate_overrides(
        config,
        str(expected_selected["candidate_id"]),
        dict(expected_selected["overrides"]),
    )
    history_selected = history.get("selected", {})
    selected_semantics = (
        history_selected.get("candidate_id") == expected_selected["candidate_id"]
        and normalize_overrides(history_selected.get("overrides", {})) == expected_selected["overrides"]
        and bool(history_selected.get("fallback", False)) == expected_selected["untuned_fallback"]
    )
    if (
        _canonical(history.get("global_coordinate_ranking")) != _canonical(proof["global_ranking"])
        or _canonical(history.get("matched_final_ranking")) != _canonical(proof["final_ranking"])
        or not selected_semantics
    ):
        raise ValueError("authoritative search history semantic mismatch")
    if proof.get("pre_freeze_absence") != {
        "heldout_states": [], "final_states": [], "commands": [], "logs": []
    }:
        raise ValueError("authoritative pre-freeze absence proof mismatch")
    return {**proof, "proof_sha256": proof_hash}


def freeze_candidate(
    run_dir: Path,
    identifier: str,
    overrides: Mapping[str, object],
    *,
    proof_path: Path | None = None,
) -> Path:
    path = Path(run_dir) / "best_config.yaml"
    if proof_path is None:
        raise ValueError("authoritative search proof is required before freeze")
    proof_path = Path(proof_path).resolve()
    proof = verify_search_proof(proof_path)
    selected = dict(proof["selected"])
    normalized = normalize_overrides(overrides)
    if selected["candidate_id"] != str(identifier) or selected["overrides"] != normalized:
        raise ValueError("frozen candidate does not match authoritative selection")
    core = {
        "schema_version": 2,
        "candidate_id": str(identifier),
        "overrides": normalized,
        "frozen_before_heldout": True,
        "frozen_at": time.time(),
        "proof_path": str(proof_path),
        "proof_sha256": str(proof["proof_sha256"]),
        "search_history_sha256": str(proof["search_history_sha256"]),
    }
    payload = {**core, "freeze_sha256": _sha256_bytes(_canonical(core))}
    serialized = yaml.safe_dump(payload, sort_keys=True)
    if path.exists():
        if path.read_text(encoding="utf-8") == serialized:
            return path
        raise FileExistsError(f"frozen candidate already exists: {path}")
    _write_yaml_atomic(path, payload)
    return path


def load_frozen_candidate(path: Path) -> dict[str, object]:
    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {
        "schema_version", "candidate_id", "overrides", "frozen_before_heldout",
        "frozen_at", "proof_path", "proof_sha256", "search_history_sha256",
        "freeze_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != required or payload.get("schema_version") != 2:
        raise ValueError("authoritative frozen configuration proof is missing")
    freeze_hash = payload.pop("freeze_sha256")
    if freeze_hash != _sha256_bytes(_canonical(payload)):
        raise ValueError("frozen candidate hash mismatch")
    proof_path = Path(str(payload["proof_path"]))
    if proof_path.parent != path.resolve().parent:
        raise ValueError("frozen proof must be owned by the same run directory")
    proof = verify_search_proof(proof_path)
    if (
        payload["proof_sha256"] != proof["proof_sha256"]
        or payload["search_history_sha256"] != proof["search_history_sha256"]
        or payload["candidate_id"] != proof["selected"]["candidate_id"]
        or payload["overrides"] != proof["selected"]["overrides"]
        or float(payload["frozen_at"]) < float(proof["created_at"])
    ):
        raise ValueError("frozen configuration does not match authoritative proof")
    for state_path in path.parent.joinpath("state").glob("*.json"):
        state = json.loads(state_path.read_text(encoding="utf-8"))
        stage = str(state.get("stage", ""))
        started = float(state.get("started_at", 0.0) or 0.0)
        ended = float(state.get("ended_at", 0.0) or 0.0)
        if (stage.startswith("tune_group_") or stage == "tune_stability") and ended > float(payload["frozen_at"]):
            raise ValueError("candidate evidence was created after freeze")
        if stage in {"heldout", "stability"} and started < float(payload["frozen_at"]):
            raise ValueError("heldout/final evidence predates freeze")
    return {**payload, "freeze_sha256": freeze_hash}


def _state_records(run_dir: Path) -> list[dict[str, object]]:
    try:
        from tools.summarize_hust_dtcc_0711 import flatten_state_metrics
    except ModuleNotFoundError:
        from summarize_hust_dtcc_0711 import flatten_state_metrics
    records = []
    for path in sorted((Path(run_dir) / "state").glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        value["_formal_state"] = True
        value["_state_path"] = str(path.resolve())
        records.append(flatten_state_metrics(value))
    return records


def _build_primary_checkpoint_manifest(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    sources = [
        row
        for row in records
        if row.get("kind") == "source"
        and row.get("status") == "succeeded"
        and row.get("stage") == "source"
        and row.get("load_split") is False
    ]
    expected = {
        (variant, source)
        for variant in ("ordinary", "robust")
        for source in range(4)
    }
    if len(sources) != len(expected):
        raise ValueError("primary source inventory must contain exactly eight succeeded sources")
    identities = []
    candidates = []
    checkpoint_hashes = []
    manifest = {}
    for row in sources:
        variant = row.get("variant")
        source = row.get("source")
        identity = (variant, source)
        if (
            identity not in expected
            or row.get("stage") != "source"
            or row.get("load_split") is not False
            or row.get("route") != f"source_{variant}"
            or row.get("source_seed") != 2025
        ):
            raise ValueError(f"unexpected primary source state: {identity}")
        candidate = row.get("candidate_id")
        outputs = row.get("output_artifact_hashes")
        checkpoint_path = row.get("source_checkpoint_path")
        if (
            candidate != f"source-{variant}-{source}-2025"
            or not isinstance(outputs, Mapping)
        ):
            raise ValueError(f"invalid primary source manifest state: {identity}")
        normalized_outputs = {
            str(Path(str(path)).resolve()): str(digest)
            for path, digest in outputs.items()
        }
        checkpoint_hash = normalized_outputs.get(
            str(Path(str(checkpoint_path)).resolve())
        )
        if checkpoint_hash is None:
            raise ValueError(f"primary source checkpoint missing from outputs: {identity}")
        identities.append(identity)
        candidates.append(candidate)
        checkpoint_hashes.append(checkpoint_hash)
        manifest[candidate] = dict(outputs)
    if set(identities) != expected or len(set(identities)) != len(expected):
        raise ValueError("primary source identities must be exact and unique")
    if len(set(candidates)) != len(expected):
        raise ValueError("primary source candidate identities must be unique")
    if len(set(checkpoint_hashes)) != len(expected):
        raise ValueError("primary source checkpoint hashes must be unique")
    return manifest


def run_tuning_stage(
    run_dir: Path, config: Mapping[str, object], gpus: Sequence[int], *,
    executor=execute_parallel_tasks, baseline_records: Sequence[Mapping[str, object]] | None = None,
    deadline: float | None = None, run_process=None,
) -> dict[str, object]:
    """Execute coordinate groups sequentially, stability-check finalists, then freeze."""
    frozen_path = Path(run_dir) / "best_config.yaml"
    if frozen_path.exists():
        return load_frozen_candidate(frozen_path)
    baseline_rows = list(baseline_records or [row for row in _state_records(run_dir) if row.get("stage") == "baseline" and row.get("route") == "0711_robust" and row.get("candidate_id") == "untuned"])
    baseline_2025 = {tuple(row["task"]): float(row["strict_online"]) for row in baseline_rows if row.get("status") == "succeeded" and int(row.get("stream_seed", 2025)) == 2025 and tuple(row["task"]) in DEV_TASKS}
    if set(baseline_2025) != set(DEV_TASKS):
        raise RuntimeError("tuning requires complete untuned seed-2025 development baseline")
    anchor = dict(config["baseline_overrides"])
    history: list[dict[str, object]] = []
    all_candidate_rows: list[dict[str, object]] = []
    initial = dict(config["search"]["initial_candidate"]["overrides"])
    for group_index, group in enumerate(config["search"]["groups"]):
        values = list(group["values"])
        if group_index == 0:
            values.append(initial)
        candidates = expand_coordinate_group(anchor, values)
        jobs = []
        for overrides in candidates:
            cid = candidate_id(overrides, 2025)
            jobs.extend(_declare_job_evidence(config, _planned_target(config, f"tune_group_{group_index:02d}_{group['name']}", "0711_robust", task, 2025, overrides, candidate=cid)) for task in DEV_TASKS)
        rows = list(executor(run_dir, config, jobs, gpus, run_process=run_process, deadline=deadline))
        failed = [str(row.get("candidate_id")) for row in rows if row.get("status") != "succeeded"]
        if len(rows) != len(jobs) or failed:
            raise RuntimeError(
                f"tuning group {group_index} failed or incomplete: "
                f"expected={len(jobs)} observed={len(rows)} failed={failed}"
            )
        all_candidate_rows.extend(rows)
        expected_stage = f"tune_group_{group_index:02d}_{group['name']}"
        ranked = rank_candidates(rows, set(DEV_TASKS), baseline_2025, config["selection"]["minimum_mean_gain"], config["selection"]["maximum_task_regression"], expected_stages={2025: expected_stage})
        if ranked:
            winner = ranked[0]
            anchor = dict(winner["overrides"])
        else:
            winner = {"candidate_id": "untuned", "overrides": dict(config["baseline_overrides"]), "fallback": True}
        history.append({"group_index": group_index, "group": group["name"], "candidate_ids": sorted({str(row["candidate_id"]) for row in rows}), "winner": winner})
    global_ranking = rank_coordinate_candidates_globally(
        all_candidate_rows,
        set(DEV_TASKS),
        baseline_2025,
        minimum_gain=float(config["selection"]["minimum_mean_gain"]),
        maximum_regression=float(config["selection"]["maximum_task_regression"]),
    )
    finalists = global_ranking[:3] if len(global_ranking) >= 3 else global_ranking[:2]
    stability_jobs = [_declare_job_evidence(config, _planned_target(config, "tune_stability", "0711_robust", task, 2026, dict(config["baseline_overrides"]), candidate="untuned")) for task in DEV_TASKS]
    for finalist in finalists:
        stability_jobs.extend(_declare_job_evidence(config, _planned_target(config, "tune_stability", "0711_robust", task, 2026, finalist["overrides"], candidate=finalist["candidate_id"])) for task in DEV_TASKS)
    stability_rows = list(executor(run_dir, config, stability_jobs, gpus, run_process=run_process, deadline=deadline))
    failed_stability = [
        str(row.get("candidate_id"))
        for row in stability_rows
        if row.get("status") != "succeeded"
    ]
    if len(stability_rows) != len(stability_jobs) or failed_stability:
        raise RuntimeError(
            "matched stability failed or incomplete: "
            f"expected={len(stability_jobs)} observed={len(stability_rows)} "
            f"failed={failed_stability}"
        )
    baseline_2026 = {tuple(row["task"]): float(row["strict_online"]) for row in stability_rows if row.get("candidate_id") == "untuned" and row.get("status") == "succeeded"}
    matched_baseline = {(task, 2025): baseline_2025[task] for task in DEV_TASKS}
    matched_baseline.update({(task, 2026): baseline_2026[task] for task in DEV_TASKS if task in baseline_2026})
    ranked_final = []
    if len(baseline_2026) == len(DEV_TASKS):
        for finalist in finalists:
            cid = str(finalist["candidate_id"])
            stage_2025 = str(dict(finalist.get("stages", {})).get(2025, ""))
            selection_rows = [row for row in all_candidate_rows + stability_rows if str(row.get("candidate_id")) == cid and (row.get("stage") == stage_2025 or row.get("stage") == "tune_stability")]
            ranked_final.extend(rank_candidates(
                selection_rows,
                set(DEV_TASKS),
                matched_baseline,
                config["selection"]["minimum_mean_gain"],
                config["selection"]["maximum_task_regression"],
                required_seeds={2025, 2026},
                expected_stages={2025: stage_2025, 2026: "tune_stability"},
                eligibility_seeds={2025},
            ))
        ranked_final.sort(key=lambda row: (-row["mean_strict_online"], -row["minimum_task_delta"], row["runtime_seconds"], row["candidate_id"]))
    selected = ranked_final[0] if ranked_final else {"candidate_id": "untuned", "overrides": dict(config["baseline_overrides"]), "fallback": True}
    validate_candidate_overrides(
        config, str(selected["candidate_id"]), dict(selected["overrides"])
    )
    search_history = {
        "schema_version": 2,
        "groups": history,
        "global_coordinate_ranking": global_ranking,
        "finalists": finalists,
        "matched_final_ranking": ranked_final,
        "matched_seed2026": stability_rows,
        "selected": selected,
    }
    atomic_write_json(Path(run_dir) / "search_history.json", search_history)
    checkpoint_manifest = _build_primary_checkpoint_manifest(_state_records(run_dir))
    proof_path = write_search_proof(
        run_dir,
        config,
        coordinate_rows=all_candidate_rows,
        stability_rows=stability_rows,
        global_ranking=global_ranking,
        final_ranking=ranked_final,
        finalists=finalists,
        selected=selected,
        checkpoint_manifest=checkpoint_manifest,
    )
    freeze_candidate(
        run_dir,
        str(selected["candidate_id"]),
        dict(selected["overrides"]),
        proof_path=proof_path,
    )
    return load_frozen_candidate(frozen_path)


def _dry_run_jobs(config: Mapping[str, object], stage: str) -> list[dict[str, object]]:
    tasks = [tuple(x) for x in config["tasks"]["development"] + config["tasks"]["heldout"]]
    routes = ("dtcc_ordinary", "0711_robust", "0711_common") if stage in {"all", "baseline"} else ("0711_robust",)
    return [{"route": route, "overrides": {}, "task": task, "stream_seed": 2025} for route in routes for task in tasks]


def prepare_dry_run(run_dir: Path, config: Mapping[str, object], stage: str, gpus: Sequence[int]) -> list[list[str]]:
    """Write exact unresolved/resolved plans without launching subprocesses."""
    def unresolved_final():
        heldout = tuple(tuple(x) for x in config["tasks"]["heldout"])
        preview = [_declare_job_evidence(config, _planned_target(config, "heldout-preview-unresolved", "0711_robust", task, 2025, {}, candidate="FROZEN_CONFIG_REQUIRED"), resolve=False) for task in heldout]
        for cid in ("untuned", "FROZEN_CONFIG_REQUIRED"):
            preview.extend(_declare_job_evidence(config, _planned_target(config, "stability-preview-unresolved", "0711_robust", task, 2026, {}, candidate=cid), resolve=False) for task in ALL_TASKS)
        return preview

    if stage == "all":
        jobs = plan_stage_jobs(config, "source", run_dir, resolve_artifacts=False) + plan_stage_jobs(config, "beginning", run_dir, resolve_artifacts=False) + plan_stage_jobs(config, "baseline", run_dir, resolve_artifacts=False) + plan_stage_jobs(config, "tune", run_dir, resolve_artifacts=False) + unresolved_final()
    elif stage == "final" and not (Path(run_dir) / "best_config.yaml").exists():
        jobs = unresolved_final()
    else:
        jobs = plan_stage_jobs(config, stage, run_dir, resolve_artifacts=False)
    commands: list[list[str]] = []
    labels: list[str] = []
    for index, job in enumerate(jobs):
        gpu = int(gpus[index % len(gpus)])
        if job["kind"] == "source":
            command = build_stage_command(config, "source", source=int(job["source"]), variant=str(job["variant"]), load_split=bool(job.get("load_split")), gpu=gpu)
        else:
            command = _target_command_for_job(config, job, gpu)
        commands.append(command)
        labels.append(_job_identity(config, job)[1])
    command_dir = Path(run_dir) / "commands"
    state_dir = Path(run_dir) / "state"
    command_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    for index, (label, command) in enumerate(zip(labels, commands, strict=True)):
        (command_dir / f"dry_run_{index:03d}_{label}.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
        preview = {**jobs[index], "status": "pending", "dry_run": True, "command": command, "command_sha256": _sha256_bytes(_canonical(command)), "gpu": int(gpus[index % len(gpus)])}
        (state_dir / f"dry_run_{index:03d}_{label}.json").write_text(json.dumps(preview, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    atomic_write_json(Path(run_dir) / "dry_run.json", {"stage": stage, "jobs": len(commands), "gpus": list(dict.fromkeys(gpus)), "launched": False, "unresolved_jobs": sum(job["evidence_status"] == "unresolved" for job in jobs)})
    return commands


def ensure_v2_load_cache(
    config: Mapping[str, object],
    *,
    builder_run: Callable[[list[str]], object] | None = None,
) -> dict[str, object]:
    """Build a missing V2 load cache, then validate it before any GPU audit job."""
    try:
        from Lib.hust_strict_protocol import validate_cache
    except ModuleNotFoundError:
        from hust_strict_protocol import validate_cache
    output = Path(str(config["protocol"]["load_data_path"]))
    try:
        existing = validate_cache(output)
    except ValueError:
        existing = None
    if existing is not None:
        if existing.get("version") != 2 or existing.get("split") != "load" or not existing.get("content_sha256"):
            raise ValueError("load audit cache is not a content-bound V2 load cache")
        return existing
    command = [
        sys.executable,
        "tools/build_hust_strict_cache.py",
        "--raw-root",
        "Dataset/HUST",
        "--output",
        str(output),
        "--seed",
        "2025",
        "--split",
        "load",
    ]
    if builder_run is None:
        subprocess.run(command, check=True)
    else:
        builder_run(command)
    built = validate_cache(output)
    if built.get("version") != 2 or built.get("split") != "load" or not built.get("content_sha256"):
        raise ValueError("load cache builder did not produce a valid content-bound V2 cache")
    return built


def run_formal_stage(run_dir: Path, config: Mapping[str, object], stage: str, gpus: Sequence[int], *, run_process=None, retry_failed: bool = False, deadline: float | None = None):
    if stage == "report":
        try:
            from tools.summarize_hust_dtcc_0711 import write_reports
        except ModuleNotFoundError:
            from summarize_hust_dtcc_0711 import write_reports
        frozen_path = Path(run_dir) / "best_config.yaml"
        frozen = load_frozen_candidate(frozen_path) if frozen_path.exists() else None
        records = _state_records(run_dir)
        checkpoint_manifest = _build_primary_checkpoint_manifest(records)
        return write_reports(
            run_dir,
            records,
            frozen=frozen,
            checkpoint_manifest=checkpoint_manifest,
            strict_final=True,
        )
    if stage == "tune":
        return run_tuning_stage(run_dir, config, gpus, deadline=deadline, run_process=run_process)
    if stage == "all":
        results = {}
        for name in ("source", "beginning"):
            results[name] = run_formal_stage(run_dir, config, name, gpus, run_process=run_process, retry_failed=retry_failed, deadline=deadline)
        beginning_records = [row for row in _state_records(run_dir) if row.get("stage") == "beginning" and row.get("status") == "succeeded"]
        statuses = {}
        try:
            from tools.summarize_hust_dtcc_0711 import beginning_audit
        except ModuleNotFoundError:
            from summarize_hust_dtcc_0711 import beginning_audit
        for variant in ("ordinary", "robust"):
            unique = {tuple(row["task"]): float(row["before"]) for row in beginning_records if row.get("variant") == variant and int(row.get("stream_seed", 2025)) == 2025}
            statuses[variant] = beginning_audit(list(unique.values())) if len(unique) == 12 else "supplementary_load_audit_required"
        if "supplementary_load_audit_required" in statuses.values():
            results["load-cache"] = ensure_v2_load_cache(config)
            results["load-audit"] = run_formal_stage(run_dir, config, "load-audit", gpus, run_process=run_process, retry_failed=retry_failed, deadline=deadline)
        results["baseline"] = run_formal_stage(run_dir, config, "baseline", gpus, run_process=run_process, retry_failed=retry_failed, deadline=deadline)
        results["tune"] = run_formal_stage(run_dir, config, "tune", gpus, run_process=run_process, retry_failed=retry_failed, deadline=deadline)
        results["final"] = run_formal_stage(run_dir, config, "final", gpus, run_process=run_process, retry_failed=retry_failed, deadline=deadline)
        results["report"] = run_formal_stage(run_dir, config, "report", gpus)
        return results
    jobs = plan_stage_jobs(config, stage, run_dir)
    records = execute_parallel_tasks(
        run_dir,
        config,
        jobs,
        gpus,
        run_process=run_process,
        retry_failed=retry_failed,
        deadline=deadline,
    )
    failed = [
        str(row.get("candidate_id", row.get("stage", "unknown")))
        for row in records
        if row.get("status") != "succeeded"
    ]
    if len(records) != len(jobs) or failed:
        raise RuntimeError(
            f"formal stage {stage} failed or incomplete: "
            f"expected={len(jobs)} observed={len(records)} failed={failed}"
        )
    return records


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("source", "beginning", "baseline", "tune", "final", "report", "load-audit", "all"), default="all")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--gpus", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    config = load_config(args.config)
    run_dir = args.run_dir or Path("logs") / f"HUST_DTCC_0711_STRICT_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}"
    if args.stage == "report" and not args.dry_run:
        run_formal_stage(run_dir, config, "report", [])
        return 0
    requested = discover_idle_gpus() if args.gpus == "auto" and not args.dry_run else ([0] if args.gpus == "auto" else list(dict.fromkeys(int(x) for x in args.gpus.split(","))))
    if args.dry_run:
        prepare_dry_run(run_dir, config, args.stage, requested)
        return 0
    eligible = set(discover_idle_gpus(max_utilization=config["protocol"]["max_gpu_utilization"], max_memory_fraction=config["protocol"]["max_gpu_memory_fraction"]))
    if not set(requested) <= eligible:
        raise RuntimeError(f"requested GPUs are not currently eligible: {sorted(set(requested) - eligible)}")
    deadline = time.time() + float(config["budget"]["tuning_hours"]) * 3600
    run_formal_stage(run_dir, config, args.stage, requested, retry_failed=args.retry_failed, deadline=deadline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
