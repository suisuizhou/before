#!/usr/bin/env python3
from __future__ import annotations

from collections.abc import Mapping
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Literal

import yaml


HISTORICAL_BASELINE = {
    (0, 1): 40.18,
    (0, 2): 94.93,
    (0, 3): 68.45,
    (1, 0): 65.58,
    (1, 2): 63.18,
    (1, 3): 45.65,
    (2, 0): 95.61,
    (2, 1): 43.52,
    (2, 3): 70.20,
    (3, 0): 67.09,
    (3, 1): 30.15,
    (3, 2): 68.27,
}
DEV_TASKS = ((0, 1), (0, 3), (1, 0))
HELDOUT_TASKS = tuple(task for task in HISTORICAL_BASELINE if task not in DEV_TASKS)

TUNABLE_KEYS = frozenset(
    {
        "Opt.lr_tar",
        "TTA0711.warp_lr_scale",
        "TTA0711.ema_beta",
        "TTA0711.warmup_batches",
        "TTA0711.aux_ramp_batches",
        "TTA0711.min_reliability",
        "TTA0711.lambda_mt",
        "TTA0711.lambda_pcl",
        "TTA0711.lambda_ncl",
        "TTA0711.memory_per_class",
        "TTA0711.pcl_temperature",
        "TTA0711.ncl_temperature",
    }
)

EXPECTED_TOP_LEVEL = frozenset(
    {
        "version",
        "protocol",
        "checkpoints",
        "recovery",
        "tasks",
        "budget",
        "selection",
        "search",
    }
)
IMMUTABLE_PROTOCOL = {
    "dataset": "PU4D",
    "data_path": "Dataset/PU4D_CACHE",
    "runner": "main_tta_0711_strict_online.py",
    "model": "ResNet18_1D_SDE",
    "model_type": "linear",
    "batch_size": 128,
    "source_seed": 2025,
    "stream_seed": 2025,
    "mode": "full",
    "passes": 1,
    "gpu": 0,
}

BASELINE_TUNING_OVERRIDES: dict[str, object] = {
    "Opt.lr_tar": 0.015,
    "TTA0711.warp_lr_scale": 0.1,
    "TTA0711.ema_beta": 0.995,
    "TTA0711.warmup_batches": 10,
    "TTA0711.aux_ramp_batches": 20,
    "TTA0711.min_reliability": 0.20,
    "TTA0711.lambda_mt": 0.02,
    "TTA0711.lambda_pcl": 0.02,
    "TTA0711.lambda_ncl": 0.01,
    "TTA0711.memory_per_class": 64,
    "TTA0711.pcl_temperature": 0.20,
    "TTA0711.ncl_temperature": 0.20,
}

FIXED_RUNNER_OVERRIDES: dict[str, object] = {
    "Model.model_name": "ResNet18_1D_SDE",
    "Model.model_type": "linear",
    "Model.use_spectral_adapter": True,
    "Model.band_num": 256,
    "Model.adapter_delta": 0.1,
    "Opt.lr_src": 0.001,
    "Opt.weight_decay_tar": 0.001,
    "TTA0711.alpha": 2.0,
    "TTA0711.eta": 0.05,
    "TTA0711.teacher_temp": 1.0,
    "TTA0711.mt_warmup_scale": 0.5,
    "TTA0711.view_style_strength": 0.05,
    "TTA0711.view_style_knots": 8,
    "TTA0711.view_warp_max": 0.5,
    "TTA0711.view_warp_knots": 8,
    "TTA0711.view_gain_strength": 0.03,
    "TTA0711.view_baseline_strength": 0.02,
    "TTA0711.view_noise_std": 0.01,
    "TTA0711.view_gamma": 5.0,
    "TTA0711.evidence_interval": 1,
    "TTA0711.evidence_metric": "margin",
    "TTA0711.sampling_rate_hz": 64000,
    "TTA0711.fft_size": 1024,
    "TTA0711.spectrum_length": 512,
    "TTA0711.physical_harmonics": 8,
    "TTA0711.outer_sideband_orders": [0, 1],
    "TTA0711.inner_sideband_orders": [0, 1, 2],
    "TTA0711.mask_sigma_bins": 1.0,
    "TTA0711.physical_background_width": 7,
    "TTA0711.max_mask_ratio": 0.18,
    "TTA0711.mask_activity_threshold": 0.10,
    "TTA0711.exclude_dc": True,
    "TTA0711.min_pcl_classes": 8,
    "TTA0711.ncl_neighbors": 3,
    "TTA0711.min_ncl_classes": 16,
    "TTA0711.min_ncl_entries": 64,
    "TTA0711.use_frequency_warp": True,
    "TTA0711.warp_knots": 16,
    "TTA0711.max_warp": 2.0,
    "TTA0711.warp_smooth_weight": 2.0,
    "TTA0711.adapter_lr_scale": 1.0,
    "TTA0711.lambda_adapter": 0.001,
    "TTA0711.lambda_warp": 0.0002,
    "TTA0711.log_interval": 25,
}


@dataclass(frozen=True)
class TaskRecord:
    candidate_id: str
    stage: str
    task: tuple[int, int]
    overrides: dict[str, object]
    stream_seed: int
    command: list[str]
    status: str
    attempt: int
    started_at: str | None
    ended_at: str | None
    returncode: int | None
    log_path: str
    metrics: dict[str, object] | None
    failure_class: str | None


class SubprocessTaskRunner:
    def run(self, command: list[str], log_path: Path, env: Mapping[str, str]) -> int:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=dict(env),
                check=False,
            )
        return int(completed.returncode)


_SUMMARY_MODULE = None


def _summary_module():
    global _SUMMARY_MODULE
    if _SUMMARY_MODULE is None:
        path = Path(__file__).with_name("summarize_pu4d_0711_tuning.py")
        spec = importlib.util.spec_from_file_location("pu4d_0711_summary_runtime", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load summary module: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _SUMMARY_MODULE = module
    return _SUMMARY_MODULE


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("tuning config must be a mapping")
    return loaded


def _task_tuple(value: object) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"invalid PU4D task: {value!r}")
    source, target = value
    if not isinstance(source, int) or not isinstance(target, int) or source == target:
        raise ValueError(f"invalid PU4D task: {value!r}")
    return source, target


def parse_task_name(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)to(\d+)", value)
    if match is None:
        raise ValueError("task must use exact syntax such as 0to1")
    task = int(match.group(1)), int(match.group(2))
    if task not in HISTORICAL_BASELINE:
        raise ValueError(f"unknown PU4D task: {value}")
    return task


def validate_config(config: Mapping[str, object]) -> None:
    unknown_top = set(config) - EXPECTED_TOP_LEVEL
    missing_top = EXPECTED_TOP_LEVEL - set(config)
    if unknown_top or missing_top:
        raise ValueError(
            f"config keys invalid: unknown={sorted(unknown_top)}, missing={sorted(missing_top)}"
        )

    if config.get("version") != 1:
        raise ValueError("config version must be 1")

    protocol = config.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("protocol must be a mapping")
    for key, expected in IMMUTABLE_PROTOCOL.items():
        if protocol.get(key) != expected:
            raise ValueError(
                f"protocol invariant {key} must be {expected!r}, got {protocol.get(key)!r}"
            )

    tasks = config.get("tasks")
    if not isinstance(tasks, Mapping):
        raise ValueError("tasks must be a mapping")
    development = tuple(_task_tuple(task) for task in tasks.get("development", ()))
    heldout = tuple(_task_tuple(task) for task in tasks.get("heldout", ()))
    if development != DEV_TASKS:
        raise ValueError(f"development tasks must be {DEV_TASKS!r}")
    if set(development + heldout) != set(HISTORICAL_BASELINE):
        raise ValueError("development and heldout tasks must partition all 12 PU4D tasks")

    recovery = config.get("recovery")
    if not isinstance(recovery, Mapping):
        raise ValueError("recovery must be a mapping")
    if float(recovery.get("expected_mean", -1.0)) != 62.7342:
        raise ValueError("recovery expected_mean must be 62.7342")
    expected_scores = recovery.get("expected")
    if not isinstance(expected_scores, Mapping) or len(expected_scores) != 12:
        raise ValueError("recovery expected scores must contain all 12 tasks")

    checkpoints = config.get("checkpoints")
    if not isinstance(checkpoints, Mapping):
        raise ValueError("checkpoints must be a mapping")
    expected_hashes = checkpoints.get("expected_sha256")
    if not isinstance(expected_hashes, Mapping) or set(expected_hashes) != {
        "0",
        "1",
        "2",
        "3",
    }:
        raise ValueError("checkpoint hashes must cover PU4D sources 0..3")
    if any(not re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in expected_hashes.values()):
        raise ValueError("checkpoint SHA-256 values must be lowercase hexadecimal")

    search = config.get("search")
    if not isinstance(search, Mapping) or not isinstance(search.get("groups"), list):
        raise ValueError("search.groups must be a list")
    for group in search["groups"]:
        if not isinstance(group, Mapping) or not isinstance(group.get("values"), list):
            raise ValueError("each search group must contain a values list")
        for candidate in group["values"]:
            if not isinstance(candidate, Mapping):
                raise ValueError("candidate overrides must be mappings")
            for key in candidate:
                if key not in TUNABLE_KEYS:
                    raise ValueError(f"candidate parameter {key!r} is not tunable")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_preflight(
    project_root: Path,
    config: Mapping[str, object],
    *,
    check_gpu: bool = True,
) -> dict[str, object]:
    validate_config(config)
    project_root = Path(project_root).resolve()
    protocol = config["protocol"]
    checkpoints = config["checkpoints"]
    assert isinstance(protocol, Mapping)
    assert isinstance(checkpoints, Mapping)

    dataset_path = project_root / str(protocol["data_path"])
    if not dataset_path.is_dir() or not any(dataset_path.glob("*.pt")):
        raise RuntimeError(f"PU4D cache is missing or empty: {dataset_path}")
    runner_path = project_root / str(protocol["runner"])
    if not runner_path.is_file():
        raise RuntimeError(f"strict 0711 runner is missing: {runner_path}")

    expected_hashes = checkpoints["expected_sha256"]
    assert isinstance(expected_hashes, Mapping)
    checkpoint_rows = []
    for source in range(4):
        checkpoint_path = (
            project_root
            / str(checkpoints["root"])
            / f"source_{source}"
            / f"seed_{protocol['source_seed']}"
            / str(checkpoints["filename"])
        )
        if not checkpoint_path.is_file():
            raise RuntimeError(f"source checkpoint is missing: {checkpoint_path}")
        actual = _sha256_file(checkpoint_path)
        expected = str(expected_hashes[str(source)])
        if actual != expected:
            raise RuntimeError(
                f"source {source} checkpoint SHA-256 mismatch: "
                f"expected {expected}, got {actual}"
            )
        checkpoint_rows.append(
            {"source": source, "path": str(checkpoint_path), "sha256": actual}
        )

    gpu_row = None
    if check_gpu:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"nvidia-smi failed: {completed.stderr.strip()}")
        rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
        gpu_rows = [row for row in rows if row.split(",", 1)[0].strip() == "0"]
        if len(gpu_rows) != 1:
            raise RuntimeError("GPU0 is not visible")
        gpu_row = gpu_rows[0]

    config_sha256 = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    try:
        import torch

        pytorch_version = str(torch.__version__)
        cuda_version = str(torch.version.cuda or "unavailable")
    except ImportError:
        pytorch_version = "unavailable"
        cuda_version = "unavailable"
    git_result = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )

    return {
        "passed": True,
        "dataset_path": str(dataset_path),
        "runner_path": str(runner_path),
        "checkpoints": checkpoint_rows,
        "config_sha256": config_sha256,
        "git_head": git_result.stdout.strip() if git_result.returncode == 0 else None,
        "environment": {
            "python": sys.version.splitlines()[0],
            "pytorch": pytorch_version,
            "cuda": cuda_version,
        },
        "gpu0": gpu_row,
        "completed_at": _utc_now(),
    }


def normalize_overrides(overrides: Mapping[str, object]) -> dict[str, object]:
    return {key: overrides[key] for key in sorted(overrides)}


def candidate_id(
    overrides: Mapping[str, object], source_seed: int, stream_seed: int
) -> str:
    payload = {
        "overrides": normalize_overrides(overrides),
        "source_seed": int(source_seed),
        "stream_seed": int(stream_seed),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def expand_coordinate_group(
    anchor: Mapping[str, object], values: list[Mapping[str, object]]
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    seen: set[str] = set()
    for value in values:
        merged = dict(anchor)
        merged.update(value)
        normalized = normalize_overrides(merged)
        marker = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        if marker not in seen:
            seen.add(marker)
            candidates.append(normalized)
    return candidates


def _hydra_value(value: object) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), separators=(",", ":"))
    return str(value)


def _override_arg(key: str, value: object) -> str:
    prefix = "++" if key.startswith("TTA0711.") else ""
    return f"{prefix}{key}={_hydra_value(value)}"


def build_command(
    config: Mapping[str, object],
    overrides: Mapping[str, object],
    task: tuple[int, int],
    stream_seed: int,
) -> list[str]:
    validate_config(config)
    task = _task_tuple(task)
    if task not in HISTORICAL_BASELINE:
        raise ValueError(f"unknown PU4D task: {task!r}")
    invalid = set(overrides) - TUNABLE_KEYS
    if invalid:
        raise ValueError(f"candidate parameter {sorted(invalid)[0]!r} is not tunable")

    protocol = config["protocol"]
    assert isinstance(protocol, Mapping)
    source, target = task
    command = [
        sys.executable,
        str(protocol["runner"]),
        "Model=ResNet18_1D_SDE",
        "Dataset=PU4D",
        "gpu_id=0",
        "process_wandb=False",
        f"++Dataset.data_path={protocol['data_path']}",
        f"++seed_runs=[{protocol['source_seed']}]",
        f"++only_task=[{source},{target}]",
        "batch_size=128",
        f"num_workers={protocol['num_workers']}",
        "++TTA0711.mode=full",
        "++TTA0711.passes=1",
        f"++TTA0711.stream_seed={int(stream_seed)}",
    ]

    effective = dict(FIXED_RUNNER_OVERRIDES)
    effective.update(BASELINE_TUNING_OVERRIDES)
    effective.update(normalize_overrides(overrides))
    command.extend(_override_arg(key, value) for key, value in effective.items())
    return command


def prepare_dry_run(
    run_dir: Path,
    config: Mapping[str, object],
    *,
    stage: str,
    only_task: tuple[int, int] | None,
) -> list[list[str]]:
    validate_config(config)
    if only_task is not None and stage != "recovery":
        raise ValueError("--task is only allowed for the recovery stage")
    if stage not in {"all", "recovery", "tune", "stability", "final"}:
        raise ValueError(f"unknown stage: {stage}")

    # A full dry-run deliberately stops at the first hard gate. Later-stage
    # commands depend on measured recovery/search results and are not invented.
    tasks = (only_task,) if only_task is not None else tuple(HISTORICAL_BASELINE)
    commands = [
        build_command(
            config,
            BASELINE_TUNING_OVERRIDES,
            task=task,
            stream_seed=2025,
        )
        for task in tasks
    ]
    command_dir = Path(run_dir) / "commands"
    command_dir.mkdir(parents=True, exist_ok=True)
    for task, command in zip(tasks, commands, strict=True):
        source, target = task
        (command_dir / f"dry_run_recovery_{source}to{target}.txt").write_text(
            shlex.join(command) + "\n", encoding="utf-8"
        )
    atomic_write_json(
        Path(run_dir) / "dry_run.json",
        {
            "stage": stage,
            "gate": "recovery",
            "task_count": len(commands),
            "created_at": _utc_now(),
        },
    )
    return commands


def atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_task_record(path: Path, record: TaskRecord) -> None:
    atomic_write_json(path, asdict(record))


def load_task_record(path: Path) -> TaskRecord:
    value = json.loads(path.read_text(encoding="utf-8"))
    value["task"] = tuple(value["task"])
    return TaskRecord(**value)


def classify_failure(
    returncode: int, log_text: str
) -> Literal["transient", "permanent"]:
    if returncode == 0:
        return "permanent"
    normalized = log_text.casefold()
    transient_markers = (
        "cuda initialization error",
        "cuda driver initialization failed",
        "cuda error: system not yet initialized",
    )
    if any(marker in normalized for marker in transient_markers):
        return "transient"
    return "permanent"


def can_launch(
    deadline: float,
    now: float,
    estimated_seconds: float,
    reserve_seconds: float,
) -> bool:
    return now + estimated_seconds + reserve_seconds <= deadline


def should_skip(record: TaskRecord) -> bool:
    if record.status != "succeeded" or record.returncode != 0 or not record.metrics:
        return False
    path = Path(record.log_path)
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8", errors="ignore")
    required_patterns = (
        r"Beginning Acc T\s*=\s*[0-9.]+%",
        r"Strict Online Acc\s*=\s*[0-9.]+%",
        r"Post-stream Full-Target Acc\s*=\s*[0-9.]+%",
        r"\[STRICT 0711\]\s+batches=\d+",
        r"mean_batch_ms=[0-9.]+",
    )
    return all(re.search(pattern, text) for pattern in required_patterns)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _task_stem(
    stage: str, candidate: Mapping[str, object], task: tuple[int, int], stream_seed: int
) -> tuple[str, str]:
    identifier = candidate_id(candidate, 2025, stream_seed)
    source, target = task
    return identifier, f"{stage}_{identifier}_{source}to{target}_stream{stream_seed}"


def _record_metrics_dict(metrics: object) -> dict[str, object]:
    return {
        "before": float(metrics.before),
        "strict_online": float(metrics.strict_online),
        "post_stream": float(metrics.post_stream),
        "batches": int(metrics.batches),
        "runtime_seconds": float(metrics.runtime_seconds),
    }


def _metrics_from_record(record: TaskRecord):
    if record.metrics is None:
        raise ValueError(f"task record has no metrics: {record.task}")
    return _summary_module().RunnerMetrics(**record.metrics)


def execute_task(
    run_dir: Path,
    config: Mapping[str, object],
    stage: str,
    candidate: Mapping[str, object],
    task: tuple[int, int],
    stream_seed: int,
    runner: object | None = None,
) -> TaskRecord:
    validate_config(config)
    task = _task_tuple(task)
    run_dir = Path(run_dir)
    identifier, stem = _task_stem(stage, candidate, task, stream_seed)
    state_path = run_dir / "state" / f"{stem}.json"
    log_path = (run_dir / "logs" / f"{stem}.log").resolve()
    command_path = run_dir / "commands" / f"{stem}.txt"
    command = build_command(config, candidate, task, stream_seed)

    if state_path.is_file():
        existing = load_task_record(state_path)
        if should_skip(existing):
            return existing
        initial_attempt = existing.attempt + 1
    else:
        initial_attempt = 1

    command_path.parent.mkdir(parents=True, exist_ok=True)
    command_path.write_text(shlex.join(command) + "\n", encoding="utf-8")
    active_runner = runner if runner is not None else SubprocessTaskRunner()
    environment = os.environ.copy()
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = "0"

    for attempt in range(initial_attempt, min(initial_attempt + 2, 3)):
        started_at = _utc_now()
        running = TaskRecord(
            candidate_id=identifier,
            stage=stage,
            task=task,
            overrides=normalize_overrides(candidate),
            stream_seed=int(stream_seed),
            command=command,
            status="running",
            attempt=attempt,
            started_at=started_at,
            ended_at=None,
            returncode=None,
            log_path=str(log_path),
            metrics=None,
            failure_class=None,
        )
        write_task_record(state_path, running)
        returncode = int(active_runner.run(command, log_path, environment))
        ended_at = _utc_now()

        parse_error = None
        parsed = None
        if returncode == 0:
            try:
                parsed = _summary_module().parse_runner_log(log_path)
            except (OSError, ValueError) as exc:
                parse_error = str(exc)
        if returncode == 0 and parsed is not None:
            succeeded = TaskRecord(
                **{
                    **asdict(running),
                    "status": "succeeded",
                    "ended_at": ended_at,
                    "returncode": 0,
                    "metrics": _record_metrics_dict(parsed),
                }
            )
            write_task_record(state_path, succeeded)
            return succeeded

        log_text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
        if parse_error:
            log_text = f"{log_text}\n{parse_error}"
        failure_class = classify_failure(returncode, log_text)
        failed = TaskRecord(
            **{
                **asdict(running),
                "status": "failed",
                "ended_at": ended_at,
                "returncode": returncode,
                "failure_class": failure_class,
            }
        )
        write_task_record(state_path, failed)
        if failure_class != "transient" or attempt >= 2:
            return failed
    return failed


def _require_recovery_gate(run_dir: Path) -> None:
    gate_path = Path(run_dir) / "recovery_gate.json"
    if not gate_path.is_file():
        raise RuntimeError("recovery gate has not been completed")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("passed") is not True:
        raise RuntimeError("recovery gate did not pass")


def run_recovery_gate(
    run_dir: Path,
    config: Mapping[str, object],
    runner: object | None = None,
) -> object:
    validate_config(config)
    records = [
        execute_task(
            run_dir,
            config,
            stage="recovery",
            candidate=BASELINE_TUNING_OVERRIDES,
            task=task,
            stream_seed=2025,
            runner=runner,
        )
        for task in HISTORICAL_BASELINE
    ]
    metrics = {
        record.task: _metrics_from_record(record)
        for record in records
        if record.status == "succeeded"
    }
    recovery_cfg = config["recovery"]
    assert isinstance(recovery_cfg, Mapping)
    result = _summary_module().validate_recovery(
        metrics,
        HISTORICAL_BASELINE,
        float(recovery_cfg["expected_mean"]),
        float(recovery_cfg["tolerance"]),
    )
    atomic_write_json(
        Path(run_dir) / "recovery_gate.json",
        {
            "passed": bool(result.passed),
            "exact_mean": result.exact_mean,
            "mean_delta": result.mean_delta,
            "task_failures": {
                f"{source}to{target}": delta
                for (source, target), delta in result.task_failures.items()
            },
            "completed_at": _utc_now(),
        },
    )
    if not result.passed:
        raise RuntimeError(
            f"PU4D recovery gate failed: mean={result.exact_mean}, "
            f"task_failures={result.task_failures}"
        )
    return result


def _aggregate_records(records: list[TaskRecord], required_tasks):
    rows = [
        {
            "candidate_id": record.candidate_id,
            "task": record.task,
            "overrides": record.overrides,
            "metrics": _metrics_from_record(record),
        }
        for record in records
        if record.status == "succeeded"
    ]
    return _summary_module().aggregate_candidate(rows, required_tasks)


def _load_candidate_aggregate(
    run_dir: Path,
    candidate: Mapping[str, object],
    stream_seed: int,
):
    identifier = candidate_id(candidate, 2025, stream_seed)
    records = []
    for path in (Path(run_dir) / "state").glob(
        f"*_{identifier}_*stream{stream_seed}.json"
    ):
        record = load_task_record(path)
        if (
            record.status == "succeeded"
            and record.candidate_id == identifier
            and record.stream_seed == stream_seed
            and record.task in DEV_TASKS
        ):
            records.append(record)
    return _aggregate_records(records, DEV_TASKS)


def run_coordinate_search(
    run_dir: Path,
    config: Mapping[str, object],
    runner: object | None = None,
    deadline: float | None = None,
) -> dict[str, object]:
    _require_recovery_gate(run_dir)
    validate_config(config)
    budget = config["budget"]
    assert isinstance(budget, Mapping)
    estimated_seconds = float(budget["estimated_task_minutes"]) * 60.0
    reserve_seconds = float(budget["reserve_minutes"]) * 60.0
    if deadline is None:
        deadline = time.monotonic() + float(budget["tuning_hours"]) * 3600.0

    anchor = dict(BASELINE_TUNING_OVERRIDES)
    all_aggregates_by_id = {}
    search = config["search"]
    assert isinstance(search, Mapping)
    for group in search["groups"]:
        candidates = expand_coordinate_group(anchor, group["values"])
        group_aggregates = []
        for candidate in candidates:
            existing_aggregate = _load_candidate_aggregate(
                run_dir, candidate, stream_seed=2025
            )
            if existing_aggregate is not None:
                group_aggregates.append(existing_aggregate)
                all_aggregates_by_id.setdefault(
                    existing_aggregate.candidate_id, existing_aggregate
                )
                continue
            candidate_records = []
            for task in DEV_TASKS:
                if not can_launch(
                    deadline, time.monotonic(), estimated_seconds, reserve_seconds
                ):
                    break
                candidate_records.append(
                    execute_task(
                        run_dir,
                        config,
                        stage=f"search_{group['name']}",
                        candidate=candidate,
                        task=task,
                        stream_seed=2025,
                        runner=runner,
                    )
                )
            aggregate = _aggregate_records(candidate_records, DEV_TASKS)
            if aggregate is not None:
                group_aggregates.append(aggregate)
                all_aggregates_by_id.setdefault(aggregate.candidate_id, aggregate)
        if group_aggregates:
            winner = sorted(
                group_aggregates,
                key=lambda item: (
                    -item.mean_strict_online,
                    -item.minimum_task_delta,
                    item.wall_time_seconds,
                    item.candidate_id,
                ),
            )[0]
            anchor = dict(winner.overrides)

    all_aggregates = list(all_aggregates_by_id.values())
    finalists = [
        dict(item.overrides)
        for item in sorted(
            all_aggregates,
            key=lambda item: (-item.mean_strict_online, -item.minimum_task_delta),
        )[:2]
    ]
    return {"anchor": anchor, "leaderboard": all_aggregates, "finalists": finalists}


def run_stability_selection(
    run_dir: Path,
    config: Mapping[str, object],
    finalists: list[Mapping[str, object]],
    runner: object | None = None,
) -> dict[str, object]:
    selection = config["selection"]
    assert isinstance(selection, Mapping)
    stream_seed = int(selection["stability_stream_seed"])
    aggregates_2026 = []
    for candidate in finalists:
        records = [
            execute_task(
                run_dir,
                config,
                stage="stability",
                candidate=candidate,
                task=task,
                stream_seed=stream_seed,
                runner=runner,
            )
            for task in DEV_TASKS
        ]
        aggregate = _aggregate_records(records, DEV_TASKS)
        if aggregate is not None:
            aggregates_2026.append(aggregate)
    baseline_marker = normalize_overrides(BASELINE_TUNING_OVERRIDES)
    baseline_2026 = next(
        (
            aggregate
            for aggregate in aggregates_2026
            if normalize_overrides(aggregate.overrides) == baseline_marker
        ),
        None,
    )
    baseline_2025 = _load_candidate_aggregate(
        run_dir, BASELINE_TUNING_OVERRIDES, stream_seed=2025
    )
    if baseline_2025 is None or baseline_2026 is None:
        raise RuntimeError("stability selection requires a matching 2026 baseline")

    ranked = []
    minimum_gain = float(selection["minimum_mean_gain"])
    maximum_regression = float(selection["maximum_task_regression"])
    for candidate_2026 in aggregates_2026:
        if candidate_2026 is baseline_2026:
            continue
        candidate_2025 = _load_candidate_aggregate(
            run_dir, candidate_2026.overrides, stream_seed=2025
        )
        if candidate_2025 is None:
            continue
        mean_gain_2025 = (
            candidate_2025.mean_strict_online - baseline_2025.mean_strict_online
        )
        deltas = []
        for task in DEV_TASKS:
            deltas.append(
                candidate_2025.task_metrics[task].strict_online
                - baseline_2025.task_metrics[task].strict_online
            )
            deltas.append(
                candidate_2026.task_metrics[task].strict_online
                - baseline_2026.task_metrics[task].strict_online
            )
        if mean_gain_2025 < minimum_gain or min(deltas) < -maximum_regression:
            continue
        ranked.append(
            (
                sum(deltas) / len(deltas),
                min(deltas),
                -(candidate_2025.wall_time_seconds + candidate_2026.wall_time_seconds),
                candidate_2026,
            )
        )
    if not ranked:
        return dict(BASELINE_TUNING_OVERRIDES)
    winner = max(ranked, key=lambda item: (item[0], item[1], item[2], item[3].candidate_id))
    return dict(winner[3].overrides)


def build_freeze_metadata(
    run_dir: Path,
    selected: Mapping[str, object],
    preflight: Mapping[str, object],
) -> dict[str, object]:
    selection_metrics: dict[str, object] = {}
    for stream_seed in (2025, 2026):
        aggregate = _load_candidate_aggregate(run_dir, selected, stream_seed)
        if aggregate is None:
            raise RuntimeError(
                f"cannot freeze configuration without complete stream {stream_seed} "
                "development metrics"
            )
        selection_metrics[str(stream_seed)] = {
            "candidate_id": aggregate.candidate_id,
            "mean_strict_online": aggregate.mean_strict_online,
            "minimum_task_delta": aggregate.minimum_task_delta,
            "tasks": {
                f"{source}to{target}": metrics.strict_online
                for (source, target), metrics in sorted(aggregate.task_metrics.items())
            },
        }
    return {
        "selected_by": "two-stream stability",
        "frozen_at": _utc_now(),
        "source_seed": 2025,
        "stream_seeds": [2025, 2026],
        "selection_metrics": selection_metrics,
        "config_sha256": preflight.get("config_sha256"),
        "checkpoints": list(preflight.get("checkpoints", [])),
    }


def freeze_best_config(
    path: Path,
    overrides: Mapping[str, object],
    metadata: Mapping[str, object],
) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"frozen config already exists: {path}")
    payload = {
        "overrides": normalize_overrides(overrides),
        "metadata": dict(metadata),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            yaml.safe_dump(payload, sort_keys=True, allow_unicode=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_final_validation(
    run_dir: Path,
    config: Mapping[str, object],
    frozen_overrides: Mapping[str, object],
    runner: object | None = None,
) -> list[TaskRecord]:
    _require_recovery_gate(run_dir)
    if not (Path(run_dir) / "best_config.yaml").is_file():
        raise RuntimeError("best configuration must be frozen before final validation")
    return [
        execute_task(
            run_dir,
            config,
            stage="final",
            candidate=frozen_overrides,
            task=task,
            stream_seed=2025,
            runner=runner,
        )
        for task in HELDOUT_TASKS
    ]


def run_pipeline(
    config: Mapping[str, object],
    run_dir: Path,
    runner: object | None = None,
) -> dict[str, object]:
    recovery = run_recovery_gate(run_dir, config, runner=runner)
    search_result = run_coordinate_search(run_dir, config, runner=runner)
    finalists = [dict(BASELINE_TUNING_OVERRIDES), *search_result["finalists"]]
    unique_finalists = []
    seen = set()
    for candidate in finalists:
        marker = json.dumps(normalize_overrides(candidate), sort_keys=True)
        if marker not in seen:
            seen.add(marker)
            unique_finalists.append(candidate)
    selected = run_stability_selection(
        run_dir, config, finalists=unique_finalists[:3], runner=runner
    )
    freeze_best_config(
        Path(run_dir) / "best_config.yaml",
        selected,
        build_freeze_metadata(
            run_dir,
            selected,
            json.loads((Path(run_dir) / "preflight.json").read_text(encoding="utf-8")),
        ),
    )
    final_records = run_final_validation(
        run_dir, config, frozen_overrides=selected, runner=runner
    )
    return {
        "recovery": recovery,
        "search": search_result,
        "selected": selected,
        "final_records": final_records,
    }


def _aggregate_payload(aggregate: object) -> dict[str, object]:
    return {
        "candidate_id": aggregate.candidate_id,
        "overrides": dict(aggregate.overrides),
        "mean_strict_online": float(aggregate.mean_strict_online),
        "minimum_task_delta": float(aggregate.minimum_task_delta),
        "wall_time_seconds": float(aggregate.wall_time_seconds),
    }


def _unique_candidates(candidates: list[Mapping[str, object]]) -> list[dict[str, object]]:
    unique = []
    seen = set()
    for candidate in candidates:
        normalized = normalize_overrides(candidate)
        marker = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        if marker not in seen:
            seen.add(marker)
            unique.append(normalized)
    return unique


def _run_probe(
    run_dir: Path,
    config: Mapping[str, object],
    task: tuple[int, int],
) -> TaskRecord:
    record = execute_task(
        run_dir,
        config,
        stage="recovery",
        candidate=BASELINE_TUNING_OVERRIDES,
        task=task,
        stream_seed=2025,
    )
    if record.status != "succeeded" or record.metrics is None:
        raise RuntimeError(f"recovery probe {task} failed; see {record.log_path}")
    strict_online = float(record.metrics["strict_online"])
    tolerance = float(config["recovery"]["tolerance"])
    delta = strict_online - HISTORICAL_BASELINE[task]
    passed = abs(delta) <= tolerance
    atomic_write_json(
        Path(run_dir) / f"recovery_probe_{task[0]}to{task[1]}.json",
        {
            "passed": passed,
            "task": list(task),
            "strict_online": strict_online,
            "expected": HISTORICAL_BASELINE[task],
            "delta": delta,
            "completed_at": _utc_now(),
        },
    )
    if not passed:
        raise RuntimeError(
            f"recovery probe {task} missed baseline: {strict_online:.4f}% "
            f"vs {HISTORICAL_BASELINE[task]:.4f}%"
        )
    return record


def _resolve_run_dir(project_root: Path, requested: str | None, resume: bool) -> Path:
    marker = project_root / "logs" / "latest_pu4d_0711_tuning_run_dir.txt"
    if requested:
        run_dir = Path(requested)
        if not run_dir.is_absolute():
            run_dir = project_root / run_dir
    elif resume:
        if not marker.is_file():
            raise RuntimeError("--resume requires --run-dir or an existing latest-run marker")
        run_dir = Path(marker.read_text(encoding="utf-8").strip())
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = project_root / "logs" / f"PU4D_0711_STRICT_TUNING_{stamp}"
    run_dir = run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()) and not resume:
        raise RuntimeError(f"run directory is not empty; use --resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_name(f"{marker.name}.{os.getpid()}.tmp")
    temporary.write_text(str(run_dir) + "\n", encoding="utf-8")
    os.replace(temporary, marker)
    return run_dir


def _load_frozen_overrides(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("overrides"), Mapping):
        raise RuntimeError(f"invalid frozen configuration: {path}")
    return normalize_overrides(payload["overrides"])


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict PU4D 0711 recovery and tuning")
    parser.add_argument(
        "--config",
        default="Configs/Experiments/PU4D0711_strict_tuning.yaml",
    )
    parser.add_argument("--run-dir")
    parser.add_argument(
        "--stage",
        choices=("all", "recovery", "tune", "stability", "final"),
        default="all",
    )
    parser.add_argument("--task", type=parse_task_name)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    config = load_config(config_path)
    validate_config(config)
    if args.task is not None and args.stage != "recovery":
        raise ValueError("--task is only allowed with --stage recovery")
    run_dir = _resolve_run_dir(project_root, args.run_dir, args.resume)
    preflight = run_preflight(project_root, config, check_gpu=True)
    atomic_write_json(run_dir / "preflight.json", preflight)
    print(f"[PU4D 0711] run_dir={run_dir}", flush=True)
    print(f"[PU4D 0711] GPU0={preflight['gpu0']}", flush=True)

    if args.dry_run:
        commands = prepare_dry_run(
            run_dir, config, stage=args.stage, only_task=args.task
        )
        print(f"[DRY RUN] wrote {len(commands)} recovery command(s)", flush=True)
        return 0

    if args.stage == "recovery":
        if args.task is not None:
            record = _run_probe(run_dir, config, args.task)
            print(
                f"[RECOVERY PROBE PASS] {args.task[0]}to{args.task[1]} "
                f"strict_online={record.metrics['strict_online']:.4f}%",
                flush=True,
            )
        else:
            result = run_recovery_gate(run_dir, config)
            print(f"[RECOVERY GATE PASS] mean={result.exact_mean:.4f}%", flush=True)
        return 0

    if args.stage == "tune":
        _require_recovery_gate(run_dir)
        budget_path = run_dir / "tuning_budget.json"
        if budget_path.is_file():
            budget_state = json.loads(budget_path.read_text(encoding="utf-8"))
            deadline_epoch = float(budget_state["deadline_epoch"])
        else:
            deadline_epoch = time.time() + float(config["budget"]["tuning_hours"]) * 3600.0
            atomic_write_json(
                budget_path,
                {"started_at": _utc_now(), "deadline_epoch": deadline_epoch},
            )
        remaining = max(0.0, deadline_epoch - time.time())
        result = run_coordinate_search(
            run_dir,
            config,
            deadline=time.monotonic() + remaining,
        )
        atomic_write_json(
            run_dir / "search_result.json",
            {
                "anchor": result["anchor"],
                "finalists": result["finalists"],
                "leaderboard": [
                    _aggregate_payload(item) for item in result["leaderboard"]
                ],
                "completed_at": _utc_now(),
            },
        )
        print(f"[TUNING COMPLETE] finalists={len(result['finalists'])}", flush=True)
        return 0

    if args.stage == "stability":
        _require_recovery_gate(run_dir)
        search_path = run_dir / "search_result.json"
        if not search_path.is_file():
            raise RuntimeError("stability stage requires search_result.json")
        search_result = json.loads(search_path.read_text(encoding="utf-8"))
        finalists = _unique_candidates(
            [dict(BASELINE_TUNING_OVERRIDES), *search_result["finalists"]]
        )[:3]
        selected = run_stability_selection(run_dir, config, finalists=finalists)
        frozen_path = run_dir / "best_config.yaml"
        if frozen_path.exists():
            if _load_frozen_overrides(frozen_path) != normalize_overrides(selected):
                raise RuntimeError("existing frozen configuration differs from selection")
        else:
            freeze_best_config(
                frozen_path,
                selected,
                build_freeze_metadata(run_dir, selected, preflight),
            )
        print(f"[STABILITY COMPLETE] selected={json.dumps(selected, sort_keys=True)}", flush=True)
        return 0

    if args.stage == "final":
        frozen_path = run_dir / "best_config.yaml"
        if not frozen_path.is_file():
            raise RuntimeError("final stage requires best_config.yaml")
        selected = _load_frozen_overrides(frozen_path)
        records = run_final_validation(run_dir, config, selected)
        if any(record.status != "succeeded" for record in records):
            raise RuntimeError("one or more held-out final tasks failed")
        final_mean = sum(
            float(record.metrics["strict_online"]) for record in records if record.metrics
        ) / len(records)
        summary_result = _summary_module().generate_run_outputs(
            run_dir, regression_tests_passed=False
        )
        atomic_write_json(
            run_dir / "final_result.json",
            {
                "heldout_mean": final_mean,
                "exact_12task_mean": summary_result["exact_mean"],
                "recommended": summary_result["recommended"],
                "task_count": len(records),
                "completed_at": _utc_now(),
            },
        )
        print(f"[FINAL COMPLETE] heldout_mean={final_mean:.4f}%", flush=True)
        return 0

    run_pipeline(config, run_dir)
    print("[PIPELINE COMPLETE]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
