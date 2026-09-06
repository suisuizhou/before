#!/usr/bin/env python3
"""Parse HUST route logs and emit reproducible comparison artifacts."""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
import json
import math
import os
from pathlib import Path
import re
from typing import Any

import yaml


_MACHINE_COMMON_FIELDS = frozenset({
    "schema_version", "result_kind", "route", "variant", "source", "target", "task",
    "source_checkpoint_sha256", "config_sha256", "candidate_id", "source_seed",
    "stream_seed", "beginning", "strict_online", "post_stream", "macro_precision",
    "macro_recall", "macro_f1", "confusion_matrix", "samples", "batches", "passes",
    "class_coverage", "finite_losses", "trainable_parameters", "trainable_allowlist",
    "pre_update_scoring", "metadata_evidence_used", "runtime_seconds", "peak_memory_mb",
})
_MACHINE_TARGET_DIAGNOSTICS = frozenset({
    "memory_size", "memory_class_coverage", "certain_ratio", "uncertain_ratio",
    "offline_purity", "certain_purity", "evidence_applicable", "evidence_active_ratio",
    "mean_batch_ms",
})
_MACHINE_POST_STREAM_DIAGNOSTICS = frozenset({
    "post_macro_precision", "post_macro_recall", "post_macro_f1",
})
_MACHINE_0711_DIAGNOSTICS = frozenset({"offline_confusion_matrix"})
_REPORT_METRIC_FIELDS = frozenset({
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
_METRIC_IDENTITY_FIELDS = (
    "route", "variant", "candidate_id", "task", "source", "target", "source_seed",
    "stream_seed", "config_sha256", "source_checkpoint_sha256",
)


def flatten_state_metrics(state: Mapping[str, object]) -> dict[str, object]:
    """Expose reviewed report metrics without allowing nested state-control replacement."""
    value = dict(state)
    metrics = value.get("metrics")
    if value.get("kind") != "target" or not isinstance(metrics, Mapping):
        return value
    mismatched = {
        key: (value.get(key), metrics.get(key))
        for key in _METRIC_IDENTITY_FIELDS
        if key in metrics and value.get(key) != metrics.get(key)
    }
    result_kind = metrics.get("result_kind")
    expected_kind = value.get("expected_result_kind")
    if result_kind is not None and expected_kind is not None and result_kind != expected_kind:
        mismatched["result_kind"] = (expected_kind, result_kind)
    if mismatched:
        raise ValueError(f"persisted metric identity mismatch: {mismatched}")
    value.update({key: metrics[key] for key in _REPORT_METRIC_FIELDS if key in metrics})
    return value


def _sha256_file(path: Path) -> str:
    digest = __import__("hashlib").sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_saved_evidence(records: Sequence[Mapping[str, object]], checkpoint_manifest: Mapping[str, object]) -> None:
    """Fail reporting when persisted formal evidence no longer matches its hashes."""
    for row in records:
        if row.get("status") != "succeeded":
            continue
        if row.get("_formal_state"):
            _validate_formal_state(row)
        for path_key, hash_key in (("log_path", "log_sha256"), ("command_path", "command_log_sha256")):
            if path_key not in row:
                continue
            path = Path(str(row[path_key]))
            if not path.is_file() or row.get(hash_key) != _sha256_file(path):
                raise ValueError(f"saved evidence {path_key} hash mismatch")
        for hashes_key in ("artifact_hashes", "output_artifact_hashes"):
            for raw_path, expected_hash in dict(row.get(hashes_key, {})).items():
                path = Path(str(raw_path))
                if not path.is_file() or _sha256_file(path) != expected_hash:
                    raise ValueError(f"saved {hashes_key} mismatch: {path}")
    def walk(value):
        if isinstance(value, Mapping):
            for key, nested in value.items():
                path = Path(str(key))
                if isinstance(nested, str) and len(nested) == 64 and (path.suffix in {".pt", ".json", ".yaml"} or "/" in str(key)):
                    if not path.is_file() or _sha256_file(path) != nested:
                        raise ValueError(f"checkpoint manifest hash mismatch: {path}")
                else:
                    walk(nested)
    walk(checkpoint_manifest)


def _validate_formal_state(row: Mapping[str, object]) -> None:
    common = {
        "kind", "stage", "route", "variant", "candidate_id", "source", "source_seed",
        "config_sha256", "command", "command_sha256", "command_path", "command_log_sha256",
        "artifacts", "artifact_hashes", "gpu", "status", "attempt", "started_at", "ended_at",
        "returncode", "log_path", "log_sha256", "metrics", "cache_manifest_path",
        "cache_manifest_sha256", "runner_script_path", "runner_script_sha256",
        "experiment_config_path", "experiment_config_sha256", "source_checkpoint_path",
        "source_checkpoint_sha256", "source_summary_path", "source_summary_sha256",
        "expected_result_contract", "expected_result_kind", "cache_content_sha256",
        "cache_tensor_sha256s",
    }
    target = {"task", "target", "stream_seed", "result_identity", "load_split"}
    source = {"output_artifact_hashes", "load_split"}
    required = common | (source if row.get("kind") == "source" else target)
    missing = required - set(row)
    if missing:
        raise ValueError(f"formal evidence missing required keys: {sorted(missing)}")
    forbidden = (target - {"load_split"}) if row.get("kind") == "source" else (source - {"load_split"})
    unexpected = forbidden & set(row)
    if unexpected:
        raise ValueError(f"formal evidence has unexpected security keys: {sorted(unexpected)}")
    if row.get("kind") not in {"source", "target"} or row.get("status") != "succeeded" or row.get("returncode") != 0:
        raise ValueError("formal evidence state/result contract failed")
    command = list(row["command"])
    canonical_command_hash = __import__("hashlib").sha256(json.dumps(command, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if row["command_sha256"] != canonical_command_hash:
        raise ValueError("formal evidence command identity mismatch")
    for path_key, hash_key in (("command_path", "command_log_sha256"), ("log_path", "log_sha256")):
        path = Path(str(row[path_key]))
        if not path.is_file() or row[hash_key] != _sha256_file(path):
            raise ValueError(f"formal evidence {path_key} hash mismatch")
    named = {
        str(Path(str(row[path_key])).resolve()): str(row[hash_key])
        for path_key, hash_key in (
            ("cache_manifest_path", "cache_manifest_sha256"),
            ("runner_script_path", "runner_script_sha256"),
            ("experiment_config_path", "experiment_config_sha256"),
        )
    }
    cache_tensors = {
        str(Path(str(path)).resolve()): str(digest)
        for path, digest in dict(row["cache_tensor_sha256s"]).items()
    }
    if not cache_tensors or any(
        not re.fullmatch(r"[0-9a-f]{64}", digest)
        for digest in cache_tensors.values()
    ):
        raise ValueError("formal evidence cache tensor identity missing")
    try:
        from Lib.hust_strict_protocol import validate_cache
    except ModuleNotFoundError:
        from hust_strict_protocol import validate_cache
    cache = validate_cache(Path(str(row["cache_manifest_path"])).parent)
    validated_tensors = {
        str((Path(str(row["cache_manifest_path"])).parent / str(summary["tensor_file"])).resolve()): str(summary["tensor_sha256"])
        for summary in cache["domains"].values()
    }
    if (
        cache.get("version") != 2
        or cache.get("content_sha256") != row["cache_content_sha256"]
        or validated_tensors != cache_tensors
    ):
        raise ValueError("formal evidence cache content identity mismatch")
    named.update(cache_tensors)
    if row["kind"] == "target":
        named.update({
            str(Path(str(row["source_checkpoint_path"])).resolve()): str(row["source_checkpoint_sha256"]),
            str(Path(str(row["source_summary_path"])).resolve()): str(row["source_summary_sha256"]),
        })
    if row.get("freeze_config_path") is not None:
        freeze_required = {
            "freeze_config_path", "freeze_config_sha256", "freeze_sha256",
            "freeze_proof_path", "freeze_proof_file_sha256", "freeze_proof_sha256",
            "freeze_frozen_at",
        }
        if not freeze_required <= set(row):
            raise ValueError("formal freeze evidence is incomplete")
        if row["kind"] != "target":
            raise ValueError("formal freeze evidence attached to the wrong job")
        named.update({
            str(Path(str(row["freeze_config_path"])).resolve()): str(row["freeze_config_sha256"]),
            str(Path(str(row["freeze_proof_path"])).resolve()): str(row["freeze_proof_file_sha256"]),
        })
        try:
            from tools.tune_hust_0711_strict import load_frozen_candidate
        except ModuleNotFoundError:
            from tune_hust_0711_strict import load_frozen_candidate
        frozen = load_frozen_candidate(Path(str(row["freeze_config_path"])))
        if (
            frozen["candidate_id"] != row["candidate_id"]
            or dict(row.get("overrides", {})) != dict(frozen["overrides"])
            or frozen["freeze_sha256"] != row["freeze_sha256"]
            or frozen["proof_sha256"] != row["freeze_proof_sha256"]
            or float(row["started_at"]) < float(row["freeze_frozen_at"])
            or float(row["freeze_frozen_at"]) != float(frozen["frozen_at"])
        ):
            raise ValueError("formal frozen state predates or mismatches its authority")
    elif any(
        row.get(key) is not None
        for key in (
            "freeze_config_sha256", "freeze_sha256", "freeze_proof_path",
            "freeze_proof_file_sha256", "freeze_proof_sha256", "freeze_frozen_at",
        )
    ):
        raise ValueError("partial formal freeze evidence")
    artifact_hashes = {str(Path(path).resolve()): str(digest) for path, digest in dict(row["artifact_hashes"]).items()}
    if set(map(lambda value: str(Path(str(value)).resolve()), row["artifacts"])) != set(named) or artifact_hashes != named:
        raise ValueError("formal evidence input artifact set mismatch")
    if any(not Path(path).is_file() or _sha256_file(Path(path)) != digest for path, digest in named.items()):
        raise ValueError("formal evidence input artifact hash mismatch")
    if row["kind"] == "target":
        identity = dict(row["result_identity"])
        expected_identity = {
            "candidate_id": row["candidate_id"], "route": row["route"], "variant": row["variant"],
            "task": list(row["task"]), "source": int(row["source"]), "target": int(row["target"]),
            "source_seed": int(row["source_seed"]), "stream_seed": int(row["stream_seed"]),
            "config_sha256": row["config_sha256"], "result_kind": row["expected_result_kind"],
            "source_checkpoint_sha256": row["source_checkpoint_sha256"],
        }
        if identity != expected_identity or row["expected_result_contract"] != row["expected_result_kind"]:
            raise ValueError("formal evidence target state identity mismatch")
        parsed = parse_runner_log(Path(str(row["log_path"])), str(row["route"]), expected=identity)
        if parsed != row["metrics"]:
            raise ValueError("formal evidence parsed metrics mismatch")
    else:
        if row["expected_result_kind"] != "source" or row["expected_result_contract"] != "source":
            raise ValueError("formal source evidence result-kind mismatch")
        outputs = {str(Path(path).resolve()): str(digest) for path, digest in dict(row["output_artifact_hashes"]).items()}
        checkpoint_path = str(Path(str(row["source_checkpoint_path"])).resolve())
        summary_path = str(Path(str(row["source_summary_path"])).resolve())
        if set(outputs) != {checkpoint_path, summary_path}:
            raise ValueError("formal source evidence output artifact set mismatch")
        if (outputs[checkpoint_path] != row["source_checkpoint_sha256"]
                or outputs[summary_path] != row["source_summary_sha256"]):
            raise ValueError("formal source evidence persisted output hash mismatch")
        if any(not Path(path).is_file() or _sha256_file(Path(path)) != digest for path, digest in outputs.items()):
            raise ValueError("formal source evidence output hash mismatch")
        summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
        expected = {"route": row["variant"], "source": int(row["source"]), "seed": int(row["source_seed"])}
        if any(summary.get(key) != value for key, value in expected.items()) or summary.get("target_labels_consumed") is not False or summary.get("checkpoint_sha256") != outputs[checkpoint_path] or summary != row["metrics"]:
            raise ValueError("formal source evidence summary identity/metrics mismatch")


def _range(rows: Sequence[Mapping[str, object]], key: str) -> dict[str, float | None]:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return {"mean": sum(values) / len(values), "min": min(values), "max": max(values)} if values else {"mean": None, "min": None, "max": None}


def _markdown_table(rows: Sequence[Mapping[str, object]]) -> list[str]:
    lines = ["| stage | seed | route | variant | candidate | task | Beginning | strict acc | macro-F1 |", "|---|---:|---|---|---|---|---:|---:|---:|"]
    for row in rows:
        task = row.get("task", ["?", "?"])
        def value(key):
            return "NA" if row.get(key) is None else f"{float(row[key]):.4f}"
        lines.append(f"| {row.get('stage', '')} | {row.get('stream_seed', '')} | {row.get('route', '')} | {row.get('variant', '')} | {row.get('candidate_id', '')} | {task[0]}→{task[1]} | {value('before')} | {value('strict_online')} | {value('macro_f1')} |")
    return lines


def _last(pattern: str, text: str, *, flags: int = 0) -> float | None:
    matches = re.findall(pattern, text, flags)
    return float(matches[-1]) if matches else None


def parse_runner_log(path: Path, route: str, expected: Mapping[str, object] | None = None) -> dict[str, object]:
    if route not in {"dtcc_ordinary", "0711_robust", "0711_common", "0711_robust_baseline"}:
        raise ValueError(f"unknown route: {route}")
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    machine_lines = re.findall(r"^HUST_RESULT_JSON=(\{.*\})$", text, re.MULTILINE)
    if machine_lines:
        try:
            result = json.loads(machine_lines[-1])
        except json.JSONDecodeError as exc:
            raise ValueError("invalid HUST_RESULT_JSON") from exc
        result_kind = result.get("result_kind")
        if result_kind not in {"target", "beginning"}:
            raise ValueError("HUST result route/schema identity mismatch")
        target_required = {
            "strict_online", "post_stream", "macro_precision", "macro_recall", "macro_f1",
            "confusion_matrix", "class_coverage", "finite_losses", "trainable_parameters",
            "trainable_allowlist", "pre_update_scoring", "metadata_evidence_used",
            "memory_size", "memory_class_coverage", "certain_ratio", "uncertain_ratio",
            "offline_purity", "certain_purity", "evidence_applicable", "evidence_active_ratio",
            "post_macro_precision", "post_macro_recall", "post_macro_f1",
        }
        allowed = _MACHINE_COMMON_FIELDS
        if result_kind == "target":
            allowed |= _MACHINE_TARGET_DIAGNOSTICS | _MACHINE_POST_STREAM_DIAGNOSTICS
            if route.startswith("0711"):
                allowed |= _MACHINE_0711_DIAGNOSTICS
                target_required |= _MACHINE_0711_DIAGNOSTICS
        missing, unexpected = allowed - set(result), set(result) - allowed
        if missing or unexpected:
            raise ValueError(f"HUST_RESULT_JSON exact schema failed; missing={sorted(missing)} unexpected={sorted(unexpected)}")
        if result["schema_version"] != 1 or result["route"] != route or result.get("result_kind") not in {"target", "beginning"}:
            raise ValueError("HUST result route/schema identity mismatch")
        if (result["route"], result["variant"]) not in {("dtcc_ordinary", "ordinary"), ("0711_robust", "robust"), ("0711_common", "ordinary")}:
            raise ValueError("formal HUST route/variant pairing failed")
        if len(str(result["source_checkpoint_sha256"])) != 64 or len(str(result["config_sha256"])) != 64:
            raise ValueError("formal HUST artifact/config hash missing")
        if not (0.0 <= float(result["beginning"]) <= 100.0) or int(result["samples"]) <= 0 or float(result["runtime_seconds"]) < 0 or float(result["peak_memory_mb"]) < 0:
            raise ValueError("formal HUST common metric range failed")
        if result["result_kind"] == "beginning":
            if result["strict_online"] is not None or result["post_stream"] is not None or int(result["passes"]) != 0 or int(result["batches"]) != 0:
                raise ValueError("formal HUST beginning-only contract failed")
        if result["result_kind"] == "target":
            if any(result[key] is None for key in target_required):
                raise ValueError("formal HUST target result has unavailable metrics")
            if result["passes"] != 1 or int(result["batches"]) <= 0 or result["pre_update_scoring"] is not True or result["finite_losses"] is not True:
                raise ValueError("formal HUST target protocol invariant failed")
            matrix = result["confusion_matrix"]
            if len(matrix) != 7 or any(len(row) != 7 for row in matrix) or sum(map(sum, matrix)) != result["samples"] or result["class_coverage"] != 7:
                raise ValueError("formal HUST target confusion/coverage contract failed")
            exact_accuracy = 100.0 * sum(matrix[index][index] for index in range(7)) / max(result["samples"], 1)
            if abs(exact_accuracy - float(result["strict_online"])) > 1e-4:
                raise ValueError("formal HUST strict accuracy disagrees with confusion")
            precision, recall, f1 = [], [], []
            for class_id in range(7):
                tp = matrix[class_id][class_id]
                fp = sum(matrix[row][class_id] for row in range(7)) - tp
                fn = sum(matrix[class_id]) - tp
                p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
                precision.append(p); recall.append(r); f1.append(2 * p * r / max(p + r, 1e-12))
            exact_macro = [100 * sum(values) / 7 for values in (precision, recall, f1)]
            if any(abs(float(result[key]) - value) > 1e-4 for key, value in zip(("macro_precision", "macro_recall", "macro_f1"), exact_macro)):
                raise ValueError("formal HUST macro metrics disagree with confusion")
            if result["route"].startswith("0711"):
                offline_matrix = result["offline_confusion_matrix"]
                if (len(offline_matrix) != 7 or any(len(row) != 7 for row in offline_matrix)
                        or sum(map(sum, offline_matrix)) != result["samples"]):
                    raise ValueError("formal HUST teacher confusion contract failed")
                offline_accuracy = 100.0 * sum(
                    offline_matrix[index][index] for index in range(7)
                ) / max(result["samples"], 1)
                if abs(offline_accuracy - float(result["offline_purity"])) > 1e-4:
                    raise ValueError("formal HUST teacher purity disagrees with confusion")
            bounded_percent = ("strict_online", "post_stream", "macro_precision", "macro_recall", "macro_f1", "post_macro_precision", "post_macro_recall", "post_macro_f1", "offline_purity", "certain_purity")
            if any(not math.isfinite(float(result[key])) or not 0 <= float(result[key]) <= 100 for key in bounded_percent):
                raise ValueError("formal HUST percentage metric range failed")
            if any(not math.isfinite(float(result[key])) or not 0 <= float(result[key]) <= 1 for key in ("certain_ratio", "uncertain_ratio")) or abs(float(result["certain_ratio"]) + float(result["uncertain_ratio"]) - 1.0) > 1e-6:
                raise ValueError("formal HUST routing ratio contract failed")
            if not isinstance(result["memory_size"], int) or result["memory_size"] < 0 or not isinstance(result["memory_class_coverage"], int) or not 0 <= result["memory_class_coverage"] <= 7:
                raise ValueError("formal HUST memory contract failed")
            if result["route"].startswith("0711"):
                if result["evidence_applicable"] is not True or not isinstance(result["evidence_active_ratio"], (int, float)) or not 0 <= float(result["evidence_active_ratio"]) <= 1 or result["metadata_evidence_used"] is not True:
                    raise ValueError("formal HUST evidence contract failed")
            elif result["evidence_applicable"] is not False or result["evidence_active_ratio"] != "not_applicable" or result["metadata_evidence_used"] is not False:
                raise ValueError("formal HUST not-applicable evidence contract failed")
            if not result["trainable_parameters"] or result["trainable_parameters"] != result["trainable_allowlist"]:
                raise ValueError("formal HUST trainable allowlist identity failed")
            if result["route"].startswith("0711") and any(not any(token in name for token in ("band_scale", "band_bias", "warp_ctrl")) for name in result["trainable_allowlist"]):
                raise ValueError("formal 0711 trainable allowlist failed")
        if expected:
            mismatched = {key: (expected[key], result.get(key)) for key in expected if result.get(key) != expected[key]}
            if mismatched:
                raise ValueError(f"HUST result identity mismatch: {mismatched}")
        result["before"] = result["beginning"]
        result["mean_batch_ms"] = 1000.0 * float(result["runtime_seconds"]) / max(int(result["batches"]), 1)
        result["memory_class_coverage"] = result.get("memory_class_coverage")
        if not result["route"].startswith("0711"):
            result["offline_confusion_matrix"] = result.get("confusion_matrix")
        return result
    if expected is not None:
        raise ValueError("formal log missing HUST_RESULT_JSON")
    before = _last(r"Beginning Acc T\s*=\s*([0-9.]+)%", text)
    strict = _last(r"Strict Online Acc\s*=\s*([0-9.]+)%", text)
    post = _last(r"Post-stream Full-Target Acc\s*=\s*([0-9.]+)%", text)
    online = re.findall(r"Strict Online Macro P/R/F1\s*=\s*([0-9.]+)/([0-9.]+)/([0-9.]+)%", text)
    post_macro = re.findall(r"Post-stream Macro P/R/F1\s*=\s*([0-9.]+)/([0-9.]+)/([0-9.]+)%", text)
    batches = re.findall(r"\bbatches=(\d+)", text)
    mean_batch_ms = _last(r"mean_batch_ms=([0-9.]+)", text)
    required = {"before": before, "strict_online": strict, "post_stream": post, "batches": batches[-1] if batches else None, "mean_batch_ms": mean_batch_ms}
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"runner log missing required metrics: {', '.join(missing)}")
    macro_p, macro_r, macro_f1 = map(float, online[-1]) if online else (None, None, None)
    post_p, post_r, post_f1 = map(float, post_macro[-1]) if post_macro else (None, None, None)
    count = int(batches[-1])
    metrics: dict[str, object] = {
        "route": route, "before": before, "strict_online": strict,
        "macro_precision": macro_p, "macro_recall": macro_r, "macro_f1": macro_f1,
        "post_stream": post, "post_macro_precision": post_p, "post_macro_recall": post_r,
        "post_macro_f1": post_f1, "batches": count,
        "mean_batch_ms": mean_batch_ms, "runtime_seconds": count * float(mean_batch_ms) / 1000.0,
    }
    for key in ("peak_memory_mb", "memory_size", "memory_class_coverage", "certain_ratio", "uncertain_ratio", "offline_purity"):
        value = _last(rf"{key}=([0-9.]+)", text)
        metrics[key] = int(value) if value is not None and key in {"memory_size", "memory_class_coverage"} else value
    if metrics["memory_size"] is None:
        bank = _last(r"\bbank=(\d+)", text)
        metrics["memory_size"] = int(bank) if bank is not None else None
    if metrics["memory_class_coverage"] is None:
        bank_classes = _last(r"\bbank_cls=(\d+)/\d+", text)
        metrics["memory_class_coverage"] = int(bank_classes) if bank_classes is not None else None
    if metrics["offline_purity"] is None:
        metrics["offline_purity"] = _last(r"\bpseudo_purity=([0-9.]+)", text)
    confusion_match = re.search(
        r"confusion_matrix_7x7=\s*\n((?:\s*\d+(?:\s+\d+){6}\s*\n){7})",
        text,
    )
    metrics["offline_confusion_matrix"] = (
        [[int(value) for value in line.split()] for line in confusion_match.group(1).strip().splitlines()]
        if confusion_match else None
    )
    return metrics


def beginning_audit(scores: Sequence[float]) -> str:
    if len(scores) == 12 and all(25.0 <= value <= 70.0 for value in scores):
        return "preferred"
    in_range = sum(25.0 <= value <= 70.0 for value in scores)
    if len(scores) == 12 and in_range >= 9 and all(15.0 <= value <= 80.0 for value in scores):
        return "acceptable"
    return "supplementary_load_audit_required"


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row}) or ["status"]
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            encoded = {key: json.dumps(value, separators=(",", ":")) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()}
            writer.writerow(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _complete_route(rows: Sequence[Mapping[str, object]], route: str, seed: int = 2025) -> list[Mapping[str, object]]:
    selected = [row for row in rows if row.get("route") == route and int(row.get("stream_seed", seed)) == seed and row.get("status", "succeeded") == "succeeded" and row.get("strict_online") is not None and row.get("macro_f1") is not None]
    by_task = {tuple(row["task"]): row for row in selected if "task" in row}
    return [by_task[key] for key in sorted(by_task)] if set(by_task) == {(s, t) for s in range(4) for t in range(4) if s != t} and len(selected) == 12 else []


def _complete_candidate(rows: Sequence[Mapping[str, object]], candidate: str, seed: int = 2025) -> list[Mapping[str, object]]:
    selected = [row for row in rows if row.get("route") == "0711_robust" and str(row.get("candidate_id")) == candidate and int(row.get("stream_seed", seed)) == seed and row.get("status", "succeeded") == "succeeded" and row.get("strict_online") is not None and row.get("macro_f1") is not None]
    latest = {tuple(row["task"]): row for row in selected if "task" in row}
    return [latest[key] for key in sorted(latest)] if len(latest) == 12 else []


def _mean(rows: Sequence[Mapping[str, object]]) -> float | None:
    return sum(float(row["strict_online"]) for row in rows) / len(rows) if rows else None


def _exact_target_rows(
    rows: Sequence[Mapping[str, object]], *, route: str, candidate: str,
    seed: int, stage: str, expected_tasks: set[tuple[int, int]],
) -> list[Mapping[str, object]]:
    selected = [row for row in rows if row.get("route") == route and str(row.get("candidate_id")) == candidate and int(row.get("stream_seed", 2025)) == seed and str(row.get("stage", "baseline")) == stage and row.get("status", "succeeded") == "succeeded" and row.get("strict_online") is not None and row.get("macro_f1") is not None]
    by_task = {tuple(row["task"]): row for row in selected}
    return [by_task[task] for task in sorted(expected_tasks)] if len(selected) == len(expected_tasks) and set(by_task) == expected_tasks else []


def _frozen_primary_rows(rows: Sequence[Mapping[str, object]], candidate: str) -> list[Mapping[str, object]]:
    all_tasks = {(s, t) for s in range(4) for t in range(4) if s != t}
    baseline = _exact_target_rows(rows, route="0711_robust", candidate=candidate, seed=2025, stage="baseline", expected_tasks=all_tasks)
    if baseline:  # Synthetic/reproduction directories may materialize a complete frozen baseline table.
        return baseline
    dev = {(0, 1), (1, 2), (2, 3), (3, 0)}
    heldout = all_tasks - dev
    heldout_rows = _exact_target_rows(rows, route="0711_robust", candidate=candidate, seed=2025, stage="heldout", expected_tasks=heldout)
    stages = sorted({str(row.get("stage")) for row in rows if row.get("route") == "0711_robust" and str(row.get("candidate_id")) == candidate and str(row.get("stage", "")).startswith("tune_group_")})
    dev_rows = []
    for stage in reversed(stages):
        dev_rows = _exact_target_rows(rows, route="0711_robust", candidate=candidate, seed=2025, stage=stage, expected_tasks=dev)
        if dev_rows:
            break
    return [*dev_rows, *heldout_rows] if dev_rows and heldout_rows else []


def _authoritative_frozen_primary_rows(
    rows: Sequence[Mapping[str, object]],
    frozen: Mapping[str, object],
    proof: Mapping[str, object],
) -> list[Mapping[str, object]]:
    """Resolve the exact frozen seed-2025 table from the signed selection proof."""
    candidate = str(frozen.get("candidate_id", ""))
    if not candidate or str(proof.get("selected", {}).get("candidate_id", "")) != candidate:
        return []
    all_tasks = {(s, t) for s in range(4) for t in range(4) if s != t}
    dev_tasks = {(0, 1), (1, 2), (2, 3), (3, 0)}
    heldout_tasks = all_tasks - dev_tasks
    selected_entry = next(
        (row for row in proof.get("final_ranking", ()) if str(row.get("candidate_id", "")) == candidate),
        None,
    )
    if selected_entry is None and candidate != "untuned":
        return []
    if candidate == "untuned":
        complete_baseline = _exact_target_rows(
            rows, route="0711_robust", candidate="untuned", seed=2025,
            stage="baseline", expected_tasks=all_tasks,
        )
        dev_rows = [row for row in complete_baseline if tuple(row["task"]) in dev_tasks]
    else:
        stages = dict(selected_entry.get("stages", {}))
        stage = str(stages.get("2025", stages.get(2025, "")))
        dev_rows = _exact_target_rows(
            rows, route="0711_robust", candidate=candidate, seed=2025,
            stage=stage, expected_tasks=dev_tasks,
        ) if stage else []
    heldout_rows = _exact_target_rows(
        rows, route="0711_robust", candidate=candidate, seed=2025,
        stage="heldout", expected_tasks=heldout_tasks,
    )
    return [*dev_rows, *heldout_rows] if len(dev_rows) == 4 and len(heldout_rows) == 8 else []


def _authoritative_seed2026_candidate(
    rows: Sequence[Mapping[str, object]],
    candidate: str,
    *,
    proof: Mapping[str, object] | None = None,
    required: bool = False,
) -> tuple[list[Mapping[str, object]], list[dict[str, object]]]:
    """Compose the immutable four development rows with eight post-freeze rows.

    Older runs may contain four redundant post-freeze development reruns.  They
    remain independently validated evidence, but are audit-only: the pre-freeze
    ``tune_stability`` rows are the authoritative development observations.
    """
    dev = {(0, 1), (1, 2), (2, 3), (3, 0)}
    nondev = {(s, t) for s in range(4) for t in range(4) if s != t} - dev
    relevant = [
        row for row in rows
        if row.get("route") == "0711_robust"
        and str(row.get("candidate_id")) == candidate
        and int(row.get("stream_seed", 2025)) == 2026
        and row.get("stage") in {"tune_stability", "stability"}
    ]
    if not relevant and not required:
        return [], []
    tune = [row for row in relevant if row.get("stage") == "tune_stability"]
    final_nondev = [
        row for row in relevant
        if row.get("stage") == "stability" and tuple(row.get("task", ())) in nondev
    ]
    redundant = [
        row for row in relevant
        if row.get("stage") == "stability" and tuple(row.get("task", ())) in dev
    ]
    if (
        len(tune) != 4 or {tuple(row.get("task", ())) for row in tune} != dev
        or len(final_nondev) != 8
        or {tuple(row.get("task", ())) for row in final_nondev} != nondev
        or (redundant and (len(redundant) != 4 or {tuple(row.get("task", ())) for row in redundant} != dev))
    ):
        raise ValueError(
            f"authoritative tune_stability/stability inventory is incomplete for {candidate}"
        )

    tune_by_task = {tuple(row["task"]): row for row in tune}
    tune_by_source = {int(row["source"]): row for row in tune}
    reference = tune[0]
    def cache_identity(value: Mapping[str, object]) -> dict[str, object]:
        manifest_path = value.get("cache_manifest_path")
        manifest_hash = value.get("cache_manifest_sha256")
        content_hash = value.get("cache_content_sha256")
        tensors = value.get("cache_tensor_sha256s")
        if (
            not isinstance(manifest_path, str) or not manifest_path
            or not isinstance(manifest_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", manifest_hash)
            or not isinstance(content_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", content_hash)
            or not isinstance(tensors, Mapping) or not tensors
        ):
            raise ValueError(f"seed-2026 cache identity is incomplete for {candidate}")
        canonical_tensors = {
            str(Path(str(path)).resolve()): str(digest)
            for path, digest in tensors.items()
        }
        if (
            len(canonical_tensors) != len(tensors)
            or any(not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in canonical_tensors.values())
        ):
            raise ValueError(f"seed-2026 cache tensor identity is invalid for {candidate}")
        return {
            "cache_manifest_path": str(Path(manifest_path).resolve()),
            "cache_manifest_sha256": manifest_hash,
            "cache_content_sha256": content_hash,
            "cache_tensor_sha256s": dict(sorted(canonical_tensors.items())),
        }

    reference_cache = cache_identity(reference)
    common_keys = (
        "route", "variant", "candidate_id", "source_seed", "stream_seed",
        "config_sha256", "overrides",
    )
    row_keys = ("source", "target", "source_checkpoint_sha256")
    for row in [*tune, *final_nondev, *redundant]:
        task = tuple(row.get("task", ()))
        if (
            len(task) != 2
            or row.get("source") != task[0]
            or row.get("target") != task[1]
            or any(row.get(key) != reference.get(key) for key in common_keys)
            or not isinstance(row.get("config_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("config_sha256")))
            or not isinstance(row.get("source_checkpoint_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("source_checkpoint_sha256")))
        ):
            raise ValueError(f"seed-2026 stability identity/config mismatch for {candidate} task {list(task)}")
        if cache_identity(row) != reference_cache:
            raise ValueError(f"seed-2026 stability cache identity mismatch for {candidate} task {list(task)}")
        source_reference = tune_by_source.get(int(row["source"]))
        if source_reference is None or any(row.get(key) != source_reference.get(key) for key in ("source", "source_checkpoint_sha256")):
            raise ValueError(f"seed-2026 stability source hash mismatch for {candidate} task {list(task)}")

    redundant_audit = []
    for row in redundant:
        prior = tune_by_task[tuple(row["task"])]
        if any(row.get(key) != prior.get(key) for key in (*common_keys, *row_keys)):
            raise ValueError(f"seed-2026 redundant overlap identity mismatch for {candidate} task {row['task']}")
        redundant_audit.append({
            **dict(row),
            "authoritative_stage": "tune_stability",
            "authoritative_strict_online": prior.get("strict_online"),
            "authoritative_macro_f1": prior.get("macro_f1"),
            "strict_online_drift": float(row["strict_online"]) - float(prior["strict_online"]),
            "macro_f1_drift": float(row["macro_f1"]) - float(prior["macro_f1"]),
            "excluded_from_authoritative_metrics": True,
        })

    if proof is not None:
        if cache_identity(proof) != reference_cache:
            raise ValueError(f"authoritative proof cache identity mismatch for {candidate}")
        evidence = [
            row for row in proof.get("stability_evidence", ())
            if str(row.get("candidate_id")) == candidate
        ]
        if len(evidence) != 4 or {tuple(row.get("task", ())) for row in evidence} != dev:
            raise ValueError(f"authoritative proof stability identity is incomplete for {candidate}")
        evidence_by_task = {tuple(row["task"]): row for row in evidence}
        for row in tune:
            bound = evidence_by_task[tuple(row["task"])]
            state_path = row.get("_state_path")
            if (
                state_path is None
                or str(Path(str(state_path)).resolve()) != str(Path(str(bound.get("state_path"))).resolve())
                or row.get("config_sha256") != bound.get("config_sha256")
                or row.get("stage") != bound.get("stage")
                or int(row.get("stream_seed", -1)) != int(bound.get("stream_seed", -2))
            ):
                raise ValueError(f"authoritative proof stability identity mismatch for {candidate} task {row['task']}")

    authoritative = sorted([*tune, *final_nondev], key=lambda row: tuple(row["task"]))
    return authoritative, sorted(redundant_audit, key=lambda row: tuple(row["task"]))


def _primary_beginning_variant(row: Mapping[str, object]) -> str | None:
    """Return the exact primary Beginning family, excluding failed or malformed states."""
    family = (str(row.get("route")), str(row.get("variant")), str(row.get("candidate_id")))
    families = {
        ("dtcc_ordinary", "ordinary", "beginning-ordinary"): "ordinary",
        ("0711_robust", "robust", "beginning-robust"): "robust",
    }
    task = row.get("task")
    if not isinstance(task, (list, tuple)) or len(task) != 2:
        return None
    source, target = task
    numeric = lambda value: isinstance(value, (int, float)) and not isinstance(value, bool)
    valid = (
        row.get("kind") == "target" and row.get("status") == "succeeded"
        and row.get("stage") == "beginning" and row.get("load_split") is False
        and row.get("result_kind") == "beginning" and row.get("schema_version") == 1
        and row.get("source_seed") == 2025 and row.get("stream_seed") == 2025
        and isinstance(source, int) and not isinstance(source, bool)
        and isinstance(target, int) and not isinstance(target, bool)
        and 0 <= source < 4 and 0 <= target < 4 and source != target
        and row.get("source") == source and row.get("target") == target
        and numeric(row.get("before")) and numeric(row.get("beginning"))
        and float(row["before"]) == float(row["beginning"])
        and isinstance(row.get("samples"), int) and not isinstance(row.get("samples"), bool)
        and int(row["samples"]) > 0 and row.get("batches") == 0 and row.get("passes") == 0
        and row.get("strict_online") is None and row.get("post_stream") is None
        and row.get("macro_precision") is None and row.get("macro_recall") is None
        and row.get("macro_f1") is None and row.get("confusion_matrix") is None
        and row.get("class_coverage") is None and row.get("finite_losses") is True
    )
    return families.get(family) if valid else None


def _preflight_final_inventory(
    run_dir: Path,
    records: Sequence[Mapping[str, object]],
    frozen: Mapping[str, object] | None,
    checkpoint_manifest: Mapping[str, object],
) -> tuple[Mapping[str, object], list[Mapping[str, object]]]:
    """Reject incomplete formal inventories before any publication file is written."""
    all_tasks = {(s, t) for s in range(4) for t in range(4) if s != t}
    source_rows = [
        row
        for row in records
        if row.get("kind") == "source"
        and row.get("stage") == "source"
        and row.get("load_split") is False
        and row.get("status") == "succeeded"
    ]
    source_ids = {
        (str(row.get("variant")), int(row.get("source", -1)))
        for row in source_rows
    }
    expected_sources = {
        (variant, source)
        for variant in ("ordinary", "robust")
        for source in range(4)
    }
    beginning_ok = all(
        len([
            row for row in records if _primary_beginning_variant(row) == variant
        ]) == 12
        and len({
            tuple(row["task"])
            for row in records
            if _primary_beginning_variant(row) == variant
        }) == 12
        for variant in ("ordinary", "robust")
    )
    baselines_ok = all(
        bool(_exact_target_rows(
            records,
            route=route,
            candidate=candidate,
            seed=2025,
            stage="baseline",
            expected_tasks=all_tasks,
        ))
        for route, candidate in (
            ("dtcc_ordinary", "dtcc_ordinary"),
            ("0711_robust", "untuned"),
            ("0711_common", "0711_common"),
        )
    )
    frozen_id = str(frozen.get("candidate_id", "")) if frozen else ""
    proof = None
    authoritative_frozen = None
    recomputed_manifest = None
    if frozen:
        try:
            from tools.tune_hust_0711_strict import (
                _build_primary_checkpoint_manifest,
                load_frozen_candidate,
                verify_search_proof,
            )
        except ModuleNotFoundError:
            from tune_hust_0711_strict import (
                _build_primary_checkpoint_manifest,
                load_frozen_candidate,
                verify_search_proof,
            )
        try:
            best_path = Path(run_dir) / "best_config.yaml"
            authoritative_frozen = load_frozen_candidate(best_path)
            if authoritative_frozen != dict(frozen):
                raise ValueError("authoritative frozen configuration mismatch")
            proof = verify_search_proof(Path(str(frozen["proof_path"])))
        except (KeyError, OSError, TypeError, ValueError):
            proof = None
        if proof is not None:
            try:
                recomputed_manifest = _build_primary_checkpoint_manifest(records)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("checkpoint manifest cannot be recomputed from exact source states") from exc
            proof_manifest = proof.get("checkpoint_manifest")
            if (
                not isinstance(proof_manifest, Mapping)
                or dict(checkpoint_manifest) != dict(proof_manifest)
                or dict(recomputed_manifest) != dict(proof_manifest)
            ):
                raise ValueError("checkpoint manifest differs from authoritative proof or source states")
    frozen_2025 = (
        _authoritative_frozen_primary_rows(records, frozen, proof)
        if frozen is not None and proof is not None
        else []
    )
    seed2026_candidates = list(dict.fromkeys(("untuned", frozen_id))) if frozen_id else []
    seed2026_ok = bool(seed2026_candidates) and all(
        len(_authoritative_seed2026_candidate(
            records, candidate, proof=proof, required=True
        )[0]) == 12
        for candidate in seed2026_candidates
    )
    required_rows = [
        row
        for row in records
        if row.get("stage") in {"source", "beginning", "baseline", "heldout", "tune_stability", "stability"}
        or str(row.get("stage", "")).startswith("tune_group_")
    ]
    formal_evidence_ok = all(row.get("_formal_state") is True for row in required_rows)
    if not (
        len(source_rows) == 8
        and source_ids == expected_sources
        and len(checkpoint_manifest) == 8
        and recomputed_manifest is not None
        and beginning_ok
        and baselines_ok
        and len(frozen_2025) == 12
        and seed2026_ok
        and formal_evidence_ok
        and proof is not None
        and authoritative_frozen is not None
        and not any(row.get("status") == "failed" for row in records)
    ):
        raise ValueError("final inventory is incomplete or lacks formal evidence")
    return proof, frozen_2025


def write_reports(run_dir: Path, records: Sequence[Mapping[str, object]], *, frozen: Mapping[str, object] | None, checkpoint_manifest: Mapping[str, object], supplementary_rows: Sequence[Mapping[str, object]] = (), strict_final: bool = False) -> dict[str, object]:
    run_dir = Path(run_dir)
    authoritative_frozen_rows: list[Mapping[str, object]] | None = None
    if strict_final:
        _proof, authoritative_frozen_rows = _preflight_final_inventory(
            run_dir, records, frozen, checkpoint_manifest
        )
    _validate_saved_evidence(records, checkpoint_manifest)
    run_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(records, key=lambda row: (str(row.get("route", "")), int(row.get("stream_seed", 0)), tuple(row.get("task", (-1, -1))), str(row.get("candidate_id", ""))))
    frozen_candidate_id = str(frozen.get("candidate_id", "")) if frozen else ""
    stability_candidates = list(dict.fromkeys(("untuned", frozen_candidate_id))) if frozen_candidate_id else []
    authoritative_stability: dict[str, list[Mapping[str, object]]] = {}
    redundant_stability: list[dict[str, object]] = []
    for candidate in stability_candidates:
        has_tune_evidence = any(
            row.get("route") == "0711_robust"
            and str(row.get("candidate_id")) == candidate
            and row.get("stage") == "tune_stability"
            and int(row.get("stream_seed", 2025)) == 2026
            for row in ordered
        )
        if strict_final or has_tune_evidence:
            authoritative, redundant = _authoritative_seed2026_candidate(
                ordered, candidate, required=True
            )
            authoritative_stability[candidate] = authoritative
            redundant_stability.extend(redundant)
    redundant_keys = {
        (str(row["candidate_id"]), tuple(row["task"])) for row in redundant_stability
    }
    identities = []
    for row in ordered:
        if "task" not in row:
            continue
        identities.append((str(row.get("route", "")), str(row.get("variant", "")), str(row.get("candidate_id", "")), int(row.get("stream_seed", 2025)), int(row.get("source", row["task"][0])), int(row.get("target", row["task"][1])), str(row.get("stage", "baseline"))))
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate route/variant/candidate/seed/task/stage identity")
    stage_sets: dict[tuple[str, str, str, int], set[str]] = {}
    for route, variant, candidate, seed, _source, _target, stage in identities:
        stage_sets.setdefault((route, variant, candidate, seed), set()).add(stage)
    for (route, _variant, aggregate_candidate, aggregate_seed), stages in stage_sets.items():
        allowed_frozen_mix = route == "0711_robust" and any(stage.startswith("tune_group_") for stage in stages) and stages <= {*[stage for stage in stages if stage.startswith("tune_group_")], "heldout"}
        allowed_stability_overlap = route == "0711_robust" and stages == {"tune_stability", "stability"}
        allowed_untuned_fallback = (
            strict_final and frozen_candidate_id == "untuned" and route == "0711_robust"
            and aggregate_candidate == "untuned" and aggregate_seed == 2025
            and stages == {"baseline", "heldout"}
        )
        if len(stages) > 1 and not (allowed_frozen_mix or allowed_stability_overlap or allowed_untuned_fallback):
            raise ValueError(f"mixed stages in one aggregate identity: {sorted(stages)}")
    _write_csv(run_dir / "metrics.csv", ordered)
    target_table_rows = [
        row for row in ordered
        if row.get("status") == "succeeded"
        and row.get("result_kind") == "target"
        and row.get("strict_online") is not None
        and row.get("macro_f1") is not None
        and not (
            row.get("stage") == "stability"
            and int(row.get("stream_seed", 2025)) == 2026
            and (str(row.get("candidate_id")), tuple(row.get("task", ()))) in redundant_keys
        )
    ]
    _write_csv(run_dir / "per_task_metrics.csv", target_table_rows)
    _write_csv(run_dir / "stability_seed2026_redundant_audit.csv", redundant_stability)
    attempt_inventory = []
    for command_path in sorted((run_dir / "commands").glob("*.attempt-*.txt")):
        stem = command_path.name[:-4]
        log_path = run_dir / "logs" / f"{stem}.log"
        match = re.search(r"\.attempt-(\d+)$", stem)
        attempt_inventory.append({
            "attempt": int(match.group(1)) if match else None, "command_path": str(command_path.resolve()),
            "command_sha256": _sha256_file(command_path), "log_path": str(log_path.resolve()),
            "log_exists": log_path.is_file(), "log_sha256": _sha256_file(log_path) if log_path.is_file() else None,
        })
    if not attempt_inventory:
        attempt_inventory = [row for row in ordered if "attempt" in row]
    _write_csv(run_dir / "attempts.csv", attempt_inventory)

    candidate_rows: dict[tuple[str, str, str, int, str], list[Mapping[str, object]]] = {}
    for row in ordered:
        if row.get("status", "succeeded") == "succeeded" and row.get("strict_online") is not None and row.get("macro_f1") is not None and row.get("result_kind", "target") == "target":
            if (
                str(row.get("candidate_id")) in authoritative_stability
                and int(row.get("stream_seed", 2025)) == 2026
                and row.get("stage") in {"tune_stability", "stability"}
            ):
                continue
            key = (str(row.get("route", "")), str(row.get("variant", "")), str(row.get("candidate_id", "")), int(row.get("stream_seed", 2025)), str(row.get("stage", "baseline")))
            candidate_rows.setdefault(key, []).append(row)
    leaderboard = []
    incomplete = []
    all_tasks = {(s, t) for s in range(4) for t in range(4) if s != t}
    dev_tasks = {(0, 1), (1, 2), (2, 3), (3, 0)}
    for (route, variant, cid, seed, stage), rows in candidate_rows.items():
        tasks = [tuple(row["task"]) for row in rows]
        expected_tasks = dev_tasks if stage.startswith("tune_group_") else all_tasks if stage in {"baseline", "stability"} else set(tasks)
        if len(tasks) != len(set(tasks)) or set(tasks) != expected_tasks:
            incomplete.append({"route": route, "variant": variant, "candidate_id": cid, "stream_seed": seed, "stage": stage, "rows": len(rows), "expected_rows": len(expected_tasks)})
            continue
        leaderboard.append({"route": route, "variant": variant, "candidate_id": cid, "stream_seed": seed, "stage": stage, "mean_strict_online": sum(float(row["strict_online"]) for row in rows) / len(rows), "mean_macro_f1": sum(float(row["macro_f1"]) for row in rows) / len(rows), "complete_rows": len(rows)})
    for cid, rows in authoritative_stability.items():
        leaderboard.append({
            "route": "0711_robust", "variant": "robust", "candidate_id": cid,
            "stream_seed": 2026, "stage": "stability_authoritative",
            "mean_strict_online": sum(float(row["strict_online"]) for row in rows) / len(rows),
            "mean_macro_f1": sum(float(row["macro_f1"]) for row in rows) / len(rows),
            "complete_rows": len(rows),
        })
    leaderboard.sort(key=lambda row: (row["route"], row["variant"], row["stream_seed"], row["stage"], -row["mean_strict_online"], row["candidate_id"]))
    _write_csv(run_dir / "leaderboard.csv", leaderboard)
    _write_csv(run_dir / "incomplete_candidates.csv", incomplete)

    beginning_rows = [row for row in ordered if _primary_beginning_variant(row) is not None]
    _write_csv(run_dir / "beginning_audit.csv", beginning_rows)
    family_status = {}
    family_violations = {}
    for variant in ("ordinary", "robust"):
        unique = {tuple(row["task"]): float(row["before"]) for row in beginning_rows if _primary_beginning_variant(row) == variant}
        family_status[variant] = beginning_audit(list(unique.values())) if len(unique) == 12 else "supplementary_load_audit_required"
        family_violations[variant] = {f"{task[0]}to{task[1]}": value for task, value in unique.items() if not 25.0 <= value <= 70.0}
    status = family_status["robust"]
    load_rows = list(supplementary_rows) or [row for row in ordered if row.get("stage") == "load-audit-beginning" and row.get("before") is not None]
    load_sources = [row for row in ordered if row.get("stage") == "load-audit-source"]
    expected_load = {(s, t) for s in range(3) for t in range(3) if s != t}
    source_inventory = []
    source_identities = []
    source_valid = True
    for row in load_sources:
        variant, source_id = str(row.get("variant")), int(row.get("source", -1))
        output_hashes = dict(row.get("output_artifact_hashes", {}))
        checkpoint_items = [(path, digest) for path, digest in output_hashes.items() if Path(path).suffix == ".pt"]
        checkpoint_path = checkpoint_items[0][0] if len(checkpoint_items) == 1 else row.get("source_checkpoint_path")
        checkpoint_hash = checkpoint_items[0][1] if len(checkpoint_items) == 1 else row.get("source_checkpoint_sha256")
        valid_hash = isinstance(checkpoint_hash, str) and bool(re.fullmatch(r"[0-9a-f]{64}", checkpoint_hash))
        valid = (row.get("kind") == "source" and row.get("load_split") is True and row.get("status") == "succeeded"
                 and variant in {"ordinary", "robust"} and source_id in {0, 1, 2}
                 and row.get("route") == f"source_{variant}" and int(row.get("source_seed", -1)) == 2025
                 and checkpoint_path is not None and valid_hash)
        source_valid = source_valid and valid
        source_identities.append((variant, source_id))
        source_inventory.append({"variant": variant, "source": source_id, "route": row.get("route"), "source_seed": row.get("source_seed"), "checkpoint_path": checkpoint_path, "checkpoint_sha256": checkpoint_hash, "valid": valid})
    expected_sources = {(variant, source) for variant in ("ordinary", "robust") for source in range(3)}
    source_complete = source_valid and len(source_identities) == 6 and len(set(source_identities)) == 6 and set(source_identities) == expected_sources
    _write_csv(run_dir / "supplementary_load_source_inventory.csv", source_inventory)
    _write_csv(run_dir / "supplementary_load_beginning_audit.csv", load_rows)
    load_status: dict[str, object] = {}
    beginning_complete = True
    for variant in ("ordinary", "robust"):
        expected_route = "0711_robust" if variant == "robust" else "dtcc_ordinary"
        family = [row for row in load_rows if row.get("variant") == variant]
        _write_csv(run_dir / f"supplementary_load_{variant}_six_task.csv", family)
        tasks = [tuple(row["task"]) for row in family if "task" in row]
        valid = (len(tasks) == 6 and len(set(tasks)) == 6 and set(tasks) == expected_load and all(
            row.get("kind", "target") == "target" and row.get("stage") == "load-audit-beginning"
            and row.get("load_split") is True and row.get("route") == expected_route
            and row.get("result_kind", "beginning") == "beginning" and row.get("status", "succeeded") == "succeeded"
            and int(row.get("source_seed", -1)) == 2025 and int(row.get("stream_seed", -1)) == 2025
            for row in family
        ))
        load_status[variant] = "complete" if valid else "incomplete"
        beginning_complete = beginning_complete and valid
    supplementary_complete = source_complete and beginning_complete and len(load_sources) + len(load_rows) == 18
    load_status.update({"source_inventory": "complete" if source_complete else "incomplete", "complete_records": 18 if supplementary_complete else len(load_sources) + len(load_rows), "expected_records": 18, "complete": supplementary_complete})
    _atomic_text(run_dir / "supplementary_load_status.json", json.dumps(load_status, sort_keys=True, indent=2) + "\n")

    all_tasks = {(s, t) for s in range(4) for t in range(4) if s != t}
    dtcc = _exact_target_rows(ordered, route="dtcc_ordinary", candidate="dtcc_ordinary", seed=2025, stage="baseline", expected_tasks=all_tasks)
    if not dtcc:  # Compatibility with pre-contract synthetic rows.
        dtcc = _complete_route(ordered, "dtcc_ordinary")
    frozen_id = str(frozen.get("candidate_id")) if frozen else ""
    tuned = (
        authoritative_frozen_rows
        if authoritative_frozen_rows is not None
        else (_frozen_primary_rows(ordered, frozen_id) if frozen_id else [])
    )
    baseline = _exact_target_rows(ordered, route="0711_robust", candidate="untuned", seed=2025, stage="baseline", expected_tasks=all_tasks)
    if not baseline:
        baseline = _complete_route(ordered, "0711_robust_baseline")
    if not baseline:
        baseline_candidates = sorted({str(row.get("candidate_id")) for row in ordered if row.get("route") == "0711_robust" and str(row.get("candidate_id")) != frozen_id and (row.get("stage") == "baseline" or row.get("candidate_id") in {"baseline", "base", "untuned"})})
        if baseline_candidates:
            baseline = _complete_candidate(ordered, baseline_candidates[0])
    common = _exact_target_rows(ordered, route="0711_common", candidate="0711_common", seed=2025, stage="baseline", expected_tasks=all_tasks)
    if not common:
        common = _complete_route(ordered, "0711_common")
    full_rows = [*dtcc, *baseline, *tuned]
    _write_csv(run_dir / "full_pipeline_12task.csv", full_rows)
    _write_csv(run_dir / "common_source_12task.csv", [*dtcc, *common])
    if authoritative_stability:
        stability = [
            row
            for candidate in stability_candidates
            for row in authoritative_stability.get(candidate, ())
        ]
    else:
        stability = _exact_target_rows(
            ordered, route="0711_robust", candidate="untuned", seed=2026,
            stage="stability", expected_tasks=all_tasks,
        )
        if frozen_id and frozen_id != "untuned":
            stability = [
                *stability,
                *_exact_target_rows(
                    ordered, route="0711_robust", candidate=frozen_id, seed=2026,
                    stage="stability", expected_tasks=all_tasks,
                ),
            ]
    _write_csv(run_dir / "stability_seed2026.csv", stability)

    manifest_path = run_dir / "checkpoint_manifest.json"
    _atomic_text(manifest_path, json.dumps(checkpoint_manifest, sort_keys=True, indent=2) + "\n")
    if frozen is not None and strict_final:
        try:
            from tools.tune_hust_0711_strict import load_frozen_candidate
        except ModuleNotFoundError:
            from tune_hust_0711_strict import load_frozen_candidate
        best_path = run_dir / "best_config.yaml"
        if not best_path.is_file() or load_frozen_candidate(best_path) != dict(frozen):
            raise ValueError("authoritative frozen configuration mismatch")

    failed = [row for row in ordered if row.get("status") == "failed"]
    _write_csv(run_dir / "failed_jobs.csv", failed)
    rejected = [row for row in leaderboard if row["route"] == "0711_robust" and frozen and row["candidate_id"] not in {frozen["candidate_id"], "untuned"} and (str(row.get("stage", "")).startswith("tune_group_") or row.get("stage") == "tune_stability")]
    _write_csv(run_dir / "rejected_candidates.csv", rejected)
    environment = {
        "python": __import__("sys").version, "platform": __import__("platform").platform(),
        "source_seed": 2025, "primary_stream_seed": 2025, "stability_stream_seed": 2026,
        "gpu_ids": sorted({int(row["gpu"]) for row in ordered if row.get("gpu") is not None}),
        "commands": [row["command"] for row in ordered if row.get("command")],
        "runtime_seconds": _range(ordered, "runtime_seconds"),
        "peak_memory_mb": _range(ordered, "peak_memory_mb"),
    }
    _atomic_text(run_dir / "environment.json", json.dumps(environment, sort_keys=True, indent=2) + "\n")

    tuned_mean, baseline_mean, dtcc_mean = _mean(tuned), _mean(baseline), _mean(dtcc)
    primary_gain = tuned_mean - baseline_mean if tuned_mean is not None and baseline_mean is not None else None
    primary_worst_delta = min(float(row["strict_online"]) - float(next(base["strict_online"] for base in baseline if tuple(base["task"]) == tuple(row["task"]))) for row in tuned) if tuned and baseline else None
    development_tasks = {(0, 1), (1, 2), (2, 3), (3, 0)}
    tuned_dev = [row for row in tuned if tuple(row["task"]) in development_tasks]
    baseline_dev = [row for row in baseline if tuple(row["task"]) in development_tasks]
    dev_gain = (
        _mean(tuned_dev) - _mean(baseline_dev)
        if len(tuned_dev) == len(baseline_dev) == 4
        else None
    )
    dev_worst_delta = (
        min(
            float(row["strict_online"])
            - float(next(base["strict_online"] for base in baseline_dev if tuple(base["task"]) == tuple(row["task"])))
            for row in tuned_dev
        )
        if len(tuned_dev) == len(baseline_dev) == 4
        else None
    )
    untuned_fallback = frozen_id == "untuned"
    recommended = (
        not untuned_fallback
        and frozen_id != ""
        and len(tuned) == 12
        and len(baseline) == 12
        and tuned_mean is not None
        and baseline_mean is not None
        and tuned_mean > baseline_mean
    )
    if untuned_fallback:
        recommendation_reason = "untuned_fallback"
    elif len(tuned) != 12 or len(baseline) != 12:
        recommendation_reason = "incomplete_full12_inventory"
    elif recommended:
        recommendation_reason = "strict_full12_mean_improvement"
    else:
        recommendation_reason = "no_strict_full12_mean_improvement"
    seed2025_guard_passed = (
        dev_gain is not None
        and dev_worst_delta is not None
        and dev_gain >= 0.30
        and dev_worst_delta >= -1.0
    )
    beats_dtcc = tuned_mean is not None and dtcc_mean is not None and tuned_mean > dtcc_mean
    frozen_2026 = (
        authoritative_stability.get(frozen_id, [])
        if authoritative_stability
        else _exact_target_rows(ordered, route="0711_robust", candidate=frozen_id, seed=2026, stage="stability", expected_tasks=all_tasks)
    ) if frozen_id else []
    untuned_2026 = (
        authoritative_stability.get("untuned", [])
        if authoritative_stability
        else _exact_target_rows(ordered, route="0711_robust", candidate="untuned", seed=2026, stage="stability", expected_tasks=all_tasks)
    )
    stability_gain = (_mean(frozen_2026) - _mean(untuned_2026)) if frozen_2026 and untuned_2026 else None
    stability_regression = min(float(row["strict_online"]) - float(next(base["strict_online"] for base in untuned_2026 if tuple(base["task"]) == tuple(row["task"]))) for row in frozen_2026) if frozen_2026 and untuned_2026 else None
    target_ranges = {
        "beginning_ordinary": {"beginning": _range([row for row in beginning_rows if row.get("variant") == "ordinary"], "before")},
        "beginning_robust": {"beginning": _range([row for row in beginning_rows if row.get("variant") == "robust"], "before")},
        "dtcc": {"strict_online": _range(dtcc, "strict_online"), "macro_f1": _range(dtcc, "macro_f1")},
        "untuned_robust": {"strict_online": _range(baseline, "strict_online"), "macro_f1": _range(baseline, "macro_f1")},
        "frozen": {"strict_online": _range(tuned, "strict_online"), "macro_f1": _range(tuned, "macro_f1")},
        "common": {"strict_online": _range(common, "strict_online"), "macro_f1": _range(common, "macro_f1")},
    }
    protocol_audit = {
        "primary_split": "bearing", "supplementary_split": "load", "supplementary_never_primary": True,
        "source_seeds": sorted({int(row.get("source_seed", 2025)) for row in ordered}),
        "stream_seeds": sorted({int(row["stream_seed"]) for row in ordered if row.get("stream_seed") is not None}), "passes": 1,
        "cache_manifest_hashes": {path: digest for row in ordered for path, digest in dict(row.get("artifact_hashes", {})).items() if Path(path).name == "manifest.json"},
        "checkpoint_manifest": checkpoint_manifest, "frozen_candidate": frozen,
    }
    _atomic_text(run_dir / "protocol_audit.json", json.dumps(protocol_audit, sort_keys=True, indent=2) + "\n")
    redundant_audit_summary = {
        "rows": len(redundant_stability),
        "excluded_from_authoritative_metrics": True,
        "max_abs_strict_online_drift": max(
            (abs(float(row["strict_online_drift"])) for row in redundant_stability),
            default=0.0,
        ),
        "max_abs_macro_f1_drift": max(
            (abs(float(row["macro_f1_drift"])) for row in redundant_stability),
            default=0.0,
        ),
    }
    machine_summary = {
        "recommended": recommended, "beats_dtcc": beats_dtcc,
        "recommendation_reason": recommendation_reason,
        "tuned_vs_untuned_gain": primary_gain, "tuned_vs_untuned_worst_task_delta": primary_worst_delta,
        "seed2025_development_gain": dev_gain,
        "seed2025_development_worst_task_delta": dev_worst_delta,
        "seed2025_development_guard_passed": seed2025_guard_passed,
        "stability_seed2026_gain": stability_gain, "stability_seed2026_worst_task_delta": stability_regression,
        "stability_seed2026_guard_passed": stability_gain is not None and stability_regression is not None and stability_gain >= 0.30 and stability_regression >= -1.0,
        "stability_redundant_audit": redundant_audit_summary,
        "ranges": target_ranges, "beginning_family_status": family_status,
        "beginning_violations": family_violations, "frozen": frozen,
        "failed_jobs": len(failed), "rejected_candidates": len(rejected), "incomplete_candidates": len(incomplete),
    }
    _atomic_text(run_dir / "report_summary.json", json.dumps(machine_summary, sort_keys=True, indent=2) + "\n")
    report = [
        (
            "# HUST DtCC / 0711 strict-online report"
            if strict_final
            else "# DRAFT HUST DtCC / 0711 strict-online report"
        ), "",
        "## Beginning soft audit", "",
        f"Ordinary status: `{family_status['ordinary']}`; robust status: `{family_status['robust']}`. Primary bearing split retained; supplementary load results are audit-only and never replace it.",
        f"Preferred-range violations: `{json.dumps(family_violations, sort_keys=True)}`.", "",
        "## Full-pipeline conclusion", "",
        f"Frozen 0711 recommended over untuned 0711: **{str(recommended).lower()}**. Tuned mean={tuned_mean}; untuned mean={baseline_mean}.",
        (
            "Recommendation basis: untuned fallback selected; no tuned candidate is recommended regardless of numeric rerun jitter."
            if untuned_fallback
            else f"Recommendation basis: `{recommendation_reason}`."
        ),
        f"Separate DtCC comparison: frozen 0711 beats DtCC: **{str(beats_dtcc).lower()}**. DtCC mean={dtcc_mean}.", "",
        "## Common-source conclusion", "",
        f"0711 common-source mean={_mean(common)}; ordinary-source DtCC mean={dtcc_mean}. This ablation is not merged with the full-pipeline conclusion.", "",
        "## Tuned-vs-untuned gain and matched stability guard", "",
        f"Seed-2025 full-12 gain: `{machine_summary['tuned_vs_untuned_gain']}`; development gain: `{dev_gain}`; development worst task delta: `{dev_worst_delta}`; development guard passed: `{seed2025_guard_passed}`. Seed-2026 matched gain: `{stability_gain}`; worst task delta: `{stability_regression}`; guard passed: `{machine_summary['stability_seed2026_guard_passed']}`.", "",
        "## Seed-2026 redundant development rerun audit", "",
        f"Authoritative seed-2026 tables reuse the four pre-freeze `tune_stability` development rows and the eight post-freeze non-development rows per candidate. Redundant post-freeze development reruns={len(redundant_stability)}; they are excluded from every authoritative mean and recommendation. Drift summary: `{json.dumps(redundant_audit_summary, sort_keys=True)}`. Exact rows are in `stability_seed2026_redundant_audit.csv`.", "",
        "## Frozen configuration", "",
        f"Candidate: `{frozen_id or 'not frozen'}`; overrides: `{json.dumps(frozen.get('overrides', {}) if frozen else {}, sort_keys=True)}`.", "",
        "## Exact per-task Beginning / strict accuracy / macro-F1", "",
        *_markdown_table([*beginning_rows, *target_table_rows]), "",
        "## Means and ranges", "", f"`{json.dumps(target_ranges, sort_keys=True)}`", "",
        "## Failed, incomplete, rejected, and retry attempts", "",
        f"Failed={len(failed)}; incomplete={len(incomplete)}; rejected={len(rejected)}; persisted attempts={len(attempt_inventory)}.", "",
        "## Commands, environment, GPU, runtime, and memory", "",
        "Exact commands and immutable attempt logs are recorded in `attempts.csv`; environment and seeds are recorded in `environment.json`.",
        f"Runtime range: `{json.dumps(_range(target_table_rows, 'runtime_seconds'), sort_keys=True)}`; peak GPU memory range: `{json.dumps(_range(target_table_rows, 'peak_memory_mb'), sort_keys=True)}`.", "",
        "## Cache, source checkpoints, seeds, and protocol audit", "",
        "Primary bearing and supplementary load identities, cache/source hashes, source seed 2025, stream seeds 2025/2026, and the frozen record are in `protocol_audit.json`.", "",
        "## Supplementary load audit", "",
        f"Load Beginning rows={len(load_rows)}; audit-only and never mixed into the primary tables.", "",
        *_markdown_table(load_rows), "",
        f"Frozen configuration hash: `{frozen.get('freeze_sha256') if frozen else 'not frozen'}`.",
    ]
    _atomic_text(run_dir / "report.md", "\n".join(report) + "\n")
    return {"recommended": recommended, "beats_dtcc": beats_dtcc, "beginning_status": status, "beginning_family_status": family_status, "supplementary_complete": supplementary_complete}


def load_state_records(run_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted((Path(run_dir) / "state").glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        value["_formal_state"] = True
        value["_state_path"] = str(path.resolve())
        records.append(flatten_state_metrics(value))
    return records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--write-final", action="store_true", help="write the final report artifacts")
    args = parser.parse_args(argv)
    frozen_path = args.run_dir / "best_config.yaml"
    frozen = yaml.safe_load(frozen_path.read_text()) if frozen_path.exists() else None
    manifest_path = args.run_dir / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    write_reports(
        args.run_dir,
        load_state_records(args.run_dir),
        frozen=frozen,
        checkpoint_manifest=manifest,
        strict_final=args.write_final,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
