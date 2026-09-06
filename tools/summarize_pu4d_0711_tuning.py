#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any

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


@dataclass(frozen=True)
class RunnerMetrics:
    before: float
    strict_online: float
    post_stream: float
    batches: int
    runtime_seconds: float


@dataclass(frozen=True)
class CandidateAggregate:
    candidate_id: str
    overrides: dict[str, object]
    task_metrics: dict[tuple[int, int], RunnerMetrics]
    mean_strict_online: float
    minimum_task_delta: float
    wall_time_seconds: float


@dataclass(frozen=True)
class RecoveryResult:
    passed: bool
    exact_mean: float | None
    mean_delta: float | None
    task_failures: dict[tuple[int, int], float | None]


def _last_float(pattern: str, text: str) -> float | None:
    matches = re.findall(pattern, text)
    return float(matches[-1]) if matches else None


def parse_runner_log(path: Path) -> RunnerMetrics:
    text = path.read_text(encoding="utf-8", errors="ignore")
    before = _last_float(r"Beginning Acc T\s*=\s*([0-9.]+)%", text)
    strict_online = _last_float(r"Strict Online Acc\s*=\s*([0-9.]+)%", text)
    post_stream = _last_float(
        r"Post-stream Full-Target Acc\s*=\s*([0-9.]+)%", text
    )
    batch_matches = re.findall(r"\[STRICT 0711\]\s+batches=(\d+)", text)
    mean_batch_ms = _last_float(r"mean_batch_ms=([0-9.]+)", text)
    missing = [
        name
        for name, value in (
            ("before", before),
            ("strict_online", strict_online),
            ("post_stream", post_stream),
            ("batches", batch_matches[-1] if batch_matches else None),
            ("mean_batch_ms", mean_batch_ms),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"runner log missing required metrics: {', '.join(missing)}")

    batches = int(batch_matches[-1])
    return RunnerMetrics(
        before=float(before),
        strict_online=float(strict_online),
        post_stream=float(post_stream),
        batches=batches,
        runtime_seconds=batches * float(mean_batch_ms) / 1000.0,
    )


def aggregate_candidate(
    records: Sequence[Mapping[str, Any]],
    required_tasks: Sequence[tuple[int, int]],
) -> CandidateAggregate | None:
    if not records:
        return None
    candidate_ids = {str(record["candidate_id"]) for record in records}
    if len(candidate_ids) != 1:
        raise ValueError("candidate records must share one candidate_id")
    task_metrics = {
        tuple(record["task"]): record["metrics"]
        for record in records
        if isinstance(record.get("metrics"), RunnerMetrics)
    }
    required = tuple(required_tasks)
    if any(task not in task_metrics for task in required):
        return None

    selected = {task: task_metrics[task] for task in required}
    scores = [metrics.strict_online for metrics in selected.values()]
    deltas = [
        selected[task].strict_online - HISTORICAL_BASELINE[task] for task in required
    ]
    return CandidateAggregate(
        candidate_id=next(iter(candidate_ids)),
        overrides=dict(records[0].get("overrides", {})),
        task_metrics=selected,
        mean_strict_online=sum(scores) / len(scores),
        minimum_task_delta=min(deltas),
        wall_time_seconds=sum(metrics.runtime_seconds for metrics in selected.values()),
    )


def rank_candidates(
    aggregates: Sequence[CandidateAggregate],
    baseline: Mapping[tuple[int, int], float],
    minimum_gain: float,
    maximum_regression: float,
) -> list[CandidateAggregate]:
    baseline_mean = sum(baseline.values()) / len(baseline)
    accepted = []
    for candidate in aggregates:
        task_deltas = [
            metrics.strict_online - baseline[task]
            for task, metrics in candidate.task_metrics.items()
        ]
        if candidate.mean_strict_online - baseline_mean < minimum_gain:
            continue
        if min(task_deltas) < -maximum_regression:
            continue
        accepted.append(candidate)
    return sorted(
        accepted,
        key=lambda item: (
            -item.mean_strict_online,
            -item.minimum_task_delta,
            item.wall_time_seconds,
            item.candidate_id,
        ),
    )


def validate_recovery(
    records: Mapping[tuple[int, int], RunnerMetrics],
    expected: Mapping[tuple[int, int], float],
    expected_mean: float,
    tolerance: float,
) -> RecoveryResult:
    failures: dict[tuple[int, int], float | None] = {}
    scores = []
    for task, expected_score in expected.items():
        metrics = records.get(task)
        if metrics is None:
            failures[task] = None
            continue
        scores.append(metrics.strict_online)
        delta = metrics.strict_online - expected_score
        if abs(delta) > tolerance:
            failures[task] = delta

    if len(scores) != len(expected):
        return RecoveryResult(False, None, None, failures)

    exact_mean = sum(scores) / len(scores)
    mean_delta = exact_mean - expected_mean
    passed = not failures and abs(mean_delta) <= tolerance
    return RecoveryResult(passed, exact_mean, mean_delta, failures)


def write_outputs(
    run_dir: Path,
    records: Sequence[Mapping[str, Any]],
    leaderboard: Sequence[CandidateAggregate],
    final_rows: Sequence[Mapping[str, Any]],
    report_context: Mapping[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_fields = [
        "stage",
        "candidate_id",
        "source",
        "target",
        "stream_seed",
        "before",
        "strict_online",
        "post_stream",
        "batches",
        "runtime_seconds",
        "status",
    ]
    with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metrics_fields)
        writer.writeheader()
        for record in sorted(
            records,
            key=lambda row: (
                str(row.get("stage", "")),
                str(row.get("candidate_id", "")),
                int(row.get("stream_seed", 0)),
                tuple(row["task"]),
            ),
        ):
            source, target = record["task"]
            metrics = record.get("metrics")
            writer.writerow(
                {
                    "stage": record.get("stage", ""),
                    "candidate_id": record.get("candidate_id", ""),
                    "source": source,
                    "target": target,
                    "stream_seed": record.get("stream_seed", ""),
                    "before": metrics.before if metrics else "",
                    "strict_online": metrics.strict_online if metrics else "",
                    "post_stream": metrics.post_stream if metrics else "",
                    "batches": metrics.batches if metrics else "",
                    "runtime_seconds": metrics.runtime_seconds if metrics else "",
                    "status": record.get("status", ""),
                }
            )

    leaderboard_fields = [
        "rank",
        "candidate_id",
        "mean_strict_online",
        "minimum_task_delta",
        "wall_time_seconds",
        "overrides",
    ]
    with (run_dir / "leaderboard.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=leaderboard_fields)
        writer.writeheader()
        for rank, candidate in enumerate(leaderboard, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "candidate_id": candidate.candidate_id,
                    "mean_strict_online": candidate.mean_strict_online,
                    "minimum_task_delta": candidate.minimum_task_delta,
                    "wall_time_seconds": candidate.wall_time_seconds,
                    "overrides": json.dumps(
                        candidate.overrides, sort_keys=True, separators=(",", ":")
                    ),
                }
            )

    final_fields = [
        "source",
        "target",
        "before",
        "strict_online",
        "post_stream",
        "baseline",
        "delta",
        "status",
        "runtime_seconds",
    ]
    with (run_dir / "final_12task.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=final_fields)
        writer.writeheader()
        for row in sorted(final_rows, key=lambda item: tuple(item["task"])):
            source, target = row["task"]
            metrics = row.get("metrics")
            baseline = row.get("baseline", HISTORICAL_BASELINE.get((source, target)))
            strict_online = metrics.strict_online if metrics else None
            writer.writerow(
                {
                    "source": source,
                    "target": target,
                    "before": metrics.before if metrics else "",
                    "strict_online": strict_online if strict_online is not None else "",
                    "post_stream": metrics.post_stream if metrics else "",
                    "baseline": baseline if baseline is not None else "",
                    "delta": (
                        strict_online - baseline
                        if strict_online is not None and baseline is not None
                        else ""
                    ),
                    "status": row.get("status", ""),
                    "runtime_seconds": metrics.runtime_seconds if metrics else "",
                }
            )

    recommended = "yes" if report_context.get("recommended") else "no"
    exact_mean = report_context.get("exact_mean")
    report = [
        "# PU4D 0711 Strict-Online Tuning Report",
        "",
        f"Recommended: {recommended}",
        f"Exact 12-task mean: {exact_mean if exact_mean is not None else 'NA'}",
        "",
    ]
    (run_dir / "report.md").write_text("\n".join(report), encoding="utf-8")


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _metric_object(value: Mapping[str, Any] | None) -> RunnerMetrics | None:
    if not isinstance(value, Mapping):
        return None
    required = ("before", "strict_online", "post_stream", "batches", "runtime_seconds")
    if any(key not in value for key in required):
        return None
    return RunnerMetrics(
        before=float(value["before"]),
        strict_online=float(value["strict_online"]),
        post_stream=float(value["post_stream"]),
        batches=int(value["batches"]),
        runtime_seconds=float(value["runtime_seconds"]),
    )


def load_run_records(run_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted((Path(run_dir) / "state").glob("*.json")):
        value = _read_json(path)
        if not isinstance(value, Mapping) or "task" not in value:
            continue
        row = dict(value)
        row["task"] = tuple(int(item) for item in value["task"])
        row["metrics"] = _metric_object(value.get("metrics"))
        row["state_path"] = str(path)
        records.append(row)
    return records


def _same_overrides(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(
        right, sort_keys=True, separators=(",", ":")
    )


def _latest_by_task(records: Sequence[Mapping[str, Any]]) -> dict[tuple[int, int], dict[str, Any]]:
    selected: dict[tuple[int, int], dict[str, Any]] = {}
    for record in records:
        task = tuple(record["task"])
        previous = selected.get(task)
        if previous is None or str(record.get("ended_at") or "") > str(
            previous.get("ended_at") or ""
        ):
            selected[task] = dict(record)
    return selected


def _candidate_aggregates(
    records: Sequence[Mapping[str, Any]],
) -> list[CandidateAggregate]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for record in records:
        if (
            record.get("status") != "succeeded"
            or int(record.get("stream_seed", -1)) != 2025
            or tuple(record["task"]) not in ((0, 1), (0, 3), (1, 0))
            or record.get("metrics") is None
        ):
            continue
        overrides = dict(record.get("overrides", {}))
        marker = json.dumps(overrides, sort_keys=True, separators=(",", ":"))
        grouped.setdefault((str(record.get("candidate_id", "")), marker), []).append(record)
    aggregates = []
    for rows in grouped.values():
        latest = list(_latest_by_task(rows).values())
        aggregate = aggregate_candidate(latest, ((0, 1), (0, 3), (1, 0)))
        if aggregate is not None:
            aggregates.append(aggregate)
    return sorted(
        aggregates,
        key=lambda item: (
            -item.mean_strict_online,
            -item.minimum_task_delta,
            item.wall_time_seconds,
            item.candidate_id,
        ),
    )


def _protocol_invariants_pass(
    final_records: Sequence[Mapping[str, Any]],
    frozen_at: str,
    recovery: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> bool:
    if not recovery.get("passed") or not preflight.get("passed"):
        return False
    if len(final_records) != len(HISTORICAL_BASELINE):
        return False
    for record in final_records:
        if record.get("status") != "succeeded" or record.get("metrics") is None:
            return False
        task = tuple(record["task"])
        command = [str(item) for item in record.get("command", [])]
        required = (
            "Dataset=PU4D",
            "gpu_id=0",
            "batch_size=128",
            "++TTA0711.mode=full",
            "++TTA0711.passes=1",
            "++TTA0711.stream_seed=2025",
        )
        if command and any(item not in command for item in required):
            return False
        if task not in ((0, 1), (0, 3), (1, 0)):
            if record.get("stage") != "final":
                return False
            if frozen_at and str(record.get("started_at") or "") < frozen_at:
                return False
    return True


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return lines


def generate_run_outputs(
    run_dir: Path,
    *,
    regression_tests_passed: bool,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    frozen_path = run_dir / "best_config.yaml"
    if not frozen_path.is_file():
        raise RuntimeError("best_config.yaml is required")
    frozen = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
    if not isinstance(frozen, Mapping) or not isinstance(frozen.get("overrides"), Mapping):
        raise RuntimeError("best_config.yaml has no frozen overrides")
    overrides = dict(frozen["overrides"])
    metadata = dict(frozen.get("metadata") or {})
    frozen_at = str(metadata.get("frozen_at") or "")
    records = load_run_records(run_dir)

    selected_rows = [
        record
        for record in records
        if record.get("status") == "succeeded"
        and int(record.get("stream_seed", -1)) == 2025
        and _same_overrides(dict(record.get("overrides", {})), overrides)
        and (
            (tuple(record["task"]) in ((0, 1), (0, 3), (1, 0)))
            or record.get("stage") == "final"
        )
    ]
    final_by_task = _latest_by_task(selected_rows)
    final_records = [final_by_task[task] for task in HISTORICAL_BASELINE if task in final_by_task]
    final_rows = [
        {
            "task": tuple(record["task"]),
            "metrics": record.get("metrics"),
            "baseline": HISTORICAL_BASELINE[tuple(record["task"])],
            "status": record.get("status", ""),
        }
        for record in final_records
    ]
    aggregates = _candidate_aggregates(records)
    recovery = _read_json(run_dir / "recovery_gate.json", {})
    preflight = _read_json(run_dir / "preflight.json", {})
    invariants_passed = _protocol_invariants_pass(
        final_records, frozen_at, recovery, preflight
    )
    scores = [row["metrics"].strict_online for row in final_rows if row["metrics"]]
    exact_mean = sum(scores) / len(scores) if len(scores) == 12 else None
    historical_mean = 62.7342
    recommended = bool(
        exact_mean is not None
        and exact_mean > historical_mean
        and invariants_passed
        and regression_tests_passed
    )

    write_outputs(
        run_dir,
        records=records,
        leaderboard=aggregates,
        final_rows=final_rows,
        report_context={"recommended": recommended, "exact_mean": exact_mean},
    )

    recovery_records = _latest_by_task(
        [record for record in records if record.get("stage") == "recovery"]
    )
    baseline_dev = {
        task: recovery_records[task]["metrics"].strict_online
        for task in ((0, 1), (0, 3), (1, 0))
        if task in recovery_records and recovery_records[task].get("metrics")
    }
    baseline_dev_mean = (
        sum(baseline_dev.values()) / len(baseline_dev) if len(baseline_dev) == 3 else None
    )
    failures = [record for record in records if record.get("status") != "succeeded"]
    report = [
        "# PU4D 0711 Strict-Online Tuning Report",
        "",
        f"Recommended: {'yes' if recommended else 'no'}",
        f"Protocol invariants: {'PASS' if invariants_passed else 'FAIL'}",
        f"Regression tests: {'PASS' if regression_tests_passed else 'FAIL/NOT RUN'}",
        f"Exact 12-task mean: {exact_mean if exact_mean is not None else 'NA'}",
        f"Rounded 12-task mean: {round(exact_mean, 2) if exact_mean is not None else 'NA'}",
        f"Historical baseline mean: {historical_mean}",
        f"Mean delta: {exact_mean - historical_mean if exact_mean is not None else 'NA'}",
        "",
        "## Protocol invariants",
        "",
        "Dataset=PU4D; cache=Dataset/PU4D_CACHE; model=ResNet18_1D_SDE; "
        "mode=full; passes=1; batch_size=128; source_seed=2025; "
        "stream_seed=2025; physical GPU=0; strict pre-update online metric.",
        "",
        "## Frozen configuration",
        "",
        "```yaml",
        yaml.safe_dump(frozen, sort_keys=True, allow_unicode=True).rstrip(),
        "```",
        "",
        "## Recovery gate",
        "",
    ]
    report.extend(
        _markdown_table(
            ("Task", "Strict online", "Historical", "Delta"),
            [
                (
                    f"{task[0]}→{task[1]}",
                    f"{record['metrics'].strict_online:.2f}",
                    f"{HISTORICAL_BASELINE[task]:.2f}",
                    f"{record['metrics'].strict_online - HISTORICAL_BASELINE[task]:+.2f}",
                )
                for task, record in sorted(recovery_records.items())
                if record.get("metrics")
            ],
        )
    )
    report.extend(["", "## Search history", ""])
    report.extend(
        _markdown_table(
            ("Rank", "Candidate", "Dev mean", "Min historical delta", "Guard"),
            [
                (
                    rank,
                    aggregate.candidate_id,
                    f"{aggregate.mean_strict_online:.4f}",
                    f"{aggregate.minimum_task_delta:+.4f}",
                    "accepted"
                    if baseline_dev_mean is not None
                    and aggregate.mean_strict_online - baseline_dev_mean >= 0.30
                    and min(
                        aggregate.task_metrics[task].strict_online - baseline_dev[task]
                        for task in baseline_dev
                    )
                    >= -0.50
                    else "rejected",
                )
                for rank, aggregate in enumerate(aggregates, start=1)
            ],
        )
    )
    report.extend(["", "## Final 12 tasks", ""])
    report.extend(
        _markdown_table(
            ("Task", "Before", "Strict online", "Post-stream", "Baseline", "Delta"),
            [
                (
                    f"{row['task'][0]}→{row['task'][1]}",
                    f"{row['metrics'].before:.2f}",
                    f"{row['metrics'].strict_online:.2f}",
                    f"{row['metrics'].post_stream:.2f}",
                    f"{row['baseline']:.2f}",
                    f"{row['metrics'].strict_online - row['baseline']:+.2f}",
                )
                for row in final_rows
                if row.get("metrics")
            ],
        )
    )
    report.extend(["", "## Failure records", ""])
    report.append(
        "None."
        if not failures
        else "\n".join(
            f"- {record.get('state_path')}: status={record.get('status')}, "
            f"failure_class={record.get('failure_class')}"
            for record in failures
        )
    )
    report.extend(["", "## Checkpoint hashes", ""])
    report.extend(
        _markdown_table(
            ("Source", "SHA-256", "Path"),
            [
                (row.get("source"), row.get("sha256"), row.get("path"))
                for row in preflight.get("checkpoints", [])
            ],
        )
    )
    report.extend(["", "## Environment", ""])
    environment = preflight.get("environment", {})
    report.extend(
        [
            f"- Python: {environment.get('python', 'NA')}",
            f"- PyTorch: {environment.get('pytorch', 'NA')}",
            f"- CUDA: {environment.get('cuda', 'NA')}",
            f"- GPU 0: {preflight.get('gpu0', 'NA')}",
            f"- Git HEAD: {preflight.get('git_head', 'NA')}",
            f"- Config SHA-256: {preflight.get('config_sha256', 'NA')}",
            "",
            "The tuned configuration is recommended only when the full 12-task mean "
            "exceeds 62.7342%, all strict invariants pass, and regression tests pass. "
            + ("The frozen configuration is recommended." if recommended else "The 0.015 baseline remains official."),
        ]
    )
    (run_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    result = {
        "recommended": recommended,
        "protocol_invariants_passed": invariants_passed,
        "regression_tests_passed": bool(regression_tests_passed),
        "exact_mean": exact_mean,
        "rounded_mean": round(exact_mean, 2) if exact_mean is not None else None,
        "historical_mean": historical_mean,
        "mean_delta": exact_mean - historical_mean if exact_mean is not None else None,
        "task_count": len(final_rows),
        "selected_overrides": overrides,
    }
    _write_json_atomic(run_dir / "summary_result.json", result)
    heldout_scores = [
        row["metrics"].strict_online
        for row in final_rows
        if tuple(row["task"]) not in ((0, 1), (0, 3), (1, 0)) and row.get("metrics")
    ]
    existing_final = _read_json(run_dir / "final_result.json", {})
    final_result = dict(existing_final) if isinstance(existing_final, Mapping) else {}
    final_result.update(
        {
            "heldout_mean": (
                sum(heldout_scores) / len(heldout_scores)
                if len(heldout_scores) == 9
                else None
            ),
            "exact_12task_mean": exact_mean,
            "recommended": recommended,
            "regression_tests_passed": bool(regression_tests_passed),
            "task_count": len(heldout_scores),
        }
    )
    _write_json_atomic(run_dir / "final_result.json", final_result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize strict PU4D 0711 tuning")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--regression-tests-passed", action="store_true")
    args = parser.parse_args(argv)
    result = generate_run_outputs(
        args.run_dir, regression_tests_passed=args.regression_tests_passed
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
