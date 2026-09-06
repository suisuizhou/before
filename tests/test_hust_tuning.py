from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
import shlex
import threading

import pytest
import yaml
import tools.tune_hust_0711_strict as tuning
import tools.summarize_hust_dtcc_0711 as summarizer

from tools.tune_hust_0711_strict import (
    DEV_TASKS,
    build_command,
    candidate_id,
    discover_idle_gpus,
    execute_parallel_tasks,
    expand_coordinate_group,
    freeze_candidate,
    load_frozen_candidate,
    plan_stage_jobs,
    rank_candidates,
    resume_pipeline,
    run_tuning_stage,
    validate_config,
)
from tools.summarize_hust_dtcc_0711 import parse_runner_log, write_reports
from tools.summarize_hust_dtcc_0711 import main as summarize_main
from main_tta_dtcc_hust_strict import build_hust_result_record


@pytest.fixture
def config():
    return yaml.safe_load(Path("Configs/Experiments/HUST0711_strict_tuning.yaml").read_text())


def test_schema_expands_exact_declared_partition_and_search(config):
    validate_config(config)
    assert config["tasks"]["development"] == [[0, 1], [1, 2], [2, 3], [3, 0]]
    all_tasks = {tuple(x) for x in config["tasks"]["development"] + config["tasks"]["heldout"]}
    assert all_tasks == {(s, t) for s in range(4) for t in range(4) if s != t}
    assert len(config["search"]["groups"]) == 10
    assert [x["Opt.lr_tar"] for x in config["search"]["groups"][0]["values"]] == [0.008, 0.015, 0.024, 0.036]
    assert config["budget"]["tuning_hours"] == 24
    assert config["selection"] == {
        "minimum_mean_gain": 0.30,
        "maximum_task_regression": 1.00,
        "stability_stream_seed": 2026,
    }


def test_candidate_expansion_retains_anchor_once_and_ids_are_order_independent():
    anchor = {"Opt.lr_tar": 0.015, "TTA0711.ema_beta": 0.995}
    values = [{"Opt.lr_tar": 0.008}, {"Opt.lr_tar": 0.015}, {"Opt.lr_tar": 0.008}]
    expanded = expand_coordinate_group(anchor, values)
    assert expanded == [
        {"Opt.lr_tar": 0.015, "TTA0711.ema_beta": 0.995},
        {"Opt.lr_tar": 0.008, "TTA0711.ema_beta": 0.995},
    ]
    assert candidate_id(anchor, 2025, 2025) == candidate_id(dict(reversed(list(anchor.items()))), 2025, 2025)
    assert candidate_id(anchor, 2025, 2025) == candidate_id(anchor, 2025, 2026)


def test_real_stage_plans_have_exact_route_counts_and_checkpoint_pairing(config, tmp_path):
    source = plan_stage_jobs(config, "source", tmp_path)
    beginning = plan_stage_jobs(config, "beginning", tmp_path)
    baseline = plan_stage_jobs(config, "baseline", tmp_path)
    assert len(source) == 8
    assert {(job["kind"], job["variant"], job["source"]) for job in source} == {
        ("source", variant, source_id) for variant in ("ordinary", "robust") for source_id in range(4)
    }
    assert len(beginning) == 24 and all(job["beginning_only"] for job in beginning)
    assert len(baseline) == 36
    assert {job["route"] for job in baseline} == {"dtcc_ordinary", "0711_robust", "0711_common"}
    for job in baseline:
        want = "robust" if job["route"] == "0711_robust" else "ordinary"
        assert job["variant"] == want and job["stage"] == "baseline"


def test_tune_plan_never_contains_heldout_and_final_requires_prior_freeze(
    config, tmp_path, monkeypatch
):
    tune = plan_stage_jobs(config, "tune", tmp_path)
    assert tune and {tuple(job["task"]) for job in tune} == set(DEV_TASKS)
    assert all(job["route"] == "0711_robust" and job["stage"].startswith("tune_group_") for job in tune)
    with pytest.raises(FileNotFoundError, match="best_config"):
        plan_stage_jobs(config, "final", tmp_path)
    freeze_path = tmp_path / "best_config.yaml"
    proof_path = tmp_path / "search_proof.json"
    freeze_path.write_text("authoritative freeze\n")
    proof_path.write_text("authoritative proof\n")
    overrides = {"Opt.lr_tar": 0.024}
    frozen = {
        "candidate_id": candidate_id(overrides, 2025),
        "overrides": overrides,
        "freeze_sha256": "a" * 64,
        "proof_path": str(proof_path.resolve()),
        "proof_sha256": "b" * 64,
        "frozen_at": 1.0,
    }
    monkeypatch.setattr(tuning, "load_frozen_candidate", lambda _path: frozen)
    final = plan_stage_jobs(config, "final", tmp_path)
    heldout = [job for job in final if job["stage"] == "heldout"]
    stability = [job for job in final if job["stage"] == "stability"]
    assert len(heldout) == 8 and {job["candidate_id"] for job in heldout} == {frozen["candidate_id"]}
    assert len(final) == 24 and len(stability) == 16
    assert {job["candidate_id"] for job in stability} == {"untuned", frozen["candidate_id"]}
    assert {tuple(job["task"]) for job in stability} == set(map(tuple, config["tasks"]["heldout"]))
    assert not ({tuple(job["task"]) for job in stability} & set(DEV_TASKS))


def test_final_plan_untuned_fallback_is_one_exact_bound_inventory(
    config, tmp_path, monkeypatch
):
    freeze_path = tmp_path / "best_config.yaml"
    proof_path = tmp_path / "search_proof.json"
    freeze_path.write_text("authoritative freeze\n")
    proof_path.write_text("authoritative proof\n")
    frozen = {
        "candidate_id": "untuned",
        "overrides": dict(config["baseline_overrides"]),
        "freeze_sha256": "a" * 64,
        "proof_path": str(proof_path.resolve()),
        "proof_sha256": "b" * 64,
        "frozen_at": 1.0,
    }
    monkeypatch.setattr(tuning, "load_frozen_candidate", lambda _path: frozen)
    final = plan_stage_jobs(config, "final", tmp_path, resolve_artifacts=False)
    heldout = [job for job in final if job["stage"] == "heldout"]
    stability = [job for job in final if job["stage"] == "stability"]
    assert len(heldout) == 8 and len(stability) == 8
    assert all(job["candidate_id"] == "untuned" for job in final)
    assert all(job["freeze_config_path"] == str(freeze_path.resolve()) for job in final)
    assert len({(job["stage"], tuple(job["task"]), job["stream_seed"]) for job in final}) == 16


def test_load_audit_is_supplementary_beginning_only(config, tmp_path):
    jobs = plan_stage_jobs(config, "load-audit", tmp_path)
    assert len(jobs) == 18  # six sources plus two checkpoint families x six directed tasks
    assert all(job["load_split"] for job in jobs)
    assert not any(job.get("stage", "").startswith("tune") for job in jobs)
    source_command = tuning.build_stage_command(config, "source", source=0, variant="ordinary", load_split=True)
    target_command = tuning._target_command_for_job(config, next(job for job in jobs if job["kind"] == "target"), 0)
    assert "++hust_protocol_split=load" in source_command and "++hust_protocol_split=load" in target_command
    assert any("TTA_Model_HUST_STRICT_LOAD" in arg for arg in source_command + target_command)


def test_coordinate_search_is_sequential_and_freezes_before_any_heldout(
    config, tmp_path, monkeypatch
):
    calls = []
    baseline = [
        {"candidate_id": "untuned", "route": "0711_robust", "stage": "baseline", "task": task,
         "stream_seed": 2025, "status": "succeeded", "strict_online": 50.0}
        for task in DEV_TASKS
    ]

    def fake_executor(_run_dir, _config, jobs, _gpus, **_kwargs):
        calls.append(list(jobs))
        rows = []
        for job in jobs:
            lr = float(job["overrides"].get("Opt.lr_tar", 0.015))
            score = 51.0 if lr == 0.024 else 50.5
            rows.append({**job, "status": "succeeded", "strict_online": score, "runtime_seconds": 1.0})
        return rows

    selected = {}

    def fake_proof(*_args, **kwargs):
        selected.update(kwargs["selected"])
        path = tmp_path / "search_proof.json"
        path.write_text("proof\n")
        return path

    def fake_freeze(_run_dir, identifier, overrides, **_kwargs):
        selected.update(candidate_id=identifier, overrides=dict(overrides))
        path = tmp_path / "best_config.yaml"
        path.write_text("freeze\n")
        return path

    monkeypatch.setattr(tuning, "_build_primary_checkpoint_manifest", lambda _rows: {})
    monkeypatch.setattr(tuning, "write_search_proof", fake_proof)
    monkeypatch.setattr(tuning, "freeze_candidate", fake_freeze)
    monkeypatch.setattr(
        tuning,
        "load_frozen_candidate",
        lambda _path: {
            **selected,
            "freeze_sha256": "a" * 64,
            "proof_sha256": "b" * 64,
        },
    )

    frozen = run_tuning_stage(tmp_path, config, [0], executor=fake_executor, baseline_records=baseline)
    assert frozen["overrides"]["Opt.lr_tar"] == 0.024
    assert all(tuple(job["task"]) in set(DEV_TASKS) for batch in calls for job in batch)
    assert len(calls) == 11  # ten sequential coordinate groups, then matched stability
    call_count = len(calls)
    assert run_tuning_stage(tmp_path, config, [0], executor=fake_executor, baseline_records=baseline) == frozen
    assert len(calls) == call_count


def test_auto_gpu_filter_uses_only_idle_rows():
    rows = "0, RTX, 16311, 15, 0\n1, RTX, 32760, 16247, 99\n2, RTX, 32760, 200, 3\n"
    assert discover_idle_gpus(rows, max_utilization=10, max_memory_fraction=0.10) == [0, 2]


@pytest.mark.parametrize(
    ("route", "runner", "variant"),
    [
        ("dtcc_ordinary", "main_tta_dtcc_hust_strict.py", None),
        ("0711_robust", "main_tta_0711_hust_strict.py", "robust"),
        ("0711_common", "main_tta_0711_hust_strict.py", "ordinary"),
    ],
)
def test_target_commands_force_route_and_strict_invariants(config, route, runner, variant):
    command = build_command(config, route, {"Opt.lr_tar": 0.024}, (0, 1), 2025, gpu=2)
    joined = " ".join(command)
    assert runner in joined
    if variant:
        assert f"source_variant={variant}" in joined
        assert "TTA0711.passes=1" in joined
        assert "TTA0711.sampling_rate_hz=51200" in joined
        assert "TTA0711.fft_size=2048" in joined
    assert "gpu_id=0" in joined and "only_task=[0,1]" in joined
    assert "num_workers=4" in joined


def test_every_source_and_load_command_forces_four_workers(config):
    """Dropping the worker override from either source family or split must fail."""
    commands = [
        tuning.build_stage_command(
            config, "source", source=source, variant=variant, load_split=load_split
        )
        for load_split in (False, True)
        for variant in ("ordinary", "robust")
        for source in (0,)
    ]
    assert all("num_workers=4" in command for command in commands)


def test_formal_config_uses_new_cache_and_checkpoint_roots(config):
    """Future formal work must never silently append to the historical V1 lineage."""
    validate_config(config)
    assert config["protocol"]["data_path"] == "Dataset/HUST_STRICT_CACHE_V2"
    assert config["protocol"]["load_data_path"] == "Dataset/HUST_STRICT_LOAD_CACHE_V2"
    assert config["checkpoints"] == {
        "root": "TTA_Model_HUST_STRICT_V2",
        "load_root": "TTA_Model_HUST_STRICT_LOAD_V2",
    }


def test_incomplete_failed_and_regressing_candidates_never_rank():
    required = {(0, 1), (1, 2), (2, 3), (3, 0)}
    baseline = {task: 50.0 for task in required}
    rows = [
        {"candidate_id": "partial", "task": task, "status": "succeeded", "strict_online": 60.0}
        for task in list(required)[:3]
    ]
    rows += [
        {"candidate_id": "bad", "task": task, "status": "succeeded", "strict_online": 55.0 if task != (0, 1) else 48.9, "runtime_seconds": 1}
        for task in required
    ]
    rows += [
        {"candidate_id": "winner", "task": task, "status": "succeeded", "strict_online": 51.0, "runtime_seconds": 2}
        for task in required
    ]
    rows.append({"candidate_id": "winner", "task": (0, 1), "status": "failed", "strict_online": 99.0})
    rows.pop()  # A duplicate failed identity is now rejected rather than ignored.
    for row in rows:
        row.update(route="0711_robust", variant="robust", stage="tune_group_00", source=row["task"][0], target=row["task"][1],
                   source_seed=2025, stream_seed=2025, config_sha256="a" * 64)
    assert [row["candidate_id"] for row in rank_candidates(rows, required, baseline, 0.30, 1.00, expected_stages={2025: "tune_group_00"})] == ["winner"]


def test_stability_ranking_is_deterministic_then_minimum_delta_then_runtime():
    required = {(0, 1), (1, 2)}
    baseline = {task: 50.0 for task in required}
    rows = []
    for cid, scores, runtime in [("slow", [51.5, 50.5], 5), ("balanced", [51.0, 51.0], 6)]:
        for seed in (2025, 2026):
            for task, score in zip(sorted(required), scores):
                rows.append({"candidate_id": cid, "task": task, "stream_seed": seed, "status": "succeeded", "strict_online": score, "runtime_seconds": runtime})
    for row in rows:
        row.update(route="0711_robust", variant="robust", stage="tune_stability" if row["stream_seed"] == 2026 else "tune_group_00", source=row["task"][0], target=row["task"][1],
                   source_seed=2025, config_sha256="a" * 64)
    assert [x["candidate_id"] for x in rank_candidates(rows, required, baseline, 0.30, 1.00, required_seeds={2025, 2026}, expected_stages={2025: "tune_group_00", 2026: "tune_stability"})] == ["balanced", "slow"]


def test_matched_seed2026_guard_uses_matching_baseline():
    tasks = {(0, 1), (1, 2)}
    baseline = {(task, seed): 50.0 for task in tasks for seed in (2025, 2026)}
    rows = []
    for seed, score in ((2025, 51.0), (2026, 50.2)):
        rows.extend({"candidate_id": "unstable", "task": task, "stream_seed": seed, "status": "succeeded", "strict_online": score} for task in tasks)
    for row in rows:
        row.update(route="0711_robust", variant="robust", stage="tune_group_00", source=row["task"][0], target=row["task"][1],
                   source_seed=2025, config_sha256="a" * 64)
        if row["stream_seed"] == 2026:
            row["stage"] = "tune_stability"
    assert rank_candidates(rows, tasks, baseline, required_seeds={2025, 2026}, expected_stages={2025: "tune_group_00", 2026: "tune_stability"}) == []


def test_matched_final_ranking_applies_eligibility_only_to_seed2025():
    """Reapplying the dev gate to seed 2026 would incorrectly discard a matched finalist."""
    tasks = {(0, 1), (1, 2)}
    baseline = {(task, seed): 50.0 for task in tasks for seed in (2025, 2026)}
    rows = []
    for seed, scores in ((2025, (51.0, 51.0)), (2026, (49.0, 50.0))):
        for task, score in zip(sorted(tasks), scores):
            rows.append({
                "candidate_id": "eligible-dev", "route": "0711_robust",
                "variant": "robust", "stage": "tune_group_00" if seed == 2025 else "tune_stability",
                "task": task, "source": task[0], "target": task[1], "source_seed": 2025,
                "stream_seed": seed, "config_sha256": "a" * 64, "status": "succeeded",
                "strict_online": score, "runtime_seconds": 1.0,
            })
    ranked = rank_candidates(
        rows, tasks, baseline, required_seeds={2025, 2026},
        expected_stages={2025: "tune_group_00", 2026: "tune_stability"},
        eligibility_seeds={2025},
    )
    assert [row["candidate_id"] for row in ranked] == ["eligible-dev"]


def test_global_coordinate_ranking_deduplicates_all_complete_eligible_candidates():
    """Selecting only recent group winners would miss the globally best earlier candidate."""
    rows = []
    tasks = set(DEV_TASKS)
    baseline = {task: 50.0 for task in tasks}
    for stage, candidate, score in (
        ("tune_group_00_learning_rate", "global-best", 53.0),
        ("tune_group_01_adapter_lr_scale", "recent-winner", 52.0),
        ("tune_group_09_contrastive_temperature", "recent-winner", 52.0),
    ):
        for task in tasks:
            rows.append({
                "candidate_id": candidate, "route": "0711_robust", "variant": "robust",
                "stage": stage, "task": task, "source": task[0], "target": task[1],
                "source_seed": 2025, "stream_seed": 2025, "config_sha256":
                    ("a" if candidate == "global-best" else "b") * 64,
                "status": "succeeded", "strict_online": score, "runtime_seconds": 1.0,
                "overrides": {"Opt.lr_tar": 0.024 if candidate == "global-best" else 0.015},
            })
    ranked = tuning.rank_coordinate_candidates_globally(
        rows, tasks, baseline, minimum_gain=0.30, maximum_regression=1.00
    )
    assert [row["candidate_id"] for row in ranked] == ["global-best", "recent-winner"]


def test_candidate_overrides_must_match_identity_and_declared_search_space(config):
    """A valid-looking candidate id must not authorize an undeclared override/value."""
    valid = {"Opt.lr_tar": 0.024, "TTA0711.warp_lr_scale": 0.10}
    tuning.validate_candidate_overrides(config, candidate_id(valid, 2025), valid)
    with pytest.raises(ValueError, match="candidate_id"):
        tuning.validate_candidate_overrides(config, "0" * 12, valid)
    with pytest.raises(ValueError, match="search space"):
        tuning.validate_candidate_overrides(
            config,
            candidate_id({"Opt.lr_tar": 9.9}, 2025),
            {"Opt.lr_tar": 9.9},
        )


def test_rank_candidates_rejects_foreign_identity_duplicate_and_extra_rows():
    tasks = {(0, 1), (1, 2)}
    baseline = {task: 50.0 for task in tasks}
    valid = [{"candidate_id": "candidate", "route": "0711_robust", "variant": "robust",
              "stage": "tune_group_03", "task": task, "source": task[0], "target": task[1],
              "source_seed": 2025, "stream_seed": 2025, "config_sha256": "a" * 64,
              "status": "succeeded", "strict_online": 51.0}
             for task in tasks]
    kwargs = {"expected_stages": {2025: "tune_group_03"}}
    assert [row["candidate_id"] for row in rank_candidates(valid, tasks, baseline, **kwargs)] == ["candidate"]
    for mutation in (
        {"route": "dtcc_ordinary"}, {"variant": "ordinary"}, {"stage": "baseline"},
        {"stream_seed": 2026}, {"task": (2, 3), "source": 2, "target": 3},
    ):
        bad = [dict(row) for row in valid]
        bad[0].update(mutation)
        assert rank_candidates(bad, tasks, baseline, **kwargs) == []
    assert rank_candidates([*valid, dict(valid[0])], tasks, baseline, **kwargs) == []


def test_rank_candidates_three_argument_call_fails_closed_for_incomplete_rows():
    required = {(0, 1), (1, 2), (2, 3), (3, 0)}
    rows = [
        {"candidate_id": "a", "task": task, "status": "succeeded", "strict_online": 50.0}
        for task in [(0, 1), (1, 2), (2, 3)]
    ]
    assert rank_candidates(rows, required, {}) == []


def test_rank_candidates_three_argument_call_infers_one_complete_stage():
    tasks = {(0, 1), (1, 2)}
    rows = [
        {"candidate_id": "candidate", "route": "0711_robust", "variant": "robust",
         "stage": "tune_group_03", "task": task, "source": task[0], "target": task[1],
         "source_seed": 2025, "stream_seed": 2025, "config_sha256": "a" * 64,
         "status": "succeeded", "strict_online": 51.0}
        for task in tasks
    ]
    assert [row["candidate_id"] for row in rank_candidates(rows, tasks, {})] == ["candidate"]


@pytest.mark.parametrize(
    ("mutation", "remove_key", "apply_to_all"),
    [
        ({}, "config_sha256", False),
        ({"config_sha256": "b" * 64}, None, False),
        ({"config_sha256": "malformed"}, None, True),
        ({}, "source_seed", False),
        ({"source_seed": 2026}, None, False),
        ({"source_seed": 2025.0}, None, True),
        ({"source_seed": True}, None, True),
        ({"source_seed": "2025"}, None, True),
    ],
    ids=[
        "missing-config-hash", "differing-config-hash", "malformed-config-hash",
        "missing-source-seed", "wrong-source-seed", "float-source-seed",
        "bool-source-seed", "string-source-seed",
    ],
)
def test_rank_candidates_rejects_invalid_formal_config_or_source_identity(mutation, remove_key, apply_to_all):
    tasks = {(0, 1), (1, 2)}
    rows = [
        {"candidate_id": "candidate", "route": "0711_robust", "variant": "robust",
         "stage": "tune_group_03", "task": task, "source": task[0], "target": task[1],
         "source_seed": 2025, "stream_seed": 2025, "config_sha256": "a" * 64,
         "status": "succeeded", "strict_online": 51.0}
        for task in sorted(tasks)
    ]
    if remove_key is not None:
        rows[0].pop(remove_key)
    for row in rows if apply_to_all else rows[:1]:
        row.update(mutation)
    assert rank_candidates(rows, tasks, {}, expected_stages={2025: "tune_group_03"}) == []


def test_rank_candidates_rejects_mixed_source_seed_values():
    tasks = {(0, 1), (1, 2)}
    rows = [
        {"candidate_id": "candidate", "route": "0711_robust", "variant": "robust",
         "stage": "tune_group_03", "task": task, "source": task[0], "target": task[1],
         "source_seed": source_seed, "stream_seed": 2025, "config_sha256": "a" * 64,
         "status": "succeeded", "strict_online": 51.0}
        for task, source_seed in zip(sorted(tasks), (2025, 2026))
    ]
    assert rank_candidates(rows, tasks, {}, expected_stages={2025: "tune_group_03"}) == []


def test_freeze_is_immutable_and_hash_validated(tmp_path):
    core = {
        "candidate_id": candidate_id({"Opt.lr_tar": 0.024}, 2025),
        "overrides": {"Opt.lr_tar": 0.024},
        "frozen_before_heldout": True,
    }
    payload = {
        **core,
        "freeze_sha256": hashlib.sha256(
            json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    frozen = tmp_path / "best_config.yaml"
    frozen.write_text(yaml.safe_dump(payload))

    with pytest.raises(ValueError, match="authoritative|proof"):
        load_frozen_candidate(frozen)


def _successful_log(path: Path, strict: float = 61.25):
    path.write_text(
        f"Beginning Acc T = 42.00%\nStrict Online Acc = {strict:.2f}%\n"
        "Strict Online Macro P/R/F1 = 60.00/59.00/58.00%\n"
        "Post-stream Full-Target Acc = 63.00%\nPost-stream Macro P/R/F1 = 62.00/61.00/60.00%\n"
        "[STRICT HUST] batches=12 mean_batch_ms=25.0 peak_memory_mb=3210 memory_size=40 memory_class_coverage=7 "
        "certain_ratio=0.7 uncertain_ratio=0.3 offline_purity=0.8\n"
    )


_DECLARED_TARGET_EVIDENCE_KEYS = {
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
}

_JOB_RUNTIME_SPOOFS = {
    "status": "job-spoof", "attempt": 987, "returncode": 987,
    "metrics": {"before": 99.0}, "strict_online": 99.0, "before": 99.0,
    "beginning": 99.0, "macro_precision": 99.0, "macro_recall": 99.0,
    "macro_f1": 99.0, "post_stream": 99.0, "post_macro_precision": 99.0,
    "post_macro_recall": 99.0, "post_macro_f1": 99.0, "peak_memory_mb": 999.0,
    "result_kind": "job-spoof", "runtime_seconds": 999.0, "samples": 999,
    "batches": 999, "passes": 999, "class_coverage": 999,
    "confusion_matrix": [[999]], "offline_confusion_matrix": [[999]],
    "memory_size": 999, "memory_class_coverage": 999, "certain_ratio": 99.0,
    "uncertain_ratio": 99.0, "offline_purity": 99.0, "certain_purity": 99.0,
    "evidence_applicable": "job-spoof", "evidence_active_ratio": 99.0,
    "finite_losses": "job-spoof", "trainable_parameters": ["job-spoof"],
    "trainable_allowlist": ["job-spoof"], "pre_update_scoring": "job-spoof",
    "metadata_evidence_used": "job-spoof", "mean_batch_ms": 999.0,
    "schema_version": 999, "failure_class": "job-spoof", "parse_error": "job-spoof",
    "error": "job-spoof", "exception": "job-spoof", "error_message": "job-spoof",
    "gpu": 999, "command": ["job-spoof"], "command_sha256": "job-spoof",
    "command_path": "job-spoof", "command_log_sha256": "job-spoof",
    "log_path": "job-spoof", "log_sha256": "job-spoof",
    "started_at": 999.0, "ended_at": 999.0, "finished_at": 999.0,
    "started": 999.0, "finished": 999.0,
}


def _install_fake_v2_cache(monkeypatch, cache: Path, *, split: str) -> dict:
    domains = 3 if split == "load" else 4
    cache.mkdir(exist_ok=True)
    manifest = cache / "manifest.json"
    manifest.write_text(json.dumps({"version": 2, "split": split}) + "\n")
    summaries = {}
    for domain in range(domains):
        tensor = cache / f"domain_{domain}.pt"
        tensor.write_bytes(f"tensor-{split}-{domain}\n".encode())
        summaries[str(domain)] = {
            "tensor_file": tensor.name,
            "tensor_sha256": hashlib.sha256(tensor.read_bytes()).hexdigest(),
        }
    result = {
        "version": 2,
        "split": split,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "content_sha256": hashlib.sha256(
            json.dumps(summaries, sort_keys=True).encode()
        ).hexdigest(),
        "domains": summaries,
    }
    import Lib.hust_strict_protocol as protocol
    original = protocol.validate_cache

    def validate(path):
        if Path(path).resolve() == cache.resolve():
            return result
        return original(path)

    monkeypatch.setattr(protocol, "validate_cache", validate)
    return result


def _declared_target_job(config, tmp_path: Path, monkeypatch, stage: str) -> tuple[dict, dict]:
    cache = tmp_path / "cache"
    cache_identity = _install_fake_v2_cache(monkeypatch, cache, split="bearing")
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_dir = checkpoint_root / "robust" / "source_0" / "seed_2025"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "best_source_ResNet18_1D_SDE2025fft_Linear.pt"
    checkpoint.write_bytes(b"declared checkpoint\n")
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (checkpoint_dir / "source_training_summary.json").write_text(json.dumps({
        "route": "robust",
        "source": 0,
        "seed": 2025,
        "checkpoint_sha256": checkpoint_hash,
        "target_labels_consumed": False,
        "cache_identity": {
            "mode": "formal",
            "manifest_path": str((cache / "manifest.json").resolve()),
            "manifest_sha256": cache_identity["manifest_sha256"],
            "content_sha256": cache_identity["content_sha256"],
            "tensor_sha256s": {
                str((cache / row["tensor_file"]).resolve()): row["tensor_sha256"]
                for row in cache_identity["domains"].values()
            },
        },
    }))
    runner = tmp_path / "target_runner.py"
    runner.write_text("# declared runner\n")
    cfg = json.loads(json.dumps(config))
    cfg["protocol"]["data_path"] = str(cache)
    cfg["checkpoints"]["root"] = str(checkpoint_root)
    cfg["routes"]["dtcc_ordinary"] = str(runner)
    cfg["routes"]["0711_robust"] = str(runner)
    monkeypatch.setattr(tuning, "validate_config", lambda _config: None)
    jobs = plan_stage_jobs(cfg, stage, tmp_path)
    if stage == "beginning":
        job = next(row for row in jobs if row["route"] == "0711_robust" and tuple(row["task"]) == (0, 1))
    else:
        job = next(row for row in jobs if row["route"] == "0711_robust" and tuple(row["task"]) == (0, 1))
    return cfg, job


def _write_formal_target_log(path: Path, job: dict) -> None:
    beginning_only = job["expected_result_kind"] == "beginning"
    confusion = None if beginning_only else [[5 if row == column else 0 for column in range(7)] for row in range(7)]
    diagnostics = {} if beginning_only else {
        "memory_size": 20, "memory_class_coverage": 7, "certain_ratio": 0.6,
        "uncertain_ratio": 0.4, "offline_purity": 100.0, "certain_purity": 95.0,
        "evidence_applicable": True, "evidence_active_ratio": 0.5, "mean_batch_ms": 125.0,
        "post_macro_precision": 100.0, "post_macro_recall": 100.0,
        "post_macro_f1": 100.0,
    }
    if not beginning_only and job["route"].startswith("0711"):
        diagnostics["offline_confusion_matrix"] = confusion
    result = build_hust_result_record(
        result_kind=job["expected_result_kind"], route=job["route"], variant=job["variant"],
        task=tuple(job["task"]), source_checkpoint_sha256=job["source_checkpoint_sha256"],
        config_sha256=job["config_sha256"], candidate_id=job["candidate_id"],
        source_seed=job["source_seed"], stream_seed=job["stream_seed"], beginning=42.0,
        strict_online=None if beginning_only else 100.0,
        post_stream=None if beginning_only else 100.0, confusion_matrix=confusion,
        samples=35, batches=0 if beginning_only else 4, passes=0 if beginning_only else 1,
        finite_losses=True, trainable_parameters=[] if beginning_only else ["backbone.band_scale"],
        pre_update_scoring=True, metadata_evidence_used=not beginning_only,
        runtime_seconds=0.5, peak_memory_mb=10.0,
        **diagnostics,
    )
    path.write_text("HUST_RESULT_JSON=" + json.dumps(result, separators=(",", ":")) + "\n")


def _capture_atomic_states(monkeypatch) -> list[dict]:
    writes = []
    real_atomic_write = tuning.atomic_write_json

    def capture(path, payload):
        json.dumps(payload)
        real_atomic_write(path, payload)
        writes.append(json.loads(Path(path).read_text()))

    monkeypatch.setattr(tuning, "atomic_write_json", capture)
    return writes


def _assert_declared_target_evidence(states: list[dict], job: dict) -> None:
    expected = json.loads(json.dumps(job))
    assert _DECLARED_TARGET_EVIDENCE_KEYS <= expected.keys()
    for state in states:
        assert {key: state[key] for key in _DECLARED_TARGET_EVIDENCE_KEYS} == {
            key: expected[key] for key in _DECLARED_TARGET_EVIDENCE_KEYS
        }
        json.dumps(state)


def _load_audit_scheduler_seam(config, tmp_path: Path, monkeypatch) -> tuple[dict, dict, dict]:
    cache = tmp_path / "load-cache"
    _install_fake_v2_cache(monkeypatch, cache, split="load")
    runners = tmp_path / "runners"
    runners.mkdir()
    cfg = json.loads(json.dumps(config))
    cfg["protocol"]["load_data_path"] = str(cache)
    cfg["checkpoints"]["load_root"] = str(tmp_path / "load-checkpoints")
    for route in ("source_ordinary", "source_robust", "dtcc_ordinary", "0711_robust"):
        runner = runners / f"{route}.py"
        runner.write_text(f"# {route} load runner\n")
        cfg["routes"][route] = str(runner)
    monkeypatch.setattr(tuning, "validate_config", lambda _config: None)
    jobs = plan_stage_jobs(cfg, "load-audit", tmp_path)
    source = next(
        job for job in jobs
        if job["kind"] == "source" and job["variant"] == "ordinary" and job["source"] == 0
    )
    target = next(
        job for job in jobs
        if job["kind"] == "target" and job["variant"] == "ordinary"
        and tuple(job["task"]) == (0, 1)
    )
    assert target["evidence_status"] == "unresolved"
    assert target["source_checkpoint_sha256"] is None
    assert target["source_summary_sha256"] is None
    return cfg, source, target


def _load_audit_job_for_log(config, jobs: list[dict], log_path: Path) -> dict:
    matches = [
        job for job in jobs
        if log_path.name.startswith(tuning._job_identity(config, job)[1] + ".attempt-")
    ]
    assert len(matches) == 1
    return matches[0]


def _publish_load_source(job: dict, *, target_labels_consumed: bool = False) -> tuple[Path, Path, str, str]:
    checkpoint = Path(job["source_checkpoint_path"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"load source checkpoint\n")
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    summary = Path(job["source_summary_path"])
    summary.write_text(json.dumps({
        "route": job["variant"],
        "source": job["source"],
        "seed": 2025,
        "checkpoint_sha256": checkpoint_hash,
        "target_labels_consumed": target_labels_consumed,
        "cache_identity": {
            "mode": "formal",
            "manifest_path": str(Path(job["cache_manifest_path"]).resolve()),
            "manifest_sha256": job["cache_manifest_sha256"],
            "content_sha256": job["cache_content_sha256"],
            "tensor_sha256s": {
                str(Path(path).resolve()): digest
                for path, digest in job["cache_tensor_sha256s"].items()
            },
        },
    }))
    summary_hash = hashlib.sha256(summary.read_bytes()).hexdigest()
    return checkpoint, summary, checkpoint_hash, summary_hash


def test_load_audit_multigpu_scheduler_waits_for_all_sources_before_any_target_launch(
        config, tmp_path, monkeypatch):
    cfg, _source, _target = _load_audit_scheduler_seam(config, tmp_path, monkeypatch)
    jobs = plan_stage_jobs(cfg, "load-audit", tmp_path)
    completed_sources = set()
    source_launches = []
    target_launches = []
    early_targets = []
    release_last_source = threading.Event()
    lock = threading.Lock()

    def run(command, log_path, _env):
        job = _load_audit_job_for_log(cfg, jobs, log_path)
        if any(argument.startswith("only_source=") for argument in command):
            identity = (job["variant"], job["source"])
            source_launches.append((identity, _env["CUDA_VISIBLE_DEVICES"]))
            if identity == ("robust", 2):
                release_last_source.wait(timeout=0.5)
            _publish_load_source(job)
            with lock:
                completed_sources.add(identity)
            log_path.write_text("load source complete\n")
            return 0
        with lock:
            completed_at_launch = set(completed_sources)
            if len(completed_at_launch) != 6:
                early_targets.append((tuple(job["task"]), completed_at_launch))
                release_last_source.set()
        target_launches.append((tuple(job["task"]), _env["CUDA_VISIBLE_DEVICES"]))
        checkpoint_hash = hashlib.sha256(Path(job["source_checkpoint_path"]).read_bytes()).hexdigest()
        _write_formal_target_log(log_path, {**job, "source_checkpoint_sha256": checkpoint_hash})
        return 0

    try:
        records = execute_parallel_tasks(tmp_path / "run", cfg, jobs, [2, 5], run_process=run)
    finally:
        release_last_source.set()

    assert early_targets == []
    assert len(source_launches) == 6
    assert len(target_launches) == 12
    assert {gpu for _task, gpu in target_launches} == {"2", "5"}
    assert all(record["status"] == "succeeded" for record in records)

    resumed = execute_parallel_tasks(tmp_path / "run", cfg, jobs, [2, 5], run_process=run)
    assert len(source_launches) == 6
    assert len(target_launches) == 12
    assert resumed == records


def test_load_audit_source_failure_blocks_all_targets_and_retry_runs_only_failed_source(
        config, tmp_path, monkeypatch):
    cfg, _source, _target = _load_audit_scheduler_seam(config, tmp_path, monkeypatch)
    jobs = plan_stage_jobs(cfg, "load-audit", tmp_path)
    run_dir = tmp_path / "run"
    source_launches = []
    target_launches = []

    def fail_one_source(command, log_path, _env):
        job = _load_audit_job_for_log(cfg, jobs, log_path)
        if any(argument.startswith("only_source=") for argument in command):
            identity = (job["variant"], job["source"])
            source_launches.append(identity)
            if identity == ("robust", 2):
                log_path.write_text("source protocol failure\n")
                return 2
            _publish_load_source(job)
            log_path.write_text("load source complete\n")
            return 0
        target_launches.append(tuple(job["task"]))
        checkpoint_hash = hashlib.sha256(Path(job["source_checkpoint_path"]).read_bytes()).hexdigest()
        _write_formal_target_log(log_path, {**job, "source_checkpoint_sha256": checkpoint_hash})
        return 0

    with pytest.raises(RuntimeError, match="load-audit source phase failed"):
        execute_parallel_tasks(run_dir, cfg, jobs, [2, 5], run_process=fail_one_source)

    assert len(source_launches) == 6
    assert target_launches == []
    source_launches.clear()

    def recover_failed_source(command, log_path, _env):
        job = _load_audit_job_for_log(cfg, jobs, log_path)
        if any(argument.startswith("only_source=") for argument in command):
            source_launches.append((job["variant"], job["source"]))
            _publish_load_source(job)
            log_path.write_text("load source recovered\n")
            return 0
        target_launches.append(tuple(job["task"]))
        checkpoint_hash = hashlib.sha256(Path(job["source_checkpoint_path"]).read_bytes()).hexdigest()
        _write_formal_target_log(log_path, {**job, "source_checkpoint_sha256": checkpoint_hash})
        return 0

    records = execute_parallel_tasks(
        run_dir, cfg, jobs, [2, 5], run_process=recover_failed_source, retry_failed=True
    )

    assert source_launches == [("robust", 2)]
    assert len(target_launches) == 12
    recovered = next(
        record for record in records
        if record["kind"] == "source" and record["variant"] == "robust" and record["source"] == 2
    )
    assert recovered["status"] == "succeeded" and recovered["attempt"] == 2


def test_load_audit_dependency_barrier_rechecks_deadline_before_target_phase(
        config, tmp_path, monkeypatch):
    cfg, _source, _target = _load_audit_scheduler_seam(config, tmp_path, monkeypatch)
    cfg["budget"]["reserve_minutes"] = 0
    cfg["budget"]["estimated_task_minutes"] = 0
    jobs = plan_stage_jobs(cfg, "load-audit", tmp_path)
    clock = [0.0]
    completed_sources = set()
    target_launches = []
    lock = threading.Lock()
    monkeypatch.setattr(tuning.time, "time", lambda: clock[0])

    def run(command, log_path, _env):
        job = _load_audit_job_for_log(cfg, jobs, log_path)
        if any(argument.startswith("only_source=") for argument in command):
            _publish_load_source(job)
            with lock:
                completed_sources.add((job["variant"], job["source"]))
                if len(completed_sources) == 6:
                    clock[0] = 2.0
            log_path.write_text("load source complete\n")
            return 0
        target_launches.append(tuple(job["task"]))
        checkpoint_hash = hashlib.sha256(Path(job["source_checkpoint_path"]).read_bytes()).hexdigest()
        _write_formal_target_log(log_path, {**job, "source_checkpoint_sha256": checkpoint_hash})
        return 0

    with pytest.raises(TimeoutError, match="before launch"):
        execute_parallel_tasks(
            tmp_path / "run", cfg, jobs, [2, 5], run_process=run, deadline=1.0
        )

    assert len(completed_sources) == 6
    assert target_launches == []


def test_load_audit_resume_rebinds_legacy_success_and_preserves_attempt_one(
        config, tmp_path, monkeypatch):
    cfg, source, stale_target = _load_audit_scheduler_seam(config, tmp_path, monkeypatch)
    checkpoint, summary, checkpoint_hash, summary_hash = _publish_load_source(source)
    resolved_target = tuning._declare_job_evidence(cfg, stale_target)
    run_dir = tmp_path / "run"

    def successful(_command, log_path, _env):
        _write_formal_target_log(log_path, {**stale_target, "source_checkpoint_sha256": checkpoint_hash})
        return 0

    legacy = execute_parallel_tasks(run_dir, cfg, [resolved_target], [0], run_process=successful)[0]
    state_path = next((run_dir / "state").glob("*.json"))
    attempt_one_log = Path(legacy["log_path"])
    attempt_one_command = Path(legacy["command_path"])
    immutable_log = attempt_one_log.read_bytes()
    immutable_command = attempt_one_command.read_bytes()
    legacy.update(
        evidence_status="unresolved",
        source_checkpoint_sha256=None,
        source_summary_sha256=None,
    )
    legacy["result_identity"].pop("source_checkpoint_sha256")
    state_path.write_text(json.dumps(legacy))
    resumed_launches = []

    def resumed(_command, log_path, _env):
        resumed_launches.append(log_path)
        _write_formal_target_log(log_path, {**stale_target, "source_checkpoint_sha256": checkpoint_hash})
        return 0

    current = execute_parallel_tasks(run_dir, cfg, [stale_target], [0], run_process=resumed)[0]

    assert current["status"] == "succeeded" and current["attempt"] == 2
    assert resumed_launches and resumed_launches[0].name.endswith("attempt-2.log")
    assert current["source_checkpoint_path"] == str(checkpoint)
    assert current["source_checkpoint_sha256"] == checkpoint_hash
    assert current["source_summary_path"] == str(summary)
    assert current["source_summary_sha256"] == summary_hash
    assert current["metrics"]["source_checkpoint_sha256"] == checkpoint_hash
    assert attempt_one_log.read_bytes() == immutable_log
    assert attempt_one_command.read_bytes() == immutable_command


@pytest.mark.parametrize("evidence", ["missing", "invalid-summary"])
def test_load_audit_target_fails_before_launch_when_source_evidence_is_unavailable_or_invalid(
        config, tmp_path, monkeypatch, evidence):
    cfg, source, target = _load_audit_scheduler_seam(config, tmp_path, monkeypatch)
    if evidence == "invalid-summary":
        _publish_load_source(source, target_labels_consumed=True)
    launches = []

    with pytest.raises(ValueError, match="target source evidence"):
        execute_parallel_tasks(
            tmp_path / "run", cfg, [target], [0],
            run_process=lambda *_args: launches.append("target") or 0,
        )

    assert launches == []
    assert not list((tmp_path / "run" / "commands").glob("*.txt"))
    assert not list((tmp_path / "run" / "logs").glob("*.log"))


def test_load_audit_parser_metrics_cannot_authorize_source_checkpoint_identity(
        config, tmp_path, monkeypatch):
    cfg, source, target = _load_audit_scheduler_seam(config, tmp_path, monkeypatch)
    _checkpoint, _summary, checkpoint_hash, _summary_hash = _publish_load_source(source)
    parser_claim = "f" * 64

    def run(_command, log_path, _env):
        _write_formal_target_log(log_path, {**target, "source_checkpoint_sha256": parser_claim})
        return 0

    state = execute_parallel_tasks(tmp_path / "run", cfg, [target], [0], run_process=run)[0]

    assert state["status"] == "failed"
    assert state["source_checkpoint_sha256"] == checkpoint_hash
    assert state["metrics"] is None
    assert "source_checkpoint_sha256" in state["parse_error"]


@pytest.mark.parametrize(
    ("existing", "expected_failure"),
    [
        ({"status": "failed", "attempt": 1, "failure_class": "permanent"}, "permanent"),
        ({"status": "running", "attempt": 2}, "stale_resume_exhausted"),
    ],
)
def test_unlaunched_target_resume_gates_keep_retry_policy_when_source_evidence_is_missing(
        config, tmp_path, monkeypatch, existing, expected_failure):
    cfg, _source, target = _load_audit_scheduler_seam(config, tmp_path, monkeypatch)
    run_dir = tmp_path / "run"
    _cid, stem = tuning._job_identity(cfg, target)
    state_path = run_dir / "state" / f"{stem}.json"
    tuning.atomic_write_json(state_path, {**target, **existing, "marker": "preserved"})

    result = execute_parallel_tasks(
        run_dir, cfg, [target], [0],
        run_process=lambda *_args: pytest.fail("resume gate must not authorize a launch"),
    )[0]

    assert result["status"] == "failed"
    assert result["failure_class"] == expected_failure
    assert result["attempt"] == existing["attempt"]
    assert result["marker"] == "preserved"


def _create_declared_success(run_dir: Path, config, job: dict) -> tuple[dict, list[Path]]:
    calls = []

    def successful(_command, log_path, _env):
        calls.append(log_path)
        _write_formal_target_log(log_path, job)
        return 0

    state = resume_pipeline(run_dir, config, [job], [0], run_process=successful)[0]
    return state, calls


@pytest.mark.parametrize("missing_key", sorted(_DECLARED_TARGET_EVIDENCE_KEYS))
def test_target_resume_rejects_every_declared_evidence_omission_and_launches_next_attempt(
        config, tmp_path, monkeypatch, missing_key):
    cfg, job = _declared_target_job(config, tmp_path, monkeypatch, "beginning")
    run_dir = tmp_path / "run"
    state, calls = _create_declared_success(run_dir, cfg, job)
    state_path = next((run_dir / "state").glob("*.json"))
    attempt_one_log = calls[0]
    attempt_one_command = Path(state["command_path"])
    immutable_log = attempt_one_log.read_bytes()
    immutable_command = attempt_one_command.read_bytes()
    state.pop(missing_key)
    state_path.write_text(json.dumps(state))

    assert not tuning._valid_success(state, state["command"], job["artifacts"], job)
    resumed, resumed_calls = _create_declared_success(run_dir, cfg, job)

    assert resumed["attempt"] == 2
    assert resumed_calls[0].name.endswith("attempt-2.log")
    assert attempt_one_log.read_bytes() == immutable_log
    assert attempt_one_command.read_bytes() == immutable_command
    _assert_declared_target_evidence([resumed], job)


@pytest.mark.parametrize(
    ("changed_key", "changed_value"),
    [
        ("kind", "source"), ("stage", "legacy"), ("route", "dtcc_ordinary"),
        ("variant", "ordinary"), ("source", 2), ("target", 2), ("task", [1, 2]),
        ("source_seed", 2026), ("stream_seed", 2026),
        ("overrides", {"Opt.lr_tar": 0.008}), ("candidate_id", "other-candidate"),
        ("config_sha256", "b" * 64), ("beginning_only", False),
        ("load_split", True), ("expected_result_kind", "target"),
        ("expected_result_contract", "target"), ("evidence_status", "unresolved"),
        ("artifacts", []), ("cache_manifest_path", "other-manifest.json"),
        ("cache_manifest_sha256", "b" * 64), ("runner_script_path", "other-runner.py"),
        ("runner_script_sha256", "b" * 64), ("experiment_config_path", "other-config.yaml"),
        ("experiment_config_sha256", "b" * 64),
        ("source_checkpoint_path", "other-checkpoint.pt"),
        ("source_checkpoint_sha256", "b" * 64),
        ("source_summary_path", "other-summary.json"),
        ("source_summary_sha256", "b" * 64),
    ],
)
def test_target_resume_rejects_changed_planned_identity_path_or_hash(
        config, tmp_path, monkeypatch, changed_key, changed_value):
    cfg, job = _declared_target_job(config, tmp_path, monkeypatch, "beginning")
    run_dir = tmp_path / "run"
    state, calls = _create_declared_success(run_dir, cfg, job)
    state_path = next((run_dir / "state").glob("*.json"))
    attempt_one_log = calls[0]
    attempt_one_command = Path(state["command_path"])
    immutable_log = attempt_one_log.read_bytes()
    immutable_command = attempt_one_command.read_bytes()
    state[changed_key] = changed_value
    state_path.write_text(json.dumps(state))

    assert not tuning._valid_success(state, state["command"], job["artifacts"], job)
    resumed, resumed_calls = _create_declared_success(run_dir, cfg, job)

    assert resumed["attempt"] == 2
    assert resumed_calls[0].name.endswith("attempt-2.log")
    assert attempt_one_log.read_bytes() == immutable_log
    assert attempt_one_command.read_bytes() == immutable_command
    _assert_declared_target_evidence([resumed], job)


def test_complete_declared_target_success_skips_byte_for_byte(config, tmp_path, monkeypatch):
    cfg, job = _declared_target_job(config, tmp_path, monkeypatch, "beginning")
    run_dir = tmp_path / "run"
    state, calls = _create_declared_success(run_dir, cfg, job)
    state_path = next((run_dir / "state").glob("*.json"))
    command_path = Path(state["command_path"])
    before = (state_path.read_bytes(), calls[0].read_bytes(), command_path.read_bytes())

    resumed = resume_pipeline(
        run_dir, cfg, [job], [0],
        run_process=lambda *_args: pytest.fail("complete success must not relaunch"),
    )[0]

    assert resumed == state
    assert (state_path.read_bytes(), calls[0].read_bytes(), command_path.read_bytes()) == before
    assert not list((run_dir / "logs").glob("*.attempt-2.log"))


@pytest.mark.parametrize("outcome", ["success", "transient", "permanent"])
def test_declared_target_jobs_cannot_spoof_runtime_or_parser_state(config, tmp_path, monkeypatch, outcome):
    cfg, job = _declared_target_job(config, tmp_path, monkeypatch, "baseline")
    job.update(_JOB_RUNTIME_SPOOFS)
    writes = _capture_atomic_states(monkeypatch)
    calls = 0

    def run(_command, log_path, _env):
        nonlocal calls
        calls += 1
        if outcome == "permanent":
            log_path.write_text("protocol assertion failed\n")
            return 2
        if outcome == "transient" and calls == 1:
            log_path.write_text("CUDA initialization error\n")
            return 1
        _write_formal_target_log(log_path, job)
        return 0

    result = resume_pipeline(tmp_path / f"run-{outcome}", cfg, [job], [0], run_process=run)[0]
    for state in writes:
        for key, spoof in _JOB_RUNTIME_SPOOFS.items():
            assert state.get(key, object()) != spoof, f"{key} survived in {state['status']} state"
    if outcome == "permanent":
        assert result["status"] == "failed" and result["failure_class"] == "permanent"
    else:
        assert result["status"] == "succeeded"
        for key, value in result["metrics"].items():
            assert result[key] == value


@pytest.mark.parametrize("stage", ["beginning", "baseline"])
def test_declared_target_success_states_preserve_evidence_and_pass_formal_report(config, tmp_path, monkeypatch, stage):
    cfg, job = _declared_target_job(config, tmp_path, monkeypatch, stage)
    job.update(status="succeeded", attempt=99, returncode=0, metrics={"spoofed": True},
               command=["spoofed"], log_path="spoofed.log")
    writes = _capture_atomic_states(monkeypatch)

    def successful(_command, log_path, _env):
        _write_formal_target_log(log_path, job)
        return 0

    result = resume_pipeline(tmp_path / "run", cfg, [job], [0], run_process=successful)[0]
    assert [state["status"] for state in writes] == ["running", "succeeded"]
    assert writes[0]["attempt"] == 1 and writes[0]["returncode"] is None and writes[0]["metrics"] is None
    assert writes[0]["command"] != ["spoofed"] and writes[0]["log_path"] != "spoofed.log"
    _assert_declared_target_evidence(writes, job)
    formal = {**result, "_formal_state": True}
    write_reports(tmp_path / f"report-{stage}", [formal], frozen=None, checkpoint_manifest={})


def test_declared_target_retry_and_failure_states_preserve_evidence_on_every_attempt(config, tmp_path, monkeypatch):
    cfg, job = _declared_target_job(config, tmp_path, monkeypatch, "baseline")
    writes = _capture_atomic_states(monkeypatch)
    calls = []

    def transient_then_success(_command, log_path, _env):
        calls.append(log_path)
        if len(calls) == 1:
            log_path.write_text("CUDA initialization error\n")
            return 1
        _write_formal_target_log(log_path, job)
        return 0

    result = resume_pipeline(tmp_path / "transient", cfg, [job], [0], run_process=transient_then_success)[0]
    assert [state["status"] for state in writes] == ["running", "running", "succeeded"]
    assert [state["attempt"] for state in writes] == [1, 2, 2]
    _assert_declared_target_evidence(writes, job)
    assert calls[0].read_text() == "CUDA initialization error\n"
    write_reports(tmp_path / "report-transient", [{**result, "_formal_state": True}], frozen=None, checkpoint_manifest={})

    writes.clear()
    failed_logs = []

    def permanent(_command, log_path, _env):
        failed_logs.append(log_path)
        log_path.write_text("protocol assertion failed\n")
        return 2

    first = resume_pipeline(tmp_path / "permanent", cfg, [job], [0], run_process=permanent)[0]
    second = resume_pipeline(tmp_path / "permanent", cfg, [job], [0], run_process=permanent, retry_failed=True)[0]
    assert first["status"] == second["status"] == "failed"
    assert [state["status"] for state in writes] == ["running", "failed", "running", "failed"]
    assert [state["attempt"] for state in writes] == [1, 1, 2, 2]
    _assert_declared_target_evidence(writes, job)
    assert failed_logs[0].read_text() == "protocol assertion failed\n"


def test_parallel_scheduler_isolates_one_job_per_gpu_and_writes_atomic_success(config, tmp_path):
    seen = []

    def fake_run(command, log_path, env):
        seen.append((env["CUDA_VISIBLE_DEVICES"], list(command)))
        _successful_log(log_path)
        return 0

    jobs = [
        {"route": "0711_robust", "overrides": {}, "task": (0, 1), "stream_seed": 2025},
        {"route": "0711_robust", "overrides": {}, "task": (1, 2), "stream_seed": 2025},
    ]
    records = execute_parallel_tasks(tmp_path, config, jobs, [2, 5], run_process=fake_run)
    assert {gpu for gpu, _ in seen} == {"2", "5"}
    assert all(record["status"] == "succeeded" for record in records)
    assert len(list((tmp_path / "state").glob("*.json"))) == 2
    assert not list((tmp_path / "state").glob("*.tmp"))


def _dynamic_scheduler_jobs(count: int) -> list[dict]:
    tasks = ((0, 1), (1, 2), (2, 3), (3, 0))
    return [
        {
            "route": "0711_robust",
            "overrides": {"Opt.lr_tar": 0.01 + index / 10_000},
            "task": tasks[index % len(tasks)],
            "stream_seed": 2025,
        }
        for index in range(count)
    ]


def test_dynamic_gpu_scheduler_idle_gpu_steals_all_work_from_busy_gpu(
        config, tmp_path, monkeypatch):
    jobs = _dynamic_scheduler_jobs(4)
    launched = []

    def idle(gpu, *_args):
        return gpu == 0

    def unexpected_static_bucket_wait(_seconds):
        raise AssertionError("busy GPU must not own pending work while GPU 0 is idle")

    def run(_command, log_path, env):
        launched.append((log_path.name, env["CUDA_VISIBLE_DEVICES"]))
        _successful_log(log_path)
        return 0

    monkeypatch.setattr(tuning, "_gpu_is_idle", idle)
    monkeypatch.setattr(tuning.time, "sleep", unexpected_static_bucket_wait)
    monkeypatch.setattr(tuning, "_default_run", run)

    records = execute_parallel_tasks(tmp_path, config, jobs, [0, 1])

    assert [gpu for _stem, gpu in launched] == ["0"] * len(jobs)
    assert len({stem for stem, _gpu in launched}) == len(jobs)
    assert [record["task"] for record in records] == [list(job["task"]) for job in jobs]


def test_dynamic_gpu_scheduler_runs_exactly_once_with_one_concurrent_job_per_gpu(
        config, tmp_path, monkeypatch):
    jobs = _dynamic_scheduler_jobs(6)
    first_wave = threading.Barrier(2)
    lock = threading.Lock()
    launched = []
    active_by_gpu = {0: 0, 1: 0}
    peak_by_gpu = {0: 0, 1: 0}
    active_total = 0
    peak_total = 0

    def run(_command, log_path, env):
        nonlocal active_total, peak_total
        gpu = int(env["CUDA_VISIBLE_DEVICES"])
        with lock:
            launched.append(log_path.name)
            launch_number = len(launched)
            active_by_gpu[gpu] += 1
            active_total += 1
            peak_by_gpu[gpu] = max(peak_by_gpu[gpu], active_by_gpu[gpu])
            peak_total = max(peak_total, active_total)
        try:
            if launch_number <= 2:
                first_wave.wait(timeout=1.0)
            _successful_log(log_path)
            return 0
        finally:
            with lock:
                active_by_gpu[gpu] -= 1
                active_total -= 1

    monkeypatch.setattr(tuning, "_gpu_is_idle", lambda *_args: True)
    monkeypatch.setattr(tuning, "_default_run", run)

    records = execute_parallel_tasks(tmp_path, config, jobs, [0, 1])

    assert len(launched) == len(set(launched)) == len(jobs)
    assert peak_by_gpu == {0: 1, 1: 1}
    assert peak_total == 2
    assert all(record["status"] == "succeeded" for record in records)


def test_dynamic_gpu_scheduler_recovered_gpu_joins_remaining_work(
        config, tmp_path, monkeypatch):
    jobs = _dynamic_scheduler_jobs(6)
    lock = threading.Lock()
    busy_probe_seen = threading.Event()
    recovered = threading.Event()
    gpu_one_joined = threading.Event()
    probes = {0: 0, 1: 0}
    launched = []

    def idle(gpu, *_args):
        with lock:
            probes[gpu] += 1
            probe = probes[gpu]
        if gpu == 0:
            return probe == 1 or gpu_one_joined.is_set()
        if probe == 1:
            busy_probe_seen.set()
            return False
        return recovered.is_set()

    def run(_command, log_path, env):
        gpu = int(env["CUDA_VISIBLE_DEVICES"])
        launched.append(gpu)
        if gpu == 0 and not recovered.is_set():
            assert busy_probe_seen.wait(timeout=0.5)
            recovered.set()
        if gpu == 1:
            gpu_one_joined.set()
        _successful_log(log_path)
        return 0

    monkeypatch.setattr(tuning, "_gpu_is_idle", idle)
    monkeypatch.setattr(tuning.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(tuning, "_default_run", run)

    records = execute_parallel_tasks(tmp_path, config, jobs, [0, 1])

    assert busy_probe_seen.is_set() and gpu_one_joined.is_set()
    assert len(launched) == len(jobs)
    assert all(record["status"] == "succeeded" for record in records)


def test_dynamic_gpu_scheduler_reserve_stops_jobs_not_yet_launched(
        config, tmp_path, monkeypatch):
    jobs = _dynamic_scheduler_jobs(2)
    clock = [0.0]
    launched = []
    monkeypatch.setattr(tuning.time, "time", lambda: clock[0])

    def run(_command, log_path, _env):
        launched.append(log_path.name)
        clock[0] = 300.0
        _successful_log(log_path)
        return 0

    with pytest.raises(TimeoutError, match="reserve"):
        execute_parallel_tasks(
            tmp_path, config, jobs, [0], run_process=run, deadline=3_000.0
        )

    assert len(launched) == 1


def test_dynamic_gpu_scheduler_resume_skips_fifteen_and_runs_remainder_on_idle_gpu(
        config, tmp_path, monkeypatch):
    jobs = _dynamic_scheduler_jobs(16)
    run_dir = tmp_path / "run"

    def seed_successes(_command, log_path, _env):
        _successful_log(log_path)
        return 0

    initial = execute_parallel_tasks(
        run_dir, config, jobs[:15], [0], run_process=seed_successes
    )
    assert len(initial) == 15 and all(record["attempt"] == 1 for record in initial)

    launched = []

    def idle(gpu, *_args):
        return gpu == 0

    def unexpected_static_bucket_wait(_seconds):
        raise AssertionError("resumed work must not remain bound to busy GPU 1")

    def run_remainder(_command, log_path, env):
        launched.append((log_path.name, env["CUDA_VISIBLE_DEVICES"]))
        _successful_log(log_path)
        return 0

    monkeypatch.setattr(tuning, "_gpu_is_idle", idle)
    monkeypatch.setattr(tuning.time, "sleep", unexpected_static_bucket_wait)
    monkeypatch.setattr(tuning, "_default_run", run_remainder)

    resumed = execute_parallel_tasks(run_dir, config, jobs, [0, 1])

    assert len(launched) == 1 and launched[0][1] == "0"
    assert all(record["attempt"] == 1 for record in resumed)
    assert len(list((run_dir / "commands").glob("*.attempt-1.txt"))) == 16
    assert not list((run_dir / "commands").glob("*.attempt-2.txt"))


class _SchedulerWorkerError(RuntimeError):
    pass


def test_dynamic_gpu_scheduler_probe_failure_stops_pending_before_another_gpu_claims(
        config, tmp_path, monkeypatch):
    jobs = _dynamic_scheduler_jobs(3)
    probes_concurrent = threading.Barrier(2)
    launched = []

    def idle(gpu, *_args):
        probes_concurrent.wait(timeout=1.0)
        if gpu == 0:
            raise _SchedulerWorkerError("nvidia-smi probe failed")
        return True

    def run(_command, log_path, env):
        launched.append((log_path.name, env["CUDA_VISIBLE_DEVICES"]))
        _successful_log(log_path)
        return 0

    monkeypatch.setattr(tuning, "_gpu_is_idle", idle)
    monkeypatch.setattr(tuning, "_default_run", run)

    with pytest.raises(_SchedulerWorkerError, match="probe failed"):
        execute_parallel_tasks(tmp_path, config, jobs, [0, 1])

    assert launched == []
    assert not list((tmp_path / "commands").glob("*.txt"))


def test_dynamic_gpu_scheduler_waits_for_inflight_job_after_queue_empty_failure(
        config, tmp_path, monkeypatch):
    jobs = _dynamic_scheduler_jobs(2)
    both_inflight = threading.Barrier(2)
    failure_seen = threading.Event()
    surviving_job_completed = threading.Event()
    launched = []

    def run(_command, log_path, env):
        gpu = int(env["CUDA_VISIBLE_DEVICES"])
        launched.append(gpu)
        both_inflight.wait(timeout=1.0)
        if gpu == 0:
            failure_seen.set()
            raise _SchedulerWorkerError("inflight execution failed")
        assert failure_seen.wait(timeout=0.5)
        _successful_log(log_path)
        surviving_job_completed.set()
        return 0

    monkeypatch.setattr(tuning, "_gpu_is_idle", lambda *_args: True)
    monkeypatch.setattr(tuning, "_default_run", run)

    with pytest.raises(_SchedulerWorkerError, match="execution failed"):
        execute_parallel_tasks(tmp_path, config, jobs, [0, 1])

    assert set(launched) == {0, 1}
    assert surviving_job_completed.is_set()


def test_dynamic_gpu_scheduler_single_gpu_probe_failure_does_not_deadlock(
        config, tmp_path, monkeypatch):
    def failed_probe(*_args):
        raise _SchedulerWorkerError("single GPU probe failed")

    monkeypatch.setattr(tuning, "_gpu_is_idle", failed_probe)
    monkeypatch.setattr(
        tuning, "_default_run", lambda *_args: pytest.fail("job launched after probe failure")
    )

    with pytest.raises(_SchedulerWorkerError, match="single GPU probe failed"):
        execute_parallel_tasks(tmp_path, config, _dynamic_scheduler_jobs(1), [0])

    assert not list((tmp_path / "commands").glob("*.txt"))


def test_scheduler_deduplicates_gpu_ids_and_checks_deadline_before_launch(config, tmp_path):
    calls = []
    jobs = [{"route": "0711_robust", "overrides": {}, "task": task, "stream_seed": 2025} for task in ((0, 1), (1, 2))]

    def fake_run(_command, log_path, env):
        calls.append(env["CUDA_VISIBLE_DEVICES"])
        _successful_log(log_path)
        return 0

    execute_parallel_tasks(tmp_path / "dedupe", config, jobs, [2, 2], run_process=fake_run)
    assert calls == ["2", "2"]
    with pytest.raises(TimeoutError, match="reserve"):
        execute_parallel_tasks(tmp_path / "deadline", config, jobs, [2], run_process=lambda *_: pytest.fail("launched past deadline"), deadline=0)


def test_formal_scheduler_rechecks_gpu_immediately_before_launch(config, tmp_path, monkeypatch):
    probes = []

    def idle(gpu, *_args):
        probes.append(gpu)
        return True

    def fake_default(_command, log_path, _env):
        _successful_log(log_path)
        return 0

    monkeypatch.setattr(tuning, "_gpu_is_idle", idle)
    monkeypatch.setattr(tuning, "_default_run", fake_default)
    jobs = [{"route": "0711_robust", "overrides": {}, "task": (0, 1), "stream_seed": 2025}]
    execute_parallel_tasks(tmp_path, config, jobs, [3])
    assert probes == [3]


def test_resume_skips_only_hash_validated_artifacts(config, tmp_path):
    calls = []
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"checkpoint-v1")

    def fake_run(command, log_path, env):
        calls.append(log_path)
        _successful_log(log_path)
        return 0

    jobs = [{"route": "0711_robust", "overrides": {}, "task": (0, 1), "stream_seed": 2025, "artifacts": [str(checkpoint)]}]
    resume_pipeline(tmp_path, config, jobs, [0], run_process=fake_run)
    resume_pipeline(tmp_path, config, jobs, [0], run_process=fake_run)
    assert len(calls) == 1
    checkpoint.write_bytes(b"checkpoint-v2")
    resume_pipeline(tmp_path, config, jobs, [0], run_process=fake_run)
    assert len(calls) == 2


def test_resume_keeps_attempt_logs_and_does_not_retry_permanent_without_authorization(config, tmp_path):
    calls = []

    def permanent(_command, log_path, _env):
        calls.append(log_path)
        log_path.write_text("protocol assertion failed\n")
        return 2

    jobs = [{"route": "0711_robust", "overrides": {}, "task": (0, 1), "stream_seed": 2025}]
    first = resume_pipeline(tmp_path, config, jobs, [0], run_process=permanent)
    second = resume_pipeline(tmp_path, config, jobs, [0], run_process=permanent)
    assert first[0]["status"] == second[0]["status"] == "failed" and len(calls) == 1
    assert calls[0].name.endswith("attempt-1.log") and calls[0].read_text() == "protocol assertion failed\n"
    resume_pipeline(tmp_path, config, jobs, [0], run_process=permanent, retry_failed=True)
    assert len(calls) == 2 and calls[1].name.endswith("attempt-2.log")
    assert calls[0].read_text() == "protocol assertion failed\n"


def test_transient_cuda_retries_once_into_new_immutable_log(config, tmp_path):
    calls = []

    def transient_then_success(_command, log_path, _env):
        calls.append(log_path)
        if len(calls) == 1:
            log_path.write_text("CUDA initialization error\n")
            return 1
        _successful_log(log_path)
        return 0

    jobs = [{"route": "0711_robust", "overrides": {}, "task": (0, 1), "stream_seed": 2025}]
    result = resume_pipeline(tmp_path, config, jobs, [0], run_process=transient_then_success)
    assert result[0]["status"] == "succeeded"
    assert [path.name[-13:] for path in calls] == ["attempt-1.log", "attempt-2.log"]
    assert "CUDA initialization error" in calls[0].read_text()


def test_exhausted_transient_is_terminal_until_explicit_retry(config, tmp_path):
    calls = []

    def transient(_command, log_path, _env):
        calls.append(log_path)
        log_path.write_text("CUDA initialization error\n")
        return 1

    jobs = [{"route": "0711_robust", "overrides": {}, "task": (0, 1), "stream_seed": 2025}]
    first = resume_pipeline(tmp_path, config, jobs, [0], run_process=transient)
    second = resume_pipeline(tmp_path, config, jobs, [0], run_process=transient)
    assert first[0]["failure_class"] == second[0]["failure_class"] == "transient"
    assert len(calls) == 2
    resume_pipeline(tmp_path, config, jobs, [0], run_process=transient, retry_failed=True)
    assert len(calls) == 4
    assert [path.name for path in calls] == [
        calls[0].name.replace("attempt-1", "attempt-1"),
        calls[0].name.replace("attempt-1", "attempt-2"),
        calls[0].name.replace("attempt-1", "attempt-3"),
        calls[0].name.replace("attempt-1", "attempt-4"),
    ]


def test_source_resume_validates_immutable_attempt_and_all_bound_evidence(config, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache_contract = _install_fake_v2_cache(monkeypatch, cache, split="bearing")
    root = tmp_path / "checkpoints"
    checkpoint_dir = root / "ordinary" / "source_0" / "seed_2025"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "best_source_ResNet18_1D_SDE2025fft_Linear.pt"
    summary = checkpoint_dir / "source_training_summary.json"
    checkpoint.write_bytes(b"checkpoint-v1")
    summary.write_text(json.dumps({
        "route": "ordinary", "source": 0, "seed": 2025,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "target_labels_consumed": False, "best_accuracy": 77.0,
        "cache_identity": {
            "mode": "formal",
            "manifest_path": str((cache / "manifest.json").resolve()),
            "manifest_sha256": cache_contract["manifest_sha256"],
            "content_sha256": cache_contract["content_sha256"],
            "tensor_sha256s": {
                str((cache / row["tensor_file"]).resolve()): row["tensor_sha256"]
                for row in cache_contract["domains"].values()
            },
        },
    }))
    cfg = json.loads(json.dumps(config))
    cfg["protocol"]["data_path"] = str(cache)
    cfg["checkpoints"]["root"] = str(root)
    runner = tmp_path / "source_runner.py"
    runner.write_text("# runner v1\n")
    cfg["routes"]["source_ordinary"] = str(runner)
    monkeypatch.setattr(tuning, "validate_config", lambda _config: None)
    job = plan_stage_jobs(cfg, "source", tmp_path)[0]
    calls = []

    def successful(_command, log_path, _env):
        calls.append(log_path)
        log_path.write_text("source complete\n")
        return 0

    resume_pipeline(tmp_path, cfg, [job], [0], run_process=successful)
    resume_pipeline(tmp_path, cfg, [job], [0], run_process=successful)
    assert len(calls) == 1
    state_path = next((tmp_path / "state").glob("*.json"))
    state = json.loads(state_path.read_text())
    command_path = Path(state["command_path"])
    assert command_path.is_file() and state["command_log_sha256"] == hashlib.sha256(command_path.read_bytes()).hexdigest()

    # Each mutation invalidates resume and creates a new immutable attempt.
    calls[0].write_text("tampered source log\n")
    resume_pipeline(tmp_path, cfg, [job], [0], run_process=successful)
    state = json.loads(state_path.read_text())
    Path(state["command_path"]).write_text("tampered command\n")
    resume_pipeline(tmp_path, cfg, [job], [0], run_process=successful)
    state = json.loads(state_path.read_text())
    state["metrics"]["best_accuracy"] = 99.0
    state_path.write_text(json.dumps(state))
    resume_pipeline(tmp_path, cfg, [job], [0], run_process=successful)
    runner.write_text("# runner v2\n")
    job = plan_stage_jobs(cfg, "source", tmp_path)[0]
    resume_pipeline(tmp_path, cfg, [job], [0], run_process=successful)
    checkpoint.write_bytes(b"checkpoint-v2")
    summary.write_text(json.dumps({**json.loads(summary.read_text()), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}))
    job = plan_stage_jobs(cfg, "source", tmp_path)[0]
    resume_pipeline(tmp_path, cfg, [job], [0], run_process=successful)
    assert len(calls) == 6
    assert len({path.name for path in calls}) == 6
    state = json.loads(state_path.read_text())
    state["_formal_state"] = True
    write_reports(tmp_path / "source-report", [state], frozen=None,
                  checkpoint_manifest={state["candidate_id"]: state["output_artifact_hashes"]})
    bad_state = json.loads(json.dumps(state))
    bad_state["metrics"]["best_accuracy"] = 12.0
    with pytest.raises(ValueError, match="source evidence"):
        write_reports(tmp_path / "source-report-tamper", [bad_state], frozen=None,
                      checkpoint_manifest={state["candidate_id"]: state["output_artifact_hashes"]})


def test_runner_script_is_part_of_every_job_input_contract(config, tmp_path):
    job = plan_stage_jobs(config, "baseline", tmp_path, resolve_artifacts=False)[0]
    assert job["runner_script_path"].endswith("main_tta_dtcc_hust_strict.py")
    assert job["experiment_config_path"].endswith("HUST0711_strict_tuning.yaml")
    assert job["runner_script_path"] in job["artifacts"]
    assert job["experiment_config_path"] in job["artifacts"]


def test_route_aware_parser_and_reports_keep_conclusions_separate(tmp_path):
    log = tmp_path / "route.log"
    _successful_log(log)
    parsed = parse_runner_log(log, "0711_robust")
    assert parsed["strict_online"] == 61.25
    assert parsed["macro_f1"] == 58.0 and parsed["memory_class_coverage"] == 7
    rows = []
    for route, score in [("dtcc_ordinary", 70.0), ("0711_robust_baseline", 60.0), ("0711_robust", 61.0), ("0711_common", 55.0)]:
        for source in range(4):
            for target in range(4):
                if source != target:
                    rows.append({"route": route, "candidate_id": "frozen" if route == "0711_robust" else "base", "task": [source, target], "stream_seed": 2025, "status": "succeeded", "strict_online": score, "macro_f1": score - 1})
    frozen = {"candidate_id": "frozen", "overrides": {}}
    outputs = write_reports(tmp_path, rows, frozen=frozen, checkpoint_manifest={"ordinary": "b" * 64})
    assert outputs["recommended"] is True
    assert outputs["beats_dtcc"] is False
    assert "Full-pipeline" in (tmp_path / "report.md").read_text()
    assert (tmp_path / "common_source_12task.csv").is_file()
    with (tmp_path / "full_pipeline_12task.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 36


def test_beginning_rows_are_excluded_from_leaderboard_without_losing_audit(tmp_path):
    rows = [_synthetic_beginning_row("robust", (source, target), 40.0)
            for source in range(4) for target in range(4) if source != target]
    write_reports(tmp_path, rows, frozen=None, checkpoint_manifest={})
    assert list(csv.DictReader((tmp_path / "leaderboard.csv").open())) == []
    assert len(list(csv.DictReader((tmp_path / "beginning_audit.csv").open()))) == 12


def test_aggregates_reject_duplicate_mixed_or_partial_task_sets(tmp_path):
    base = [{"route": "dtcc_ordinary", "variant": "ordinary", "candidate_id": "dtcc_ordinary",
             "task": [s, t], "stream_seed": 2025, "stage": "baseline", "result_kind": "target",
             "status": "succeeded", "strict_online": 60.0, "macro_f1": 50.0, "before": 40.0}
            for s in range(4) for t in range(4) if s != t]
    with pytest.raises(ValueError, match="duplicate"):
        write_reports(tmp_path / "duplicate", [*base, dict(base[0])], frozen=None, checkpoint_manifest={})
    with pytest.raises(ValueError, match="mixed|stage"):
        write_reports(tmp_path / "mixed", [*base[:-1], {**base[-1], "stage": "heldout"}], frozen=None, checkpoint_manifest={})


def test_shared_machine_result_record_parses_and_cross_checks_identity(tmp_path):
    confusion = [[5 if row == column else 0 for column in range(7)] for row in range(7)]
    teacher_confusion = [[0 for _column in range(7)] for _row in range(7)]
    for class_id in range(7):
        teacher_confusion[class_id][(class_id + 1) % 7] = 5
    result = build_hust_result_record(
        result_kind="target", route="0711_robust", variant="robust", task=(0, 1),
        source_checkpoint_sha256="a" * 64, config_sha256="b" * 64,
        candidate_id="candidate123", source_seed=2025, stream_seed=2026,
        beginning=42.0, strict_online=100.0, post_stream=72.0,
        confusion_matrix=confusion, samples=35, batches=4, passes=1,
        finite_losses=True, trainable_parameters=["backbone.band_scale", "backbone.band_bias", "backbone.warp_ctrl"],
        pre_update_scoring=True, metadata_evidence_used=True,
        runtime_seconds=1.25, peak_memory_mb=100.0,
        memory_size=20, memory_class_coverage=7, certain_ratio=0.6,
        uncertain_ratio=0.4, offline_purity=0.0, certain_purity=95.0,
        evidence_applicable=True, evidence_active_ratio=0.5, mean_batch_ms=312.5,
        offline_confusion_matrix=teacher_confusion,
        post_macro_precision=70.0, post_macro_recall=71.0, post_macro_f1=69.0,
    )
    log = tmp_path / "machine.log"
    log.write_text("noise\nHUST_RESULT_JSON=" + json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    parsed = parse_runner_log(log, "0711_robust", expected={
        "candidate_id": "candidate123", "task": [0, 1], "stream_seed": 2026,
        "config_sha256": "b" * 64, "source_checkpoint_sha256": "a" * 64,
    })
    assert parsed["strict_online"] == 100.0 and parsed["macro_f1"] == 100.0
    assert parsed["confusion_matrix"] == confusion
    assert parsed["offline_confusion_matrix"] == teacher_confusion
    assert parsed["post_macro_precision"] == 70.0
    with pytest.raises(ValueError, match="identity"):
        parse_runner_log(log, "0711_robust", expected={"stream_seed": 2025})


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("memory_size", None), ("memory_class_coverage", 8),
        ("certain_ratio", 1.1), ("uncertain_ratio", -0.1),
        ("offline_purity", 101.0), ("certain_purity", -1.0),
        ("runtime_seconds", -1.0), ("peak_memory_mb", -1.0),
        ("evidence_active_ratio", 1.2),
    ],
)
def test_formal_parser_rejects_invalid_mandatory_target_diagnostics(tmp_path, field, bad_value):
    confusion = [[5 if row == column else 0 for column in range(7)] for row in range(7)]
    result = build_hust_result_record(
        result_kind="target", route="0711_robust", variant="robust", task=(0, 1),
        source_checkpoint_sha256="a" * 64, config_sha256="b" * 64,
        candidate_id="candidate123", source_seed=2025, stream_seed=2025,
        beginning=42.0, strict_online=100.0, post_stream=72.0,
        confusion_matrix=confusion, samples=35, batches=4, passes=1,
        finite_losses=True, trainable_parameters=["backbone.band_scale"],
        pre_update_scoring=True, metadata_evidence_used=True,
        runtime_seconds=1.25, peak_memory_mb=100.0,
        memory_size=20, memory_class_coverage=7, certain_ratio=0.6,
        uncertain_ratio=0.4, offline_purity=100.0, certain_purity=95.0,
        evidence_applicable=True, evidence_active_ratio=0.5, mean_batch_ms=312.5,
        offline_confusion_matrix=confusion,
        post_macro_precision=70.0, post_macro_recall=71.0, post_macro_f1=69.0,
    )
    result[field] = bad_value
    log = tmp_path / f"bad-{field}.log"
    log.write_text("HUST_RESULT_JSON=" + json.dumps(result, separators=(",", ":")) + "\n")
    with pytest.raises(ValueError, match="formal HUST"):
        parse_runner_log(log, "0711_robust", expected={"candidate_id": "candidate123"})


def test_formal_parser_rejects_missing_machine_result(tmp_path):
    log = tmp_path / "legacy.log"
    _successful_log(log)
    with pytest.raises(ValueError, match="HUST_RESULT_JSON"):
        parse_runner_log(log, "0711_robust", expected={"candidate_id": "formal"})


def test_formal_beginning_record_uses_smaller_schema_and_parses(tmp_path):
    result = build_hust_result_record(
        result_kind="beginning", route="dtcc_ordinary", variant="ordinary", task=(0, 1),
        source_checkpoint_sha256="a" * 64, config_sha256="b" * 64,
        candidate_id="beginning-ordinary", source_seed=2025, stream_seed=2025,
        beginning=42.0, strict_online=None, post_stream=None, confusion_matrix=None,
        samples=140, batches=0, passes=0, finite_losses=True, trainable_parameters=[],
        pre_update_scoring=True, metadata_evidence_used=False, runtime_seconds=0.5,
        peak_memory_mb=10.0,
    )
    log = tmp_path / "beginning.log"
    log.write_text("HUST_RESULT_JSON=" + json.dumps(result, separators=(",", ":")) + "\n")
    parsed = parse_runner_log(log, "dtcc_ordinary", expected={"result_kind": "beginning"})
    assert parsed["before"] == 42.0 and parsed["strict_online"] is None


def test_summary_cli_completes_with_beginning_state_rows(tmp_path, monkeypatch):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    state, _log = _formal_beginning_state(tmp_path, monkeypatch)
    state.pop("_formal_state")
    state_dir.joinpath("beginning.json").write_text(json.dumps(state))
    assert summarize_main(["--run-dir", str(tmp_path)]) == 0
    report = (tmp_path / "report.md").read_text()
    assert "DRAFT" in report and "Beginning soft audit" in report


def test_summary_cli_write_final_flag_generates_report(tmp_path, monkeypatch):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    state, _log = _formal_beginning_state(tmp_path, monkeypatch)
    state.pop("_formal_state")
    state_dir.joinpath("beginning.json").write_text(json.dumps(state))

    with pytest.raises(ValueError, match="final inventory"):
        summarize_main(["--run-dir", str(tmp_path), "--write-final"])
    assert not (tmp_path / "report.md").exists()


def _synthetic_beginning_row(variant: str, task: tuple[int, int], before: float) -> dict:
    route = "0711_robust" if variant == "robust" else "dtcc_ordinary"
    return {
        "kind": "target", "route": route, "variant": variant,
        "candidate_id": f"beginning-{variant}", "stage": "beginning",
        "load_split": False, "schema_version": 1, "result_kind": "beginning",
        "task": list(task), "source": task[0], "target": task[1],
        "source_seed": 2025, "stream_seed": 2025, "status": "succeeded",
        "beginning": before, "before": before, "samples": 140, "batches": 0,
        "passes": 0, "strict_online": None, "post_stream": None,
        "macro_precision": None, "macro_recall": None, "macro_f1": None,
        "confusion_matrix": None, "class_coverage": None, "finite_losses": True,
    }


def test_state_records_and_report_exclude_failed_spoofed_beginning_row(config, tmp_path, monkeypatch):
    cfg, job = _declared_target_job(config, tmp_path, monkeypatch, "beginning")
    run_dir = tmp_path / "run"

    def successful(_command, log_path, _env):
        _write_formal_target_log(log_path, job)
        return 0

    resume_pipeline(run_dir, cfg, [job], [0], run_process=successful)

    failed_dir = tmp_path / "failed"

    def permanent(_command, log_path, _env):
        log_path.write_text("protocol assertion failed\n")
        return 2

    resume_pipeline(failed_dir, cfg, [job], [0], run_process=permanent)
    failed = json.loads(next((failed_dir / "state").glob("*.json")).read_text())
    failed.update(task=[1, 2], source=1, target=2, before=99.0, result_kind="beginning",
                  beginning=99.0, samples=140, batches=0, passes=0, strict_online=None,
                  post_stream=None)
    (run_dir / "state" / "failed-spoof.json").write_text(json.dumps(failed))

    records = tuning._state_records(run_dir)
    assert len(records) == 2 and any(row.get("before") == 99.0 for row in records if row["status"] == "failed")
    report_dir = tmp_path / "report"
    write_reports(report_dir, records, frozen=None, checkpoint_manifest={})
    beginning_rows = list(csv.DictReader((report_dir / "beginning_audit.csv").open()))
    assert len(beginning_rows) == 1
    assert beginning_rows[0]["status"] == "succeeded"
    assert json.loads(beginning_rows[0]["task"]) == [0, 1]
    assert json.loads((report_dir / "report_summary.json").read_text())["beginning_violations"] == {
        "ordinary": {}, "robust": {}
    }


@pytest.mark.parametrize("stage", ["beginning", "baseline"])
def test_machine_result_control_extras_are_rejected_before_persisted_state_or_report(config, tmp_path, monkeypatch, stage):
    cfg, job = _declared_target_job(config, tmp_path, monkeypatch, stage)
    run_dir = tmp_path / "run"
    extras = {
        "status": "succeeded", "stage": "beginning", "kind": "source", "attempt": 999,
        "input_sha256": "f" * 64, "artifact_hashes": {"spoof": "f" * 64},
        "command": ["spoofed"], "log_path": "spoofed.log",
    }

    def extra_result(_command, log_path, _env):
        _write_formal_target_log(log_path, job)
        payload = json.loads(log_path.read_text().split("=", 1)[1])
        payload.update(extras)
        log_path.write_text("HUST_RESULT_JSON=" + json.dumps(payload, separators=(",", ":")) + "\n")
        return 0

    result = resume_pipeline(run_dir, cfg, [job], [0], run_process=extra_result)[0]
    assert result["status"] == "failed" and result["attempt"] == 1
    assert result["stage"] == stage and result["kind"] == "target"
    assert result["command"] != extras["command"] and result["log_path"] != extras["log_path"]
    assert "unexpected" in result["parse_error"]
    records = tuning._state_records(run_dir)
    assert records[0]["status"] == "failed" and records[0]["stage"] == stage
    report_dir = tmp_path / "report"
    write_reports(report_dir, records, frozen=None, checkpoint_manifest={})
    assert list(csv.DictReader((report_dir / "beginning_audit.csv").open())) == []


def test_state_records_flattens_only_reviewed_metrics_and_never_nested_controls(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    state = {
        "kind": "target", "status": "failed", "stage": "baseline", "attempt": 1,
        "route": "0711_robust", "variant": "robust", "candidate_id": "untuned",
        "task": [0, 1], "source": 0, "target": 1, "source_seed": 2025,
        "stream_seed": 2025, "config_sha256": "a" * 64,
        "source_checkpoint_sha256": "b" * 64, "expected_result_kind": "target",
        "metrics": {
            "result_kind": "target", "route": "0711_robust", "variant": "robust",
            "candidate_id": "untuned", "task": [0, 1], "source": 0, "target": 1,
            "source_seed": 2025, "stream_seed": 2025, "config_sha256": "a" * 64,
            "source_checkpoint_sha256": "b" * 64, "strict_online": 88.0,
            "status": "succeeded", "stage": "beginning", "kind": "source",
            "attempt": 999, "input_sha256": "f" * 64,
        },
    }
    (state_dir / "failed.json").write_text(json.dumps(state))
    record = tuning._state_records(tmp_path)[0]
    assert record["status"] == "failed" and record["stage"] == "baseline"
    assert record["kind"] == "target" and record["attempt"] == 1
    assert record["strict_online"] == 88.0 and record["result_kind"] == "target"
    assert "input_sha256" not in record
    report_dir = tmp_path / "report"
    write_reports(report_dir, [record], frozen=None, checkpoint_manifest={})
    assert list(csv.DictReader((report_dir / "beginning_audit.csv").open())) == []
    state["metrics"]["candidate_id"] = "nested-spoof"
    (state_dir / "failed.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="metric identity mismatch"):
        tuning._state_records(tmp_path)


def test_beginning_audit_status_and_supplementary_never_replace_primary(tmp_path):
    tasks = [(s, t) for s in range(4) for t in range(4) if s != t]
    beginning = [_synthetic_beginning_row("robust", task, value) for task, value in zip(tasks, [10.0] + [40.0] * 11)]
    outputs = write_reports(tmp_path, beginning, frozen=None, checkpoint_manifest={}, supplementary_rows=[{"split": "load", "before": 50.0}])
    assert outputs["beginning_status"] == "supplementary_load_audit_required"
    report = (tmp_path / "report.md").read_text()
    assert "Primary bearing split retained" in report


def test_complete_synthetic_report_contains_exact_tables_and_rejects_evidence_tamper(tmp_path):
    tasks = [(s, t) for s in range(4) for t in range(4) if s != t]
    rows = []
    for variant, route in (("ordinary", "dtcc_ordinary"), ("robust", "0711_robust")):
        rows += [_synthetic_beginning_row(variant, task, 35.0 + task[0]) for task in tasks]
    families = [
        ("dtcc_ordinary", "ordinary", "dtcc_ordinary", 60.0),
        ("0711_common", "ordinary", "0711_common", 58.0),
        ("0711_robust", "robust", "untuned", 61.0),
    ]
    for route, variant, cid, score in families:
        rows += [{"route": route, "variant": variant, "candidate_id": cid, "stage": "baseline", "result_kind": "target", "task": list(task), "source": task[0], "target": task[1], "stream_seed": 2025, "status": "succeeded", "before": 40.0, "strict_online": score, "macro_f1": score - 2, "runtime_seconds": 2.0, "peak_memory_mb": 100.0} for task in tasks]
    dev_tasks = {(0, 1), (1, 2), (2, 3), (3, 0)}
    for task in tasks:
        stage = "tune_group_09_contrastive_temperature" if task in dev_tasks else "heldout"
        rows.append({"route": "0711_robust", "variant": "robust", "candidate_id": "frozen", "stage": stage, "result_kind": "target", "task": list(task), "source": task[0], "target": task[1], "stream_seed": 2025, "status": "succeeded", "before": 40.0, "strict_online": 62.0, "macro_f1": 60.0, "runtime_seconds": 2.0, "peak_memory_mb": 100.0})
    for cid, score in (("untuned", 60.0), ("frozen", 61.0)):
        rows += [{"route": "0711_robust", "variant": "robust", "candidate_id": cid, "stage": "stability", "result_kind": "target", "task": list(task), "source": task[0], "target": task[1], "stream_seed": 2026, "status": "succeeded", "before": 40.0, "strict_online": score, "macro_f1": score - 2, "runtime_seconds": 2.0, "peak_memory_mb": 100.0} for task in tasks]

    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"checkpoint")
    log = tmp_path / "source.log"
    log.write_text("source log\n")
    command = tmp_path / "source.command.txt"
    command.write_text("python source.py\n")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    rows.append({"route": "source_ordinary", "variant": "ordinary", "candidate_id": "source-ordinary-0", "stage": "source", "status": "succeeded", "attempt": 1, "returncode": 0, "log_path": str(log), "log_sha256": digest(log), "command_path": str(command), "command_log_sha256": digest(command), "output_artifact_hashes": {str(checkpoint): digest(checkpoint)}})
    frozen = {"candidate_id": "frozen", "overrides": {"Opt.lr_tar": 0.024}}
    manifest = {"ordinary": {str(checkpoint): digest(checkpoint)}}

    output = write_reports(tmp_path, rows, frozen=frozen, checkpoint_manifest=manifest)
    assert output["recommended"] is True
    report = (tmp_path / "report.md").read_text()
    assert "Exact per-task Beginning / strict accuracy / macro-F1" in report
    assert "Seed-2025 full-12 gain: `1.0`" in report and "Seed-2026 matched gain: `1.0`" in report
    assert "0→1" in report and "62.0000" in report
    summary = json.loads((tmp_path / "report_summary.json").read_text())
    assert summary["ranges"]["frozen"]["strict_online"] == {"mean": 62.0, "min": 62.0, "max": 62.0}
    assert (tmp_path / "protocol_audit.json").is_file() and (tmp_path / "attempts.csv").is_file()

    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint|artifact"):
        write_reports(tmp_path / "tampered-checkpoint", rows, frozen=frozen, checkpoint_manifest=manifest)
    checkpoint.write_bytes(b"checkpoint")
    log.write_text("tampered log\n")
    with pytest.raises(ValueError, match="log_path"):
        write_reports(tmp_path / "tampered-log", rows, frozen=frozen, checkpoint_manifest=manifest)


def test_recommendation_uses_exact_full12_mean_not_development_or_seed2026_guard(tmp_path):
    tasks = [(s, t) for s in range(4) for t in range(4) if s != t]
    rows = []
    for task in tasks:
        rows.append({
            "route": "0711_robust", "variant": "robust", "candidate_id": "untuned",
            "stage": "baseline", "result_kind": "target", "task": list(task),
            "source": task[0], "target": task[1], "stream_seed": 2025,
            "status": "succeeded", "strict_online": 50.0, "macro_f1": 49.0,
        })
        score = 48.5 if task == (0, 1) else 51.0
        rows.append({
            "route": "0711_robust", "variant": "robust", "candidate_id": "frozen",
            "stage": "baseline", "result_kind": "target", "task": list(task),
            "source": task[0], "target": task[1], "stream_seed": 2025,
            "status": "succeeded", "strict_online": score, "macro_f1": score - 1.0,
        })
    result = write_reports(
        tmp_path, rows, frozen={"candidate_id": "frozen", "overrides": {}},
        checkpoint_manifest={},
    )
    summary = json.loads((tmp_path / "report_summary.json").read_text())
    assert result["recommended"] is True
    assert summary["seed2025_development_guard_passed"] is False
    assert summary["stability_seed2026_guard_passed"] is False


def test_untuned_fallback_is_explicitly_never_recommended_despite_numeric_jitter(
    tmp_path, monkeypatch
):
    tasks = [(s, t) for s in range(4) for t in range(4) if s != t]
    baseline = [{
        "route": "0711_robust", "variant": "robust", "candidate_id": "untuned",
        "stage": "baseline", "result_kind": "target", "task": list(task),
        "source": task[0], "target": task[1], "stream_seed": 2025,
        "status": "succeeded", "strict_online": 50.0, "macro_f1": 49.0,
    } for task in tasks]
    jittered_frozen = [
        {**row, "strict_online": 50.01, "macro_f1": 49.01}
        for row in baseline
    ]
    monkeypatch.setattr(
        summarizer, "_frozen_primary_rows",
        lambda _rows, _candidate: jittered_frozen,
    )
    result = write_reports(
        tmp_path, baseline,
        frozen={"candidate_id": "untuned", "overrides": {}},
        checkpoint_manifest={},
    )
    summary = json.loads((tmp_path / "report_summary.json").read_text())
    assert result["recommended"] is False
    assert summary["recommendation_reason"] == "untuned_fallback"
    assert "untuned fallback" in (tmp_path / "report.md").read_text().lower()


def test_seed2026_final_stability_reuses_tune_development_and_audits_redundant_reruns(tmp_path):
    tasks = [(s, t) for s in range(4) for t in range(4) if s != t]
    rows = []
    for cid, score in (("untuned", 60.0), ("frozen", 61.0)):
        for task in tasks:
            rows.append({"route": "0711_robust", "variant": "robust", "candidate_id": cid,
                         "stage": "stability", "result_kind": "target", "task": list(task),
                         "source": task[0], "target": task[1], "source_seed": 2025,
                         "stream_seed": 2026, "config_sha256": ("a" if cid == "untuned" else "b") * 64,
                         "source_checkpoint_sha256": str(task[0] + 1) * 64,
                         "overrides": {"Opt.lr_tar": 0.01 if cid == "untuned" else 0.02},
                         "cache_manifest_path": "/cache/manifest.json",
                         "cache_manifest_sha256": "d" * 64,
                         "cache_content_sha256": "e" * 64,
                         "cache_tensor_sha256s": {f"/cache/domain_{domain}.pt": str(domain + 1) * 64 for domain in range(4)},
                         "status": "succeeded", "strict_online": score, "macro_f1": score - 1})
        for task in DEV_TASKS:
            final = next(row for row in rows if row["candidate_id"] == cid and tuple(row["task"]) == task)
            rows.append({**final, "stage": "tune_stability"})
    rows += [{**next(row for row in rows if row["candidate_id"] == "frozen" and tuple(row["task"]) == task),
              "candidate_id": "rejected-finalist", "config_sha256": "c" * 64,
              "stage": "tune_stability", "strict_online": 59.0, "macro_f1": 58.0}
             for task in DEV_TASKS]
    frozen = {"candidate_id": "frozen", "overrides": {}}
    redundant = next(row for row in rows if row["candidate_id"] == "frozen" and row["stage"] == "stability" and tuple(row["task"]) == (3, 0))
    redundant["strict_online"] = 61.25
    redundant["macro_f1"] = 60.125

    write_reports(tmp_path, rows, frozen=frozen, checkpoint_manifest={})
    authoritative = list(csv.DictReader((tmp_path / "stability_seed2026.csv").open()))
    audit = list(csv.DictReader((tmp_path / "stability_seed2026_redundant_audit.csv").open()))
    summary = json.loads((tmp_path / "report_summary.json").read_text())
    assert len(authoritative) == 24
    assert {row["stage"] for row in authoritative} == {"tune_stability", "stability"}
    assert len(audit) == 8 and all(row["excluded_from_authoritative_metrics"] == "True" for row in audit)
    drift = next(row for row in audit if row["candidate_id"] == "frozen" and row["task"] == "[3,0]")
    assert float(drift["strict_online_drift"]) == pytest.approx(0.25)
    assert summary["stability_seed2026_gain"] == pytest.approx(1.0)
    assert summary["stability_redundant_audit"]["rows"] == 8

    identity_mismatch = [dict(row) for row in rows]
    next(row for row in identity_mismatch if row["candidate_id"] == "frozen" and row["stage"] == "stability")["config_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="identity|config|overlap"):
        write_reports(tmp_path / "identity-mismatch", identity_mismatch, frozen=frozen, checkpoint_manifest={})

    missing_tune_dev = [
        row for row in rows
        if not (row["candidate_id"] == "frozen" and row["stage"] == "tune_stability" and tuple(row["task"]) == (3, 0))
    ]
    with pytest.raises(ValueError, match="authoritative|incomplete|tune_stability"):
        write_reports(tmp_path / "missing-tune-dev", missing_tune_dev, frozen=frozen, checkpoint_manifest={})


def _seed2026_cache_identity_rows():
    tasks = [(s, t) for s in range(4) for t in range(4) if s != t]
    cache = {
        "cache_manifest_path": "/cache/manifest.json",
        "cache_manifest_sha256": "a" * 64,
        "cache_content_sha256": "b" * 64,
        "cache_tensor_sha256s": {
            f"/cache/domain_{domain}.pt": str(domain + 1) * 64
            for domain in range(4)
        },
    }
    rows = []
    for task in tasks:
        rows.append({
            "route": "0711_robust", "variant": "robust",
            "candidate_id": "frozen", "stage": "stability",
            "result_kind": "target", "task": list(task),
            "source": task[0], "target": task[1], "source_seed": 2025,
            "stream_seed": 2026, "config_sha256": "c" * 64,
            "source_checkpoint_sha256": str(task[0] + 5) * 64,
            "overrides": {"Opt.lr_tar": 0.02}, **cache,
            "status": "succeeded", "strict_online": 61.0,
            "macro_f1": 60.0,
        })
    for task in DEV_TASKS:
        final = next(row for row in rows if tuple(row["task"]) == task)
        rows.append({
            **final, "stage": "tune_stability",
            "_state_path": f"/proof/tune_stability_{task[0]}to{task[1]}.json",
        })
    return rows, cache


def _seed2026_cache_proof(rows, cache):
    return {
        **cache,
        "stability_evidence": [
            {
                "candidate_id": row["candidate_id"], "task": row["task"],
                "state_path": row["_state_path"],
                "config_sha256": row["config_sha256"], "stage": row["stage"],
                "stream_seed": row["stream_seed"],
            }
            for row in rows if row["stage"] == "tune_stability"
        ],
    }


@pytest.mark.parametrize("row_kind", ["authoritative_nondev", "redundant_dev"])
@pytest.mark.parametrize(
    ("field", "different"),
    [
        ("cache_manifest_path", "/other-valid-cache/manifest.json"),
        ("cache_manifest_sha256", "d" * 64),
        ("cache_content_sha256", "e" * 64),
        ("cache_tensor_sha256s", {"/other-valid-cache/domain_0.pt": "f" * 64}),
    ],
)
def test_seed2026_stability_rejects_cache_identity_drift_on_every_final_row_kind(
    row_kind, field, different
):
    rows, cache = _seed2026_cache_identity_rows()
    proof = _seed2026_cache_proof(rows, cache)
    task = (0, 2) if row_kind == "authoritative_nondev" else (0, 1)
    target = next(
        row for row in rows
        if row["stage"] == "stability" and tuple(row["task"]) == task
    )
    target[field] = different
    with pytest.raises(ValueError, match="cache.*identity|proof.*cache"):
        summarizer._authoritative_seed2026_candidate(
            rows, "frozen", proof=proof, required=True
        )


@pytest.mark.parametrize(
    ("field", "different"),
    [
        ("cache_manifest_path", "/other-valid-cache/manifest.json"),
        ("cache_manifest_sha256", "d" * 64),
        ("cache_content_sha256", "e" * 64),
        ("cache_tensor_sha256s", {"/other-valid-cache/domain_0.pt": "f" * 64}),
    ],
)
def test_seed2026_stability_rejects_internally_consistent_cache_not_bound_to_proof(
    field, different
):
    rows, cache = _seed2026_cache_identity_rows()
    proof = _seed2026_cache_proof(rows, cache)
    for row in rows:
        row[field] = different
    with pytest.raises(ValueError, match="proof.*cache"):
        summarizer._authoritative_seed2026_candidate(
            rows, "frozen", proof=proof, required=True
        )


def test_supplementary_load_audit_requires_exact_18_record_inventory(tmp_path):
    records = []
    for variant in ("ordinary", "robust"):
        for source in range(3):
            records.append({"kind": "source", "stage": "load-audit-source", "route": f"source_{variant}",
                            "variant": variant, "source": source, "source_seed": 2025, "load_split": True,
                            "status": "succeeded", "source_checkpoint_path": f"/load/{variant}/{source}.pt",
                            "source_checkpoint_sha256": str(source + (1 if variant == "ordinary" else 4)) * 64})
        for source in range(3):
            for target in range(3):
                if source != target:
                    records.append({"kind": "target", "stage": "load-audit-beginning",
                                    "route": "0711_robust" if variant == "robust" else "dtcc_ordinary",
                                    "variant": variant, "candidate_id": f"load-beginning-{variant}",
                                    "task": [source, target], "source": source, "target": target,
                                    "source_seed": 2025, "stream_seed": 2025, "load_split": True,
                                    "result_kind": "beginning", "status": "succeeded", "before": 45.0,
                                    "strict_online": None, "macro_f1": None})
    result = write_reports(tmp_path, records, frozen=None, checkpoint_manifest={})
    assert result["supplementary_complete"] is True
    assert len(list(csv.DictReader((tmp_path / "supplementary_load_source_inventory.csv").open()))) == 6
    assert json.loads((tmp_path / "supplementary_load_status.json").read_text())["complete_records"] == 18
    incomplete = write_reports(tmp_path / "missing", records[:-1], frozen=None, checkpoint_manifest={})
    assert incomplete["supplementary_complete"] is False
    for name, mutation in (
        ("duplicate", lambda values: [*values, dict(values[0])]),
        ("wrong-split", lambda values: [{**values[0], "load_split": False}, *values[1:]]),
        ("wrong-route", lambda values: [{**values[0], "route": "source_robust"}, *values[1:]]),
        ("wrong-hash", lambda values: [{**values[0], "source_checkpoint_sha256": "not-a-hash"}, *values[1:]]),
    ):
        result = write_reports(tmp_path / name, mutation(records), frozen=None, checkpoint_manifest={})
        assert result["supplementary_complete"] is False


def _formal_beginning_state(tmp_path: Path, monkeypatch) -> tuple[dict, Path]:
    evidence = tmp_path / "evidence"
    evidence.mkdir(parents=True)
    paths = {name: evidence / name for name in ("manifest.json", "runner.py", "experiment.yaml", "source.pt", "source_training_summary.json")}
    for name, path in paths.items():
        path.write_bytes((name + "\n").encode())
    cache = _install_fake_v2_cache(monkeypatch, evidence, split="bearing")
    confusion = None
    result = build_hust_result_record(
        result_kind="beginning", route="dtcc_ordinary", variant="ordinary", task=(0, 1),
        source_checkpoint_sha256=hashlib.sha256(paths["source.pt"].read_bytes()).hexdigest(),
        config_sha256="b" * 64, candidate_id="beginning-ordinary", source_seed=2025,
        stream_seed=2025, beginning=42.0, strict_online=None, post_stream=None,
        confusion_matrix=confusion, samples=140, batches=0, passes=0, finite_losses=True,
        trainable_parameters=[], pre_update_scoring=True, metadata_evidence_used=False,
        runtime_seconds=0.5, peak_memory_mb=10.0,
    )
    log = evidence / "attempt-1.log"
    log.write_text("HUST_RESULT_JSON=" + json.dumps(result, separators=(",", ":")) + "\n")
    command_path = evidence / "attempt-1.txt"
    command = ["python", "runner.py"]
    command_path.write_text("python runner.py\n")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    expected = {"candidate_id": "beginning-ordinary", "route": "dtcc_ordinary", "variant": "ordinary",
                "task": [0, 1], "source": 0, "target": 1, "source_seed": 2025,
                "stream_seed": 2025, "config_sha256": "b" * 64, "result_kind": "beginning",
                "source_checkpoint_sha256": result["source_checkpoint_sha256"]}
    metrics = parse_runner_log(log, "dtcc_ordinary", expected=expected)
    cache_tensors = {
        str((evidence / row["tensor_file"]).resolve()): row["tensor_sha256"]
        for row in cache["domains"].values()
    }
    artifact_hashes = {
        **{str(path.resolve()): digest(path) for path in paths.values()},
        **cache_tensors,
    }
    state = {"_formal_state": True, "kind": "target", "stage": "beginning", "route": "dtcc_ordinary",
             "variant": "ordinary", "candidate_id": "beginning-ordinary", "task": [0, 1],
             "source": 0, "target": 1, "source_seed": 2025, "stream_seed": 2025,
             "config_sha256": "b" * 64, "load_split": False, "expected_result_kind": "beginning",
             "command": command, "command_sha256": hashlib.sha256(json.dumps(command, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
             "command_path": str(command_path.resolve()), "command_log_sha256": digest(command_path),
             "log_path": str(log.resolve()), "log_sha256": digest(log), "artifacts": list(artifact_hashes),
             "artifact_hashes": artifact_hashes, "cache_manifest_path": str(paths["manifest.json"]),
             "cache_manifest_sha256": digest(paths["manifest.json"]), "runner_script_path": str(paths["runner.py"]),
             "cache_content_sha256": cache["content_sha256"], "cache_tensor_sha256s": cache_tensors,
             "runner_script_sha256": digest(paths["runner.py"]), "experiment_config_path": str(paths["experiment.yaml"]),
             "experiment_config_sha256": digest(paths["experiment.yaml"]), "source_checkpoint_path": str(paths["source.pt"]),
             "source_checkpoint_sha256": digest(paths["source.pt"]), "source_summary_path": str(paths["source_training_summary.json"]),
             "source_summary_sha256": digest(paths["source_training_summary.json"]), "expected_result_contract": "beginning",
             "result_identity": expected, "gpu": 0, "status": "succeeded", "attempt": 1,
             "started_at": 1.0, "ended_at": 2.0, "returncode": 0, "metrics": metrics,
             **metrics}
    return state, log


def test_formal_report_reparses_hashed_log_and_rejects_evidence_deletion_or_inner_tamper(tmp_path, monkeypatch):
    state, log = _formal_beginning_state(tmp_path, monkeypatch)
    write_reports(tmp_path / "valid", [state], frozen=None, checkpoint_manifest={})
    for deleted in ("log_sha256", "command_path", "runner_script_sha256", "source_summary_path", "metrics"):
        bad = dict(state)
        bad.pop(deleted)
        with pytest.raises(ValueError, match="evidence|missing"):
            write_reports(tmp_path / f"missing-{deleted}", [bad], frozen=None, checkpoint_manifest={})
    bad = json.loads(json.dumps(state))
    bad["metrics"]["before"] = 99.0
    with pytest.raises(ValueError, match="metrics"):
        write_reports(tmp_path / "saved-metric", [bad], frozen=None, checkpoint_manifest={})
    payload = json.loads(re.search(r"HUST_RESULT_JSON=(\{.*\})", log.read_text()).group(1))
    payload["beginning"] = 43.0
    log.write_text("HUST_RESULT_JSON=" + json.dumps(payload, separators=(",", ":")) + "\n")
    bad = dict(state)
    bad["log_sha256"] = hashlib.sha256(log.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="metrics|identity"):
        write_reports(tmp_path / "inner-log", [bad], frozen=None, checkpoint_manifest={})


@pytest.mark.parametrize("stage", ["source", "beginning", "baseline", "final"])
def test_formal_cli_stage_raises_when_any_required_job_failed(
    config, tmp_path, monkeypatch, stage
):
    """Returning a failed record list with exit zero would publish a partial stage."""
    monkeypatch.setattr(
        tuning, "plan_stage_jobs", lambda *_args, **_kwargs: [{"stage": stage}]
    )
    monkeypatch.setattr(
        tuning,
        "execute_parallel_tasks",
        lambda *_args, **_kwargs: [
            {"stage": stage, "status": "failed", "attempt": 1}
        ],
    )
    with pytest.raises(RuntimeError, match="failed|incomplete"):
        tuning.run_formal_stage(
            tmp_path,
            config,
            stage,
            [0],
            run_process=lambda *_args: 0,
        )


def test_tune_stage_raises_on_failed_candidate_inventory(config, tmp_path):
    baseline = [
        {
            "candidate_id": "untuned", "route": "0711_robust",
            "stage": "baseline", "task": task, "stream_seed": 2025,
            "status": "succeeded", "strict_online": 50.0,
        }
        for task in DEV_TASKS
    ]

    def failed_executor(_run_dir, _config, jobs, _gpus, **_kwargs):
        return [{**job, "status": "failed", "attempt": 1} for job in jobs]

    with pytest.raises(RuntimeError, match="tuning group.*failed|incomplete"):
        run_tuning_stage(
            tmp_path, config, [0], executor=failed_executor,
            baseline_records=baseline,
        )


def test_strict_final_report_preflights_before_writing_any_artifact(tmp_path):
    """A partial inventory must leave no publication files behind."""
    with pytest.raises(ValueError, match="final inventory"):
        write_reports(
            tmp_path, [], frozen=None, checkpoint_manifest={}, strict_final=True
        )
    assert list(tmp_path.iterdir()) == []


def test_write_final_rejects_forged_eight_key_manifest_before_publication(
    tmp_path, monkeypatch
):
    tasks = [(s, t) for s in range(4) for t in range(4) if s != t]
    records = []
    authoritative_manifest = {}
    for variant in ("ordinary", "robust"):
        for source in range(4):
            candidate = f"source-{variant}-{source}-2025"
            checkpoint = f"/authority/{variant}/source_{source}/seed_2025/model.pt"
            summary = f"/authority/{variant}/source_{source}/seed_2025/source_training_summary.json"
            checkpoint_hash = hashlib.sha256(candidate.encode()).hexdigest()
            summary_hash = hashlib.sha256((candidate + "-summary").encode()).hexdigest()
            outputs = {checkpoint: checkpoint_hash, summary: summary_hash}
            authoritative_manifest[candidate] = outputs
            records.append({
                "_formal_state": True, "kind": "source", "stage": "source",
                "route": f"source_{variant}", "variant": variant,
                "candidate_id": candidate, "source": source, "source_seed": 2025,
                "load_split": False, "status": "succeeded",
                "source_checkpoint_path": checkpoint,
                "output_artifact_hashes": outputs,
            })
    for variant in ("ordinary", "robust"):
        records.extend({**_synthetic_beginning_row(variant, task, 40.0), "_formal_state": True} for task in tasks)
    for route, candidate in (
        ("dtcc_ordinary", "dtcc_ordinary"),
        ("0711_robust", "untuned"),
        ("0711_common", "0711_common"),
    ):
        records.extend({
            "_formal_state": True, "kind": "target", "route": route,
            "variant": "robust" if route == "0711_robust" else "ordinary",
            "candidate_id": candidate, "stage": "baseline", "task": list(task),
            "source": task[0], "target": task[1], "source_seed": 2025,
            "stream_seed": 2025, "status": "succeeded", "strict_online": 50.0,
            "macro_f1": 49.0,
        } for task in tasks)
    for candidate in ("untuned", "selected"):
        records.extend({
            "_formal_state": True, "kind": "target", "route": "0711_robust",
            "variant": "robust", "candidate_id": candidate, "stage": "stability",
            "task": list(task), "source": task[0], "target": task[1],
            "source_seed": 2025, "stream_seed": 2026, "status": "succeeded",
            "strict_online": 50.0, "macro_f1": 49.0,
        } for task in tasks)

    proof_path = tmp_path / "search_proof.json"
    proof_path.write_text("{}\n")
    frozen = {"candidate_id": "selected", "proof_path": str(proof_path.resolve())}
    (tmp_path / "best_config.yaml").write_text(yaml.safe_dump(frozen))
    forged = {f"forged-{index}": {} for index in range(8)}
    (tmp_path / "checkpoint_manifest.json").write_text(json.dumps(forged) + "\n")
    proof = {
        "selected": {"candidate_id": "selected"},
        "checkpoint_manifest": authoritative_manifest,
    }
    monkeypatch.setattr(tuning, "load_frozen_candidate", lambda _path: frozen)
    monkeypatch.setattr(tuning, "verify_search_proof", lambda _path: proof)
    monkeypatch.setattr(summarizer, "load_state_records", lambda _run_dir: records)
    monkeypatch.setattr(
        summarizer, "_authoritative_frozen_primary_rows",
        lambda _records, _frozen, _proof: [
            {"task": list(task), "strict_online": 51.0, "macro_f1": 50.0}
            for task in tasks
        ],
    )
    with pytest.raises(ValueError, match="checkpoint manifest"):
        summarize_main(["--run-dir", str(tmp_path), "--write-final"])
    assert not (tmp_path / "report.md").exists()
    assert not (tmp_path / "metrics.csv").exists()


def test_load_cache_gate_builds_v2_once_then_reuses_valid_cache(
    config, tmp_path, monkeypatch
):
    """A clean all-run must validate the builder output before load-audit launch."""
    cfg = json.loads(json.dumps(config))
    output = tmp_path / "HUST_STRICT_LOAD_CACHE_V2"
    cfg["protocol"]["load_data_path"] = str(output)
    validations = []

    def validate(path):
        validations.append(Path(path))
        if len(validations) == 1:
            raise ValueError("missing")
        return {"version": 2, "split": "load", "content_sha256": "a" * 64}

    commands = []
    monkeypatch.setattr("Lib.hust_strict_protocol.validate_cache", validate)
    result = tuning.ensure_v2_load_cache(
        cfg, builder_run=lambda command: commands.append(command)
    )
    assert result["content_sha256"] == "a" * 64
    assert len(commands) == 1
    assert commands[0][-4:] == ["--seed", "2025", "--split", "load"]
    assert str(output) in commands[0]

    tuning.ensure_v2_load_cache(
        cfg,
        builder_run=lambda _command: pytest.fail("valid cache was rebuilt"),
    )


def test_load_cache_builder_failure_propagates_without_audit_launch(
    config, tmp_path, monkeypatch
):
    cfg = json.loads(json.dumps(config))
    cfg["protocol"]["load_data_path"] = str(tmp_path / "HUST_STRICT_LOAD_CACHE_V2")
    monkeypatch.setattr(
        "Lib.hust_strict_protocol.validate_cache",
        lambda _path: (_ for _ in ()).throw(ValueError("missing")),
    )
    with pytest.raises(RuntimeError, match="builder failed"):
        tuning.ensure_v2_load_cache(
            cfg,
            builder_run=lambda _command: (_ for _ in ()).throw(
                RuntimeError("builder failed")
            ),
        )


def test_all_stage_builder_failure_occurs_before_any_load_audit_gpu_job(
    config, tmp_path, monkeypatch
):
    launched = []
    monkeypatch.setattr(
        tuning, "plan_stage_jobs",
        lambda _config, stage, _run_dir, **_kwargs: [{"stage": stage}],
    )

    def execute(_run_dir, _config, jobs, _gpus, **_kwargs):
        launched.extend(job["stage"] for job in jobs)
        return [{**job, "status": "succeeded"} for job in jobs]

    monkeypatch.setattr(tuning, "execute_parallel_tasks", execute)
    tasks = [(s, t) for s in range(4) for t in range(4) if s != t]
    beginning_rows = [
        {
            "stage": "beginning", "status": "succeeded", "variant": variant,
            "task": task, "stream_seed": 2025,
            "before": 10.0 if variant == "robust" and task == DEV_TASKS[0] else 40.0,
        }
        for variant in ("ordinary", "robust")
        for task in tasks
    ]
    monkeypatch.setattr(tuning, "_state_records", lambda _run_dir: beginning_rows)
    monkeypatch.setattr(
        tuning, "ensure_v2_load_cache",
        lambda _config: (_ for _ in ()).throw(RuntimeError("builder failed")),
    )
    with pytest.raises(RuntimeError, match="builder failed"):
        tuning.run_formal_stage(tmp_path, config, "all", [0])
    assert launched == ["source", "beginning"]


def _write_minimal_authority_state(run_dir: Path, config: dict, row: dict) -> dict:
    row = dict(row)
    planned = tuning._planned_target(
        config, row["stage"], row["route"], tuple(row["task"]),
        row["stream_seed"], row.get("overrides", {}),
        candidate=row["candidate_id"],
    )
    row["config_sha256"] = planned["config_sha256"]
    _candidate, stem = tuning._job_identity(config, row)
    command = run_dir / "commands" / f"{stem}.attempt-1.txt"
    log = run_dir / "logs" / f"{stem}.attempt-1.log"
    state_path = run_dir / "state" / f"{stem}.json"
    command.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    command_value = tuning._target_command_for_job(config, row, 0)
    command.write_text(shlex.join(command_value) + "\n")
    log.write_text("controlled result\n")
    state = {
        **row,
        "task": list(row["task"]),
        "status": "succeeded",
        "returncode": 0,
        "command": command_value,
        "command_sha256": hashlib.sha256(
            json.dumps(command_value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "gpu": 0,
        "started_at": 1.0,
        "ended_at": 2.0,
        "command_path": str(command.resolve()),
        "command_log_sha256": hashlib.sha256(command.read_bytes()).hexdigest(),
        "log_path": str(log.resolve()),
        "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
    }
    state_path.write_text(json.dumps(state, sort_keys=True, indent=2) + "\n")
    return state


def test_authoritative_freeze_recomputes_bound_evidence_and_final_jobs_bind_both_files(
    config, tmp_path, monkeypatch
):
    cfg = json.loads(json.dumps(config))
    cache_root = tmp_path / "HUST_STRICT_CACHE_V2"
    cache = _install_fake_v2_cache(monkeypatch, cache_root, split="bearing")
    cfg["protocol"]["data_path"] = str(cache_root)
    checkpoint_root = tmp_path / "sources"
    cfg["checkpoints"]["root"] = str(checkpoint_root)
    monkeypatch.setattr(tuning, "load_config", lambda _path: cfg)
    monkeypatch.setattr(tuning, "validate_config", lambda _config: None)

    checkpoint_manifest = {}
    tensor_hashes = {
        str((cache_root / row["tensor_file"]).resolve()): row["tensor_sha256"]
        for row in cache["domains"].values()
    }
    for variant in ("ordinary", "robust"):
        for source in range(4):
            directory = checkpoint_root / variant / f"source_{source}" / "seed_2025"
            checkpoint = directory / "ResNet18_1D_SDE2025fft_Linear.pt"
            summary = directory / "source_training_summary.json"
            directory.mkdir(parents=True)
            checkpoint.write_bytes(f"{variant}-{source}\n".encode())
            checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            summary.write_text(json.dumps({
                "route": variant, "source": source, "seed": 2025,
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": checkpoint_digest,
                "cache_identity": {
                    "mode": "formal",
                    "manifest_path": str((cache_root / "manifest.json").resolve()),
                    "manifest_sha256": cache["manifest_sha256"],
                    "content_sha256": cache["content_sha256"],
                    "tensor_sha256s": tensor_hashes,
                },
            }, sort_keys=True) + "\n")
            checkpoint_manifest[f"source-{variant}-{source}-2025"] = {
                str(checkpoint.resolve()): checkpoint_digest,
                str(summary.resolve()): hashlib.sha256(summary.read_bytes()).hexdigest(),
            }

    baseline_rows = []
    for task in DEV_TASKS:
        baseline_rows.append(_write_minimal_authority_state(tmp_path, cfg, {
            "kind": "target", "stage": "baseline", "route": "0711_robust",
            "variant": "robust", "candidate_id": "untuned", "task": task,
            "source": task[0], "target": task[1], "source_seed": 2025,
            "stream_seed": 2025, "config_sha256": "a" * 64,
            "strict_online": 50.0, "runtime_seconds": 1.0, "overrides": {},
        }))
    coordinate_rows = []
    baseline = {task: 50.0 for task in DEV_TASKS}
    anchor = dict(cfg["baseline_overrides"])
    group_history = []
    initial = dict(cfg["search"]["initial_candidate"]["overrides"])
    for group_index, group in enumerate(cfg["search"]["groups"]):
        values = list(group["values"])
        if group_index == 0:
            values.append(initial)
        candidates = tuning.expand_coordinate_group(anchor, values)
        stage = f"tune_group_{group_index:02d}_{group['name']}"
        stage_rows = []
        for overrides in candidates:
            identifier = candidate_id(overrides, 2025)
            digest = hashlib.sha256(
                json.dumps(overrides, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            for task in DEV_TASKS:
                stage_rows.append(_write_minimal_authority_state(tmp_path, cfg, {
                    "kind": "target", "stage": stage,
                    "route": "0711_robust", "variant": "robust",
                    "candidate_id": identifier, "task": task,
                    "source": task[0], "target": task[1], "source_seed": 2025,
                    "stream_seed": 2025, "config_sha256": digest,
                    "strict_online": 51.0, "runtime_seconds": 1.0,
                    "overrides": overrides,
                }))
        ranked_group = rank_candidates(
            stage_rows, set(DEV_TASKS), baseline,
            expected_stages={2025: stage}, eligibility_seeds={2025},
        )
        winner = ranked_group[0]
        group_history.append({
            "group_index": group_index, "group": group["name"],
            "candidate_ids": sorted({row["candidate_id"] for row in stage_rows}),
            "winner": winner,
        })
        anchor = dict(winner["overrides"])
        coordinate_rows.extend(stage_rows)
    global_ranking = tuning.rank_coordinate_candidates_globally(
        coordinate_rows, set(DEV_TASKS), baseline,
        minimum_gain=0.30, maximum_regression=1.00,
    )
    finalists = global_ranking[:3] if len(global_ranking) >= 3 else global_ranking[:2]
    stability_rows = []
    for candidate, values, score in [
        ("untuned", dict(cfg["baseline_overrides"]), 50.0),
        *[(row["candidate_id"], row["overrides"], 49.0) for row in finalists],
    ]:
        digest = hashlib.sha256(
            json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        for task in DEV_TASKS:
            stability_rows.append(_write_minimal_authority_state(tmp_path, cfg, {
                "kind": "target", "stage": "tune_stability", "route": "0711_robust",
                "variant": "robust", "candidate_id": candidate, "task": task,
                "source": task[0], "target": task[1], "source_seed": 2025,
                "stream_seed": 2026, "config_sha256": digest,
                "strict_online": score, "runtime_seconds": 1.0, "overrides": values,
            }))
    matched = {(task, seed): 50.0 for task in DEV_TASKS for seed in (2025, 2026)}
    final_ranking = []
    for finalist in finalists:
        identifier = finalist["candidate_id"]
        stage = dict(finalist["stages"])[2025]
        final_ranking.extend(rank_candidates(
            [row for row in coordinate_rows if row["candidate_id"] == identifier and row["stage"] == stage]
            + [row for row in stability_rows if row["candidate_id"] == identifier],
            set(DEV_TASKS), matched, required_seeds={2025, 2026},
            expected_stages={2025: stage, 2026: "tune_stability"},
            eligibility_seeds={2025},
        ))
    final_ranking.sort(key=lambda row: (-row["mean_strict_online"], -row["minimum_task_delta"], row["runtime_seconds"], row["candidate_id"]))
    selected = final_ranking[0]
    identifier = selected["candidate_id"]
    overrides = selected["overrides"]
    history = {
        "schema_version": 2, "groups": group_history,
        "global_coordinate_ranking": global_ranking,
        "finalists": finalists,
        "matched_final_ranking": final_ranking,
        "matched_seed2026": stability_rows,
        "selected": selected,
    }
    tuning.atomic_write_json(tmp_path / "search_history.json", history)
    proof = tuning.write_search_proof(
        tmp_path, cfg, coordinate_rows=coordinate_rows, stability_rows=stability_rows,
        global_ranking=global_ranking, final_ranking=final_ranking,
        finalists=finalists, selected=selected,
        checkpoint_manifest=checkpoint_manifest,
    )
    proof_bytes = proof.read_bytes()
    assert not (tmp_path / "best_config.yaml").exists()
    assert tuning.write_search_proof(
        tmp_path, cfg, coordinate_rows=coordinate_rows,
        stability_rows=stability_rows, global_ranking=global_ranking,
        final_ranking=final_ranking, finalists=finalists, selected=selected,
        checkpoint_manifest=checkpoint_manifest,
    ) == proof
    assert proof.read_bytes() == proof_bytes
    tampered_proof = json.loads(proof.read_text())
    tampered_proof["selected"]["candidate_id"] = "tampered"
    proof.write_text(json.dumps(tampered_proof, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="proof hash|authoritative"):
        tuning.write_search_proof(
            tmp_path, cfg, coordinate_rows=coordinate_rows,
            stability_rows=stability_rows, global_ranking=global_ranking,
            final_ranking=final_ranking, finalists=finalists, selected=selected,
            checkpoint_manifest=checkpoint_manifest,
        )
    proof.write_bytes(proof_bytes)
    with pytest.raises(ValueError, match="does not match"):
        tuning.write_search_proof(
            tmp_path, cfg, coordinate_rows=coordinate_rows,
            stability_rows=stability_rows, global_ranking=global_ranking,
            final_ranking=final_ranking, finalists=finalists,
            selected=final_ranking[1], checkpoint_manifest=checkpoint_manifest,
        )
    frozen_path = freeze_candidate(
        tmp_path, identifier, overrides, proof_path=proof
    )
    frozen = load_frozen_candidate(frozen_path)
    assert frozen["candidate_id"] == identifier

    monkeypatch.setattr(tuning, "validate_config", lambda _config: None)
    jobs = plan_stage_jobs(cfg, "final", tmp_path, resolve_artifacts=False)
    bound = [job for job in jobs if job["candidate_id"] == identifier]
    assert bound and all(job["freeze_config_path"] == str(frozen_path.resolve()) for job in bound)
    assert all(job["freeze_proof_path"] == str(proof.resolve()) for job in bound)
    assert all(job["freeze_config_sha256"] == hashlib.sha256(frozen_path.read_bytes()).hexdigest() for job in bound)

    state_path = Path(json.loads(proof.read_text())["coordinate_evidence"][0]["state_path"])
    state_path.write_text(state_path.read_text() + " ")
    with pytest.raises(ValueError, match="evidence|proof"):
        load_frozen_candidate(frozen_path)
