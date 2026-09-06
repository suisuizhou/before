from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TUNER_PATH = PROJECT_ROOT / "tools" / "tune_pu4d_0711_strict.py"
SUMMARY_PATH = PROJECT_ROOT / "tools" / "summarize_pu4d_0711_tuning.py"
CONFIG_PATH = PROJECT_ROOT / "Configs" / "Experiments" / "PU4D0711_strict_tuning.yaml"
LAUNCHER_PATH = PROJECT_ROOT / "run_pu4d_0711_tuning.sh"


def _load_tuner():
    spec = importlib.util.spec_from_file_location("pu4d_0711_tuner", TUNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_summary():
    spec = importlib.util.spec_from_file_location("pu4d_0711_summary", SUMMARY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tuner():
    if not TUNER_PATH.is_file():
        pytest.skip(f"tuning module is not implemented yet: {TUNER_PATH}")
    return _load_tuner()


@pytest.fixture
def summary():
    if not SUMMARY_PATH.is_file():
        pytest.skip(f"summary module is not implemented yet: {SUMMARY_PATH}")
    return _load_summary()


@pytest.fixture
def valid_config(tuner):
    return copy.deepcopy(tuner.load_config(CONFIG_PATH))


def test_tuning_config_module_exists():
    assert TUNER_PATH.is_file(), f"missing tuning module: {TUNER_PATH}"


def test_tuning_summary_module_exists():
    assert SUMMARY_PATH.is_file(), f"missing summary module: {SUMMARY_PATH}"


def test_real_config_is_valid(tuner):
    config = tuner.load_config(CONFIG_PATH)
    tuner.validate_config(config)

    assert config["protocol"]["dataset"] == "PU4D"
    assert config["protocol"]["runner"] == "main_tta_0711_strict_online.py"
    assert config["protocol"]["gpu"] == 0
    assert config["recovery"]["expected_mean"] == pytest.approx(62.7342)
    assert tuple(map(tuple, config["tasks"]["development"])) == (
        (0, 1),
        (0, 3),
        (1, 0),
    )
    assert set(config["checkpoints"]["expected_sha256"]) == {
        "0",
        "1",
        "2",
        "3",
    }


def test_parse_task_name_accepts_exact_pu4d_task_syntax(tuner):
    assert tuner.parse_task_name("0to1") == (0, 1)
    with pytest.raises(ValueError, match="0to1"):
        tuner.parse_task_name("0-1")
    with pytest.raises(ValueError, match="unknown"):
        tuner.parse_task_name("4to0")


def test_preflight_audits_dataset_runner_and_checkpoint_hashes(
    tuner, valid_config, tmp_path
):
    (tmp_path / "Dataset" / "PU4D_CACHE").mkdir(parents=True)
    (tmp_path / "Dataset" / "PU4D_CACHE" / "domain.pt").write_bytes(b"cache")
    (tmp_path / "main_tta_0711_strict_online.py").write_text("pass\n")
    expected = {}
    for source in range(4):
        checkpoint = (
            tmp_path
            / "TTA_Model"
            / "PU4D0.001"
            / f"source_{source}"
            / "seed_2025"
            / "best_source_ResNet18_1D_SDE2025fft_Linear.pt"
        )
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(f"source-{source}".encode())
        expected[str(source)] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    valid_config["checkpoints"]["expected_sha256"] = expected

    report = tuner.run_preflight(tmp_path, valid_config, check_gpu=False)

    assert report["passed"] is True
    assert len(report["checkpoints"]) == 4
    assert report["config_sha256"] == hashlib.sha256(
        json.dumps(valid_config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert report["environment"]["python"]
    assert "pytorch" in report["environment"]
    assert "cuda" in report["environment"]
    assert "git_head" in report
    checkpoint = Path(report["checkpoints"][0]["path"])
    checkpoint.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="SHA-256"):
        tuner.run_preflight(tmp_path, valid_config, check_gpu=False)


def test_dry_run_writes_all_recovery_commands_without_starting_tasks(
    tuner, valid_config, tmp_path
):
    commands = tuner.prepare_dry_run(
        tmp_path, valid_config, stage="recovery", only_task=None
    )

    assert len(commands) == 12
    assert len(list((tmp_path / "commands").glob("*.txt"))) == 12
    assert not (tmp_path / "state").exists()
    for command in commands:
        joined = " ".join(command)
        assert "Dataset=PU4D" in command
        assert "gpu_id=0" in command
        assert "++TTA0711.stream_seed=2025" in command
        assert "CWRU" not in joined


def test_dry_run_single_probe_only_accepts_recovery_stage(
    tuner, valid_config, tmp_path
):
    commands = tuner.prepare_dry_run(
        tmp_path, valid_config, stage="recovery", only_task=(0, 1)
    )
    assert len(commands) == 1
    assert "++only_task=[0,1]" in commands[0]

    with pytest.raises(ValueError, match="recovery"):
        tuner.prepare_dry_run(
            tmp_path, valid_config, stage="tune", only_task=(0, 1)
        )


def test_launcher_exists_and_has_valid_bash_syntax():
    assert LAUNCHER_PATH.is_file(), f"missing launcher: {LAUNCHER_PATH}"
    completed = subprocess.run(
        ["bash", "-n", str(LAUNCHER_PATH)], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("protocol", "dataset"), "CWRU"),
        (("protocol", "passes"), 2),
        (("protocol", "batch_size"), 64),
        (("protocol", "model"), "ResNet18"),
        (("protocol", "gpu"), 1),
        (("protocol", "source_seed"), 7),
    ],
)
def test_protocol_breaking_values_are_rejected(tuner, valid_config, path, value):
    target = valid_config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(ValueError, match="protocol"):
        tuner.validate_config(valid_config)


def test_unknown_candidate_parameter_is_rejected(tuner, valid_config):
    valid_config["search"]["groups"][0]["values"][0][
        "TTA0711.physical_harmonics"
    ] = 9

    with pytest.raises(ValueError, match="not tunable"):
        tuner.validate_config(valid_config)


def test_candidate_id_is_order_independent(tuner):
    left = {"Opt.lr_tar": 0.012, "TTA0711.ema_beta": 0.995}
    right = {"TTA0711.ema_beta": 0.995, "Opt.lr_tar": 0.012}

    assert tuner.candidate_id(left, 2025, 2025) == tuner.candidate_id(
        right, 2025, 2025
    )


def test_candidate_id_changes_with_stream_seed(tuner):
    overrides = {"Opt.lr_tar": 0.012}

    assert tuner.candidate_id(overrides, 2025, 2025) != tuner.candidate_id(
        overrides, 2025, 2026
    )


def test_coordinate_group_merges_anchor_and_removes_duplicates(tuner):
    anchor = {"Opt.lr_tar": 0.015, "TTA0711.ema_beta": 0.995}
    values = [
        {"Opt.lr_tar": 0.012},
        {"Opt.lr_tar": 0.015},
        {"Opt.lr_tar": 0.015},
    ]

    assert tuner.expand_coordinate_group(anchor, values) == [
        {"Opt.lr_tar": 0.012, "TTA0711.ema_beta": 0.995},
        {"Opt.lr_tar": 0.015, "TTA0711.ema_beta": 0.995},
    ]


def test_command_forces_strict_invariants(tuner, valid_config):
    command = tuner.build_command(
        valid_config,
        {"Opt.lr_tar": 0.012},
        task=(0, 1),
        stream_seed=2025,
    )
    joined = " ".join(command)

    assert command[:2] == [sys.executable, "main_tta_0711_strict_online.py"]
    assert "Dataset=PU4D" in command
    assert "Model=ResNet18_1D_SDE" in command
    assert "++TTA0711.mode=full" in command
    assert "++TTA0711.passes=1" in command
    assert "++TTA0711.stream_seed=2025" in command
    assert "batch_size=128" in command
    assert "gpu_id=0" in command
    assert "++only_task=[0,1]" in command
    assert "Opt.lr_tar=0.012" in command
    assert "++TTA0711.physical_harmonics=8" in command
    assert "CWRU" not in joined


def test_command_rejects_protocol_override(tuner, valid_config):
    with pytest.raises(ValueError, match="not tunable"):
        tuner.build_command(
            valid_config,
            {"TTA0711.passes": 2},
            task=(0, 1),
            stream_seed=2025,
        )


def test_parse_runner_log_extracts_metrics(summary, tmp_path):
    log = tmp_path / "runner.log"
    log.write_text(
        "Task: [0, 1]: Beginning Acc T = 19.86%;\n"
        "[STRICT 0711] batches=1253, passes=1, fixed_random_stream=True\n"
        "Task: [0, 1]: Strict Online Acc = 40.1812%;\n"
        "Task: [0, 1]: Post-stream Full-Target Acc = 47.44%;\n"
        "[STRICT DIAGNOSTICS] mean_batch_ms=321.12 | peak_memory_mb=252.18\n",
        encoding="utf-8",
    )

    metrics = summary.parse_runner_log(log)

    assert metrics.before == pytest.approx(19.86)
    assert metrics.strict_online == pytest.approx(40.1812)
    assert metrics.post_stream == pytest.approx(47.44)
    assert metrics.batches == 1253
    assert metrics.runtime_seconds == pytest.approx(1253 * 0.32112)


def test_parse_runner_log_rejects_missing_metric(summary, tmp_path):
    log = tmp_path / "runner.log"
    log.write_text("Task: [0, 1]: Strict Online Acc = 40.18%;\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing"):
        summary.parse_runner_log(log)


def _record(summary, candidate, task, online, runtime=10.0):
    return {
        "candidate_id": candidate,
        "task": task,
        "overrides": {"Opt.lr_tar": 0.015},
        "metrics": summary.RunnerMetrics(
            before=20.0,
            strict_online=online,
            post_stream=online + 1.0,
            batches=1253,
            runtime_seconds=runtime,
        ),
    }


def test_partial_candidate_is_excluded(summary):
    records = [
        _record(summary, "partial", (0, 1), 41.0),
        _record(summary, "partial", (0, 3), 69.0),
    ]

    assert summary.aggregate_candidate(records, ((0, 1), (0, 3), (1, 0))) is None


def test_ranking_rejects_task_regression_and_orders_valid_candidates(summary):
    required = ((0, 1), (0, 3), (1, 0))
    records = [
        _record(summary, "safe", (0, 1), 40.60),
        _record(summary, "safe", (0, 3), 68.90),
        _record(summary, "safe", (1, 0), 66.10),
        _record(summary, "regressed", (0, 1), 39.50),
        _record(summary, "regressed", (0, 3), 70.00),
        _record(summary, "regressed", (1, 0), 67.00),
    ]
    aggregates = [
        summary.aggregate_candidate(
            [row for row in records if row["candidate_id"] == candidate], required
        )
        for candidate in ("safe", "regressed")
    ]

    ranked = summary.rank_candidates(
        [item for item in aggregates if item is not None],
        baseline={(0, 1): 40.18, (0, 3): 68.45, (1, 0): 65.58},
        minimum_gain=0.30,
        maximum_regression=0.50,
    )

    assert [candidate.candidate_id for candidate in ranked] == ["safe"]


def test_recovery_rejects_one_bad_task_even_if_other_tasks_match(summary):
    records = {
        task: summary.RunnerMetrics(
            before=0.0,
            strict_online=score,
            post_stream=0.0,
            batches=1253,
            runtime_seconds=1.0,
        )
        for task, score in summary.HISTORICAL_BASELINE.items()
    }
    records[(3, 1)] = summary.RunnerMetrics(
        before=0.0,
        strict_online=30.21,
        post_stream=0.0,
        batches=1253,
        runtime_seconds=1.0,
    )

    result = summary.validate_recovery(
        records,
        summary.HISTORICAL_BASELINE,
        expected_mean=62.7342,
        tolerance=0.05,
    )

    assert not result.passed
    assert (3, 1) in result.task_failures


def test_output_files_have_fixed_columns(summary, tmp_path):
    metrics = summary.RunnerMetrics(19.86, 40.18, 47.44, 1253, 402.4)
    records = [
        {
            "stage": "recovery",
            "candidate_id": "baseline",
            "task": (0, 1),
            "stream_seed": 2025,
            "metrics": metrics,
            "status": "succeeded",
        }
    ]
    aggregate = summary.CandidateAggregate(
        candidate_id="candidate-a",
        overrides={"Opt.lr_tar": 0.018},
        task_metrics={(0, 1): metrics},
        mean_strict_online=40.18,
        minimum_task_delta=0.0,
        wall_time_seconds=402.4,
    )
    final_rows = [
        {
            "task": (0, 1),
            "metrics": metrics,
            "baseline": 40.18,
            "status": "succeeded",
        }
    ]

    summary.write_outputs(
        tmp_path,
        records=records,
        leaderboard=[aggregate],
        final_rows=final_rows,
        report_context={"recommended": True, "exact_mean": 63.0},
    )

    assert (tmp_path / "metrics.csv").read_text(encoding="utf-8").splitlines()[0] == (
        "stage,candidate_id,source,target,stream_seed,before,strict_online,"
        "post_stream,batches,runtime_seconds,status"
    )
    assert (tmp_path / "leaderboard.csv").read_text(encoding="utf-8").splitlines()[0] == (
        "rank,candidate_id,mean_strict_online,minimum_task_delta,"
        "wall_time_seconds,overrides"
    )
    assert (tmp_path / "final_12task.csv").read_text(encoding="utf-8").splitlines()[0] == (
        "source,target,before,strict_online,post_stream,baseline,delta,status,"
        "runtime_seconds"
    )
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "# PU4D 0711 Strict-Online Tuning Report" in report
    assert "Recommended: yes" in report


def test_atomic_task_record_round_trip(tuner, tmp_path):
    path = tmp_path / "state" / "task.json"
    record = tuner.TaskRecord(
        candidate_id="abc123",
        stage="recovery",
        task=(0, 1),
        overrides={"Opt.lr_tar": 0.015},
        stream_seed=2025,
        command=["python", "runner.py"],
        status="running",
        attempt=1,
        started_at="2026-08-21T00:00:00Z",
        ended_at=None,
        returncode=None,
        log_path="logs/task.log",
        metrics=None,
        failure_class=None,
    )

    tuner.write_task_record(path, record)

    assert tuner.load_task_record(path) == record
    assert not list(path.parent.glob("*.tmp"))
    assert json.loads(path.read_text(encoding="utf-8"))["task"] == [0, 1]


@pytest.mark.parametrize(
    "text",
    [
        "CUDA initialization error",
        "CUDA driver initialization failed",
        "CUDA error: system not yet initialized",
    ],
)
def test_cuda_startup_failure_is_transient(tuner, text):
    assert tuner.classify_failure(1, text) == "transient"


@pytest.mark.parametrize(
    "text",
    [
        "FileNotFoundError: checkpoint missing",
        "ValueError: Strict 0711 requires TTA0711.passes=1",
        "AssertionError: protocol",
    ],
)
def test_protocol_and_data_failures_are_permanent(tuner, text):
    assert tuner.classify_failure(1, text) == "permanent"


def test_budget_refuses_task_inside_reserve(tuner):
    assert not tuner.can_launch(
        deadline=10_000.0,
        now=9_000.0,
        estimated_seconds=900.0,
        reserve_seconds=300.0,
    )
    assert tuner.can_launch(
        deadline=10_000.0,
        now=8_000.0,
        estimated_seconds=900.0,
        reserve_seconds=300.0,
    )


def test_should_skip_only_valid_success(tuner, tmp_path):
    log = tmp_path / "task.log"
    log.write_text(
        "Task: [0, 1]: Beginning Acc T = 19.86%;\n"
        "[STRICT 0711] batches=1253, passes=1\n"
        "Task: [0, 1]: Strict Online Acc = 40.18%;\n"
        "Task: [0, 1]: Post-stream Full-Target Acc = 47.44%;\n"
        "[STRICT DIAGNOSTICS] mean_batch_ms=321.12\n",
        encoding="utf-8",
    )
    succeeded = tuner.TaskRecord(
        candidate_id="abc123",
        stage="recovery",
        task=(0, 1),
        overrides={},
        stream_seed=2025,
        command=["python", "runner.py"],
        status="succeeded",
        attempt=1,
        started_at="2026-08-21T00:00:00Z",
        ended_at="2026-08-21T00:10:00Z",
        returncode=0,
        log_path=str(log),
        metrics={"strict_online": 40.18},
        failure_class=None,
    )

    assert tuner.should_skip(succeeded)
    log.write_text("truncated", encoding="utf-8")
    assert not tuner.should_skip(succeeded)


class FakeTaskRunner:
    def __init__(self, baseline):
        self.baseline = dict(baseline)
        self.offset_by_lr = {0.018: 0.40}
        self.offset_by_stream = {}
        self.offset_by_lr_stream = {}
        self.calls = []

    @staticmethod
    def _value(command, prefix):
        return next(item.split("=", 1)[1] for item in command if item.startswith(prefix))

    def run(self, command, log_path, env):
        task_text = self._value(command, "++only_task=").strip("[]")
        task = tuple(int(part) for part in task_text.split(","))
        stream_seed = int(self._value(command, "++TTA0711.stream_seed="))
        lr = float(self._value(command, "Opt.lr_tar="))
        online = (
            self.baseline[task]
            + self.offset_by_stream.get(stream_seed, 0.0)
            + self.offset_by_lr_stream.get((lr, stream_seed), 0.0)
            + self.offset_by_lr.get(lr, 0.0)
        )
        self.calls.append(
            {
                "task": task,
                "stream_seed": stream_seed,
                "lr": lr,
                "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
                "cuda_device_order": env.get("CUDA_DEVICE_ORDER"),
                "log_path": Path(log_path),
            }
        )
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(
            f"Task: [{task[0]}, {task[1]}]: Beginning Acc T = 20.00%;\n"
            "[STRICT 0711] batches=1253, passes=1, fixed_random_stream=True\n"
            f"Task: [{task[0]}, {task[1]}]: Strict Online Acc = {online:.4f}%;\n"
            f"Task: [{task[0]}, {task[1]}]: Post-stream Full-Target Acc = {online + 1.0:.4f}%;\n"
            "[STRICT DIAGNOSTICS] mean_batch_ms=1.00 | peak_memory_mb=1.00\n",
            encoding="utf-8",
        )
        return 0


def test_execute_task_writes_success_state_and_forces_gpu0(tuner, valid_config, tmp_path):
    runner = FakeTaskRunner(tuner.HISTORICAL_BASELINE)

    record = tuner.execute_task(
        run_dir=tmp_path,
        config=valid_config,
        stage="recovery",
        candidate=tuner.BASELINE_TUNING_OVERRIDES,
        task=(0, 1),
        stream_seed=2025,
        runner=runner,
    )

    assert record.status == "succeeded"
    assert record.metrics["strict_online"] == pytest.approx(40.18)
    assert runner.calls[0]["cuda_visible_devices"] == "0"
    assert runner.calls[0]["cuda_device_order"] == "PCI_BUS_ID"
    assert len(list((tmp_path / "state").glob("*.json"))) == 1
    assert len(list((tmp_path / "commands").glob("*.txt"))) == 1


def test_recovery_failure_prevents_search(tuner, valid_config, tmp_path):
    baseline = dict(tuner.HISTORICAL_BASELINE)
    baseline[(3, 1)] += 0.06
    runner = FakeTaskRunner(baseline)

    with pytest.raises(RuntimeError, match="recovery gate"):
        tuner.run_pipeline(valid_config, tmp_path, runner=runner)

    assert len(runner.calls) == 12
    assert not (tmp_path / "best_config.yaml").exists()


def test_coordinate_search_advances_to_complete_better_candidate(
    tuner, valid_config, tmp_path
):
    valid_config["search"]["groups"] = [
        {
            "name": "learning_rate",
            "values": [{"Opt.lr_tar": 0.015}, {"Opt.lr_tar": 0.018}],
        }
    ]
    tuner.atomic_write_json(tmp_path / "recovery_gate.json", {"passed": True})
    runner = FakeTaskRunner(tuner.HISTORICAL_BASELINE)
    for task in tuner.DEV_TASKS:
        tuner.execute_task(
            tmp_path,
            valid_config,
            stage="recovery",
            candidate=tuner.BASELINE_TUNING_OVERRIDES,
            task=task,
            stream_seed=2025,
            runner=runner,
        )
    runner.calls.clear()

    result = tuner.run_coordinate_search(
        tmp_path,
        valid_config,
        runner=runner,
        deadline=time.monotonic() + 10_000.0,
    )

    assert result["anchor"]["Opt.lr_tar"] == pytest.approx(0.018)
    assert len(runner.calls) == 3
    assert {call["task"] for call in runner.calls} == set(tuner.DEV_TASKS)


def test_coordinate_search_finalists_are_two_unique_candidates(
    tuner, valid_config, tmp_path
):
    valid_config["search"]["groups"] = [
        {
            "name": "learning_rate",
            "values": [{"Opt.lr_tar": 0.015}, {"Opt.lr_tar": 0.018}],
        },
        {
            "name": "warmup_batches",
            "values": [
                {"TTA0711.warmup_batches": 10},
                {"TTA0711.warmup_batches": 25},
            ],
        },
    ]
    tuner.atomic_write_json(tmp_path / "recovery_gate.json", {"passed": True})
    runner = FakeTaskRunner(tuner.HISTORICAL_BASELINE)
    runner.offset_by_lr[0.018] = 0.8
    for task in tuner.DEV_TASKS:
        tuner.execute_task(
            tmp_path,
            valid_config,
            stage="recovery",
            candidate=tuner.BASELINE_TUNING_OVERRIDES,
            task=task,
            stream_seed=2025,
            runner=runner,
        )

    result = tuner.run_coordinate_search(
        tmp_path,
        valid_config,
        runner=runner,
        deadline=time.monotonic() + 10_000.0,
    )

    markers = {
        json.dumps(candidate, sort_keys=True, separators=(",", ":"))
        for candidate in result["finalists"]
    }
    assert len(result["finalists"]) == 2
    assert len(markers) == 2


def _seed_2025_development_records(tuner, runner, run_dir, config, candidates):
    for candidate in candidates:
        for task in tuner.DEV_TASKS:
            tuner.execute_task(
                run_dir,
                config,
                stage="search_seed",
                candidate=candidate,
                task=task,
                stream_seed=2025,
                runner=runner,
            )
    runner.calls.clear()


def test_stability_selection_uses_stream_seed_2026(tuner, valid_config, tmp_path):
    runner = FakeTaskRunner(tuner.HISTORICAL_BASELINE)
    finalists = [
        dict(tuner.BASELINE_TUNING_OVERRIDES),
        {**tuner.BASELINE_TUNING_OVERRIDES, "Opt.lr_tar": 0.018},
    ]
    _seed_2025_development_records(tuner, runner, tmp_path, valid_config, finalists)

    selected = tuner.run_stability_selection(
        tmp_path, valid_config, finalists=finalists, runner=runner
    )

    assert selected["Opt.lr_tar"] == pytest.approx(0.018)
    assert {call["stream_seed"] for call in runner.calls} == {2026}
    assert len(runner.calls) == 6


def test_stability_selection_compares_matching_2026_baseline(
    tuner, valid_config, tmp_path
):
    runner = FakeTaskRunner(tuner.HISTORICAL_BASELINE)
    runner.offset_by_stream[2026] = -1.0
    finalists = [
        dict(tuner.BASELINE_TUNING_OVERRIDES),
        {**tuner.BASELINE_TUNING_OVERRIDES, "Opt.lr_tar": 0.018},
    ]
    _seed_2025_development_records(tuner, runner, tmp_path, valid_config, finalists)

    selected = tuner.run_stability_selection(
        tmp_path, valid_config, finalists=finalists, runner=runner
    )

    assert selected["Opt.lr_tar"] == pytest.approx(0.018)


def test_stability_selection_ranks_gain_across_both_stream_seeds(
    tuner, valid_config, tmp_path
):
    runner = FakeTaskRunner(tuner.HISTORICAL_BASELINE)
    runner.offset_by_lr = {}
    runner.offset_by_lr_stream = {
        (0.018, 2025): 0.80,
        (0.018, 2026): 0.20,
        (0.024, 2025): 0.40,
        (0.024, 2026): 0.50,
    }
    finalists = [
        dict(tuner.BASELINE_TUNING_OVERRIDES),
        {**tuner.BASELINE_TUNING_OVERRIDES, "Opt.lr_tar": 0.018},
        {**tuner.BASELINE_TUNING_OVERRIDES, "Opt.lr_tar": 0.024},
    ]
    _seed_2025_development_records(tuner, runner, tmp_path, valid_config, finalists)

    selected = tuner.run_stability_selection(
        tmp_path, valid_config, finalists=finalists, runner=runner
    )

    assert selected["Opt.lr_tar"] == pytest.approx(0.018)


def test_final_stage_uses_only_heldout_tasks_and_does_not_rewrite_freeze(
    tuner, valid_config, tmp_path
):
    runner = FakeTaskRunner(tuner.HISTORICAL_BASELINE)
    frozen = {**tuner.BASELINE_TUNING_OVERRIDES, "Opt.lr_tar": 0.018}
    tuner.atomic_write_json(tmp_path / "recovery_gate.json", {"passed": True})
    tuner.freeze_best_config(
        tmp_path / "best_config.yaml",
        frozen,
        {"selected_by": "test", "frozen_at": "2026-08-21T00:00:00Z"},
    )
    before = (tmp_path / "best_config.yaml").read_bytes()

    records = tuner.run_final_validation(
        tmp_path, valid_config, frozen_overrides=frozen, runner=runner
    )

    assert tuple(record.task for record in records) == tuner.HELDOUT_TASKS
    assert tuple(call["task"] for call in runner.calls) == tuner.HELDOUT_TASKS
    assert (tmp_path / "best_config.yaml").read_bytes() == before


def test_freeze_metadata_records_both_streams_and_audited_inputs(
    tuner, valid_config, tmp_path
):
    runner = FakeTaskRunner(tuner.HISTORICAL_BASELINE)
    selected = {**tuner.BASELINE_TUNING_OVERRIDES, "Opt.lr_tar": 0.018}
    tuner.atomic_write_json(tmp_path / "recovery_gate.json", {"passed": True})
    for stream_seed in (2025, 2026):
        for task in tuner.DEV_TASKS:
            tuner.execute_task(
                tmp_path,
                valid_config,
                stage="search" if stream_seed == 2025 else "stability",
                candidate=selected,
                task=task,
                stream_seed=stream_seed,
                runner=runner,
            )
    preflight = {
        "config_sha256": "config-hash",
        "checkpoints": [{"source": 0, "sha256": "checkpoint-hash"}],
    }

    metadata = tuner.build_freeze_metadata(tmp_path, selected, preflight)

    assert metadata["source_seed"] == 2025
    assert metadata["stream_seeds"] == [2025, 2026]
    assert set(metadata["selection_metrics"]) == {"2025", "2026"}
    assert metadata["config_sha256"] == "config-hash"
    assert metadata["checkpoints"] == preflight["checkpoints"]


def test_generate_run_outputs_combines_frozen_dev_and_heldout_results(
    summary, tmp_path
):
    selected = {"Opt.lr_tar": 0.024, "TTA0711.warp_lr_scale": 0.2}
    (tmp_path / "state").mkdir()
    (tmp_path / "best_config.yaml").write_text(
        json.dumps(
            {
                "overrides": selected,
                "metadata": {
                    "frozen_at": "2026-08-21T10:00:00Z",
                    "stream_seeds": [2025, 2026],
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "recovery_gate.json").write_text(
        json.dumps({"passed": True, "exact_mean": 62.73}), encoding="utf-8"
    )
    (tmp_path / "preflight.json").write_text(
        json.dumps(
            {
                "passed": True,
                "config_sha256": "config-hash",
                "gpu0": "0, test GPU",
                "checkpoints": [
                    {"source": 0, "path": "source0.pt", "sha256": "abc"}
                ],
                "environment": {"python": "3.x", "pytorch": "2.x", "cuda": "12.x"},
            }
        ),
        encoding="utf-8",
    )

    def write_record(name, *, stage, task, overrides, score, stream_seed=2025):
        payload = {
            "candidate_id": name.split("_", 1)[0],
            "stage": stage,
            "task": list(task),
            "overrides": overrides,
            "stream_seed": stream_seed,
            "command": [
                "python",
                "runner.py",
                "Dataset=PU4D",
                "gpu_id=0",
                "batch_size=128",
                "++TTA0711.mode=full",
                "++TTA0711.passes=1",
                "++TTA0711.stream_seed=2025",
            ],
            "status": "succeeded",
            "attempt": 1,
            "started_at": "2026-08-21T10:01:00Z",
            "ended_at": "2026-08-21T10:02:00Z",
            "returncode": 0,
            "log_path": f"logs/{name}.log",
            "metrics": {
                "before": score - 1.0,
                "strict_online": score,
                "post_stream": score + 1.0,
                "batches": 100,
                "runtime_seconds": 60.0,
            },
            "failure_class": None,
        }
        (tmp_path / "state" / f"{name}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    for index, (task, baseline) in enumerate(summary.HISTORICAL_BASELINE.items()):
        write_record(
            f"baseline_{index}",
            stage="recovery",
            task=task,
            overrides={"Opt.lr_tar": 0.015},
            score=baseline,
        )
        write_record(
            f"selected_{index}",
            stage="search" if task in ((0, 1), (0, 3), (1, 0)) else "final",
            task=task,
            overrides=selected,
            score=baseline + 1.0,
        )

    result = summary.generate_run_outputs(
        tmp_path, regression_tests_passed=True
    )

    assert result["recommended"] is True
    assert result["task_count"] == 12
    assert result["exact_mean"] > 62.7342
    assert len((tmp_path / "final_12task.csv").read_text().splitlines()) == 13
    assert "Protocol invariants: PASS" in (tmp_path / "report.md").read_text()
    assert "Frozen configuration" in (tmp_path / "report.md").read_text()
    assert "Checkpoint hashes" in (tmp_path / "report.md").read_text()
    final_result = json.loads((tmp_path / "final_result.json").read_text())
    assert final_result["recommended"] is True
    assert final_result["exact_12task_mean"] == pytest.approx(result["exact_mean"])
    assert final_result["heldout_mean"] > 62.7342
