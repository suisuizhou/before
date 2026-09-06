# PU4D 0711 Strict-Online Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Reproduce the complete 12-task PU4D 0711 strict-online baseline and then find and validate one universal strict-online configuration whose mean Strict Online Accuracy exceeds 62.7342%.

**Architecture:** Add an external, resumable tuning layer around main_tta_0711_strict_online.py. The layer owns schema validation, immutable protocol injection, deterministic candidate expansion, subprocess execution, atomic task state, metric parsing, ranking, recovery gating, configuration freezing, and final reporting; it does not modify the 0711 algorithm runner.

**Tech Stack:** Python 3.13, PyYAML 6, pytest 9, Hydra command-line overrides, Bash, PyTorch runner subprocesses, CSV and JSON artifacts.

**Spec:** docs/superpowers/specs/2026-08-21-pu4d-0711-strict-tuning-design.md

## Global Constraints

- Dataset is PU4D from Dataset/PU4D_CACHE; CWRU must never appear in a generated command.
- Use main_tta_0711_strict_online.py and one universal configuration for all 12 transfer tasks.
- Force TTA0711.mode=full, TTA0711.passes=1, pre-update online scoring, fixed random target stream, and no sample revisit.
- Force ResNet18_1D_SDE, batch size 128, source seed 2025, and the four audited source checkpoints.
- Freeze backbone and classifier; only band_scale, band_bias, and warp_ctrl may be trainable.
- Target labels may affect only the accuracy meter and offline candidate ranking.
- Keep sampling rate, FFT size, physical fault frequencies, harmonics, sideband definitions, spectrum length, view strengths, adapter bounds, and regularization constants fixed.
- Use physical GPU 0 only. Do not launch, claim, inspect processes on, or terminate work on GPU 1 or GPU 2.
- Do not overwrite baseline runners, source checkpoints, historical logs, recovery backups, or unrelated dirty-worktree files.
- Complete the 12-task recovery gate before starting the independent 24-hour tuning budget.
- Recovery passes only when every task is within 0.05 percentage points of its historical Strict Online score and the mean is within 0.05 points of 62.7342%.
- A tuned configuration is recommended only after one frozen universal configuration completes all 12 tasks and its exact mean exceeds 62.7342%.

---

### Task 1: Declarative tuning configuration and strict schema

**Files:**
- Create: Configs/Experiments/PU4D0711_strict_tuning.yaml
- Create: tools/tune_pu4d_0711_strict.py
- Create: tests/test_pu4d_0711_tuning.py

**Interfaces:**
- Produces: load_config(path: Path) -> dict[str, object]
- Produces: validate_config(config: Mapping[str, object]) -> None
- Produces: HISTORICAL_BASELINE: dict[tuple[int, int], float]
- Produces: DEV_TASKS and HELDOUT_TASKS as immutable task tuples

- [ ] **Step 1: Write failing schema tests**

Add tests that load the module with importlib and assert the real YAML:

~~~python
def test_real_config_is_valid(tuner):
    cfg = tuner.load_config(Path("Configs/Experiments/PU4D0711_strict_tuning.yaml"))
    tuner.validate_config(cfg)
    assert cfg["protocol"]["dataset"] == "PU4D"
    assert cfg["protocol"]["runner"] == "main_tta_0711_strict_online.py"
    assert cfg["protocol"]["gpu"] == 0
    assert cfg["recovery"]["expected_mean"] == pytest.approx(62.7342)
    assert tuple(map(tuple, cfg["tasks"]["development"])) == ((0, 1), (0, 3), (1, 0))


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
    valid_config["search"]["groups"][0]["values"][0]["TTA0711.physical_harmonics"] = 9
    with pytest.raises(ValueError, match="not tunable"):
        tuner.validate_config(valid_config)
~~~

- [ ] **Step 2: Run schema tests and verify failure**

Run: python -m pytest tests/test_pu4d_0711_tuning.py -k "config or protocol" -vv

Expected: collection or import fails because the module and YAML do not exist.

- [ ] **Step 3: Create the exact YAML**

Define:

~~~yaml
version: 1
protocol:
  dataset: PU4D
  data_path: Dataset/PU4D_CACHE
  runner: main_tta_0711_strict_online.py
  model: ResNet18_1D_SDE
  model_type: linear
  batch_size: 128
  num_workers: 4
  source_seed: 2025
  stream_seed: 2025
  mode: full
  passes: 1
  gpu: 0
recovery:
  expected_mean: 62.7342
  tolerance: 0.05
  expected:
    "0to1": 40.18
    "0to2": 94.93
    "0to3": 68.45
    "1to0": 65.58
    "1to2": 63.18
    "1to3": 45.65
    "2to0": 95.61
    "2to1": 43.52
    "2to3": 70.20
    "3to0": 67.09
    "3to1": 30.15
    "3to2": 68.27
tasks:
  development: [[0, 1], [0, 3], [1, 0]]
  heldout: [[0, 2], [1, 2], [1, 3], [2, 0], [2, 1], [2, 3], [3, 0], [3, 1], [3, 2]]
budget:
  tuning_hours: 24
  estimated_task_minutes: 16
  reserve_minutes: 30
selection:
  minimum_mean_gain: 0.30
  maximum_task_regression: 0.50
  stability_stream_seed: 2026
search:
  groups:
    - name: learning_rate
      values:
        - {Opt.lr_tar: 0.008}
        - {Opt.lr_tar: 0.012}
        - {Opt.lr_tar: 0.015}
        - {Opt.lr_tar: 0.018}
        - {Opt.lr_tar: 0.024}
    - name: warp_lr_scale
      values:
        - {TTA0711.warp_lr_scale: 0.05}
        - {TTA0711.warp_lr_scale: 0.10}
        - {TTA0711.warp_lr_scale: 0.20}
    - name: ema_beta
      values:
        - {TTA0711.ema_beta: 0.990}
        - {TTA0711.ema_beta: 0.995}
        - {TTA0711.ema_beta: 0.999}
    - name: warmup_batches
      values:
        - {TTA0711.warmup_batches: 0}
        - {TTA0711.warmup_batches: 10}
        - {TTA0711.warmup_batches: 25}
    - name: aux_ramp_batches
      values:
        - {TTA0711.aux_ramp_batches: 10}
        - {TTA0711.aux_ramp_batches: 20}
        - {TTA0711.aux_ramp_batches: 40}
    - name: min_reliability
      values:
        - {TTA0711.min_reliability: 0.10}
        - {TTA0711.min_reliability: 0.20}
        - {TTA0711.min_reliability: 0.30}
    - name: loss_profile
      values:
        - {TTA0711.lambda_mt: 0.01, TTA0711.lambda_pcl: 0.01, TTA0711.lambda_ncl: 0.005}
        - {TTA0711.lambda_mt: 0.02, TTA0711.lambda_pcl: 0.02, TTA0711.lambda_ncl: 0.01}
        - {TTA0711.lambda_mt: 0.04, TTA0711.lambda_pcl: 0.04, TTA0711.lambda_ncl: 0.02}
    - name: memory_per_class
      values:
        - {TTA0711.memory_per_class: 32}
        - {TTA0711.memory_per_class: 64}
        - {TTA0711.memory_per_class: 128}
    - name: contrastive_temperature
      values:
        - {TTA0711.pcl_temperature: 0.10, TTA0711.ncl_temperature: 0.10}
        - {TTA0711.pcl_temperature: 0.20, TTA0711.ncl_temperature: 0.20}
        - {TTA0711.pcl_temperature: 0.30, TTA0711.ncl_temperature: 0.30}
~~~

- [ ] **Step 4: Implement minimal schema validation**

Use yaml.safe_load, require exact top-level keys, validate task partition equals all 12 directed PU4D pairs, validate immutable values exactly, and reject candidate keys outside:

~~~python
TUNABLE_KEYS = frozenset({
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
})
~~~

- [ ] **Step 5: Run schema tests**

Run: python -m pytest tests/test_pu4d_0711_tuning.py -k "config or protocol" -vv

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

~~~bash
git add Configs/Experiments/PU4D0711_strict_tuning.yaml tools/tune_pu4d_0711_strict.py tests/test_pu4d_0711_tuning.py
git commit -m "feat: define strict PU4D 0711 tuning schema"
~~~

### Task 2: Deterministic candidates and protocol-safe commands

**Files:**
- Modify: tools/tune_pu4d_0711_strict.py
- Modify: tests/test_pu4d_0711_tuning.py

**Interfaces:**
- Consumes: validated configuration from Task 1
- Produces: normalize_overrides(overrides: Mapping[str, object]) -> dict[str, object]
- Produces: candidate_id(overrides, source_seed, stream_seed) -> str
- Produces: build_command(config, overrides, task, stream_seed) -> list[str]
- Produces: expand_coordinate_group(anchor, values) -> list[dict[str, object]]

- [ ] **Step 1: Write failing determinism and command tests**

~~~python
def test_candidate_id_is_order_independent(tuner):
    left = {"Opt.lr_tar": 0.012, "TTA0711.ema_beta": 0.995}
    right = {"TTA0711.ema_beta": 0.995, "Opt.lr_tar": 0.012}
    assert tuner.candidate_id(left, 2025, 2025) == tuner.candidate_id(right, 2025, 2025)


def test_command_forces_strict_invariants(tuner, valid_config):
    cmd = tuner.build_command(
        valid_config,
        {"Opt.lr_tar": 0.012},
        task=(0, 1),
        stream_seed=2025,
    )
    joined = " ".join(cmd)
    assert cmd[:2] == [sys.executable, "main_tta_0711_strict_online.py"]
    assert "Dataset=PU4D" in cmd
    assert "++TTA0711.mode=full" in cmd
    assert "++TTA0711.passes=1" in cmd
    assert "++TTA0711.stream_seed=2025" in cmd
    assert "batch_size=128" in cmd
    assert "++only_task=[0,1]" in cmd
    assert "CWRU" not in joined


def test_command_rejects_protocol_override(tuner, valid_config):
    with pytest.raises(ValueError, match="not tunable"):
        tuner.build_command(valid_config, {"TTA0711.passes": 2}, (0, 1), 2025)
~~~

- [ ] **Step 2: Run tests and verify failure**

Run: python -m pytest tests/test_pu4d_0711_tuning.py -k "candidate_id or command or coordinate" -vv

Expected: failures report missing functions.

- [ ] **Step 3: Implement deterministic identifiers and commands**

Serialize normalized overrides with json.dumps(sort_keys=True, separators=(",", ":")), hash with SHA-256, and use the first 12 hex characters. Build argv as a list, never as shell text. Inject every immutable override from run_0711_strict_remaining9_gpu01.sh, including physical evidence constants, view strengths, optimizer regularization, and source checkpoint routing.

- [ ] **Step 4: Run candidate and command tests**

Run: python -m pytest tests/test_pu4d_0711_tuning.py -k "candidate_id or command or coordinate" -vv

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

~~~bash
git add tools/tune_pu4d_0711_strict.py tests/test_pu4d_0711_tuning.py
git commit -m "feat: build protocol-safe 0711 tuning commands"
~~~

### Task 3: Metrics, aggregation, ranking, and recovery gate

**Files:**
- Create: tools/summarize_pu4d_0711_tuning.py
- Modify: tests/test_pu4d_0711_tuning.py

**Interfaces:**
- Produces: RunnerMetrics dataclass with before, strict_online, post_stream, batches, runtime_seconds
- Produces: parse_runner_log(path: Path) -> RunnerMetrics
- Produces: aggregate_candidate(records, required_tasks) -> CandidateAggregate | None
- Produces: rank_candidates(aggregates, baseline, minimum_gain, maximum_regression) -> list[CandidateAggregate]
- Produces: validate_recovery(records, expected, expected_mean, tolerance) -> RecoveryResult

- [ ] **Step 1: Write failing parser and ranking tests**

~~~python
def test_parse_runner_log_extracts_unrounded_metrics(summary, tmp_path):
    log = tmp_path / "runner.log"
    log.write_text(
        "Task: [0, 1]: Beginning Acc T = 19.86%;\n"
        "Task: [0, 1]: Strict Online Acc = 40.1812%;\n"
        "Task: [0, 1]: Post-stream Full-Target Acc = 47.44%;\n"
        "[STRICT 0711] batches=1253, passes=1, fixed_random_stream=True\n"
        "[STRICT DIAGNOSTICS] mean_batch_ms=321.12 | peak_memory_mb=252.18\n",
        encoding="utf-8",
    )
    metrics = summary.parse_runner_log(log)
    assert metrics.strict_online == pytest.approx(40.1812)
    assert metrics.batches == 1253


def test_partial_candidate_is_excluded(summary):
    records = metric_records(tasks=[(0, 1), (0, 3)])
    assert summary.aggregate_candidate(records, ((0, 1), (0, 3), (1, 0))) is None


def test_recovery_rejects_one_bad_task_even_if_mean_matches(summary, baseline_records):
    baseline_records[(3, 1)].strict_online += 0.06
    result = summary.validate_recovery(
        baseline_records, summary.HISTORICAL_BASELINE, 62.7342, 0.05
    )
    assert not result.passed
    assert (3, 1) in result.task_failures
~~~

- [ ] **Step 2: Run parser and ranking tests and verify failure**

Run: python -m pytest tests/test_pu4d_0711_tuning.py -k "parse or aggregate or rank or recovery" -vv

Expected: import or missing-symbol failures.

- [ ] **Step 3: Implement parser, aggregation, guards, and ordering**

Use the last matching metric line. Require Before, Strict Online, Post-stream, and batch count. Rank complete candidates by descending development mean, then descending minimum task delta, then ascending wall time. Reject a finalist when mean gain is below 0.30 or any development task delta is below -0.50. Recovery requires all 12 tasks and checks each task plus the exact mean.

- [ ] **Step 4: Add CSV and Markdown output tests**

Assert metrics.csv, leaderboard.csv, final_12task.csv, and report.md have fixed columns and deterministic task order.

- [ ] **Step 5: Run parser and report tests**

Run: python -m pytest tests/test_pu4d_0711_tuning.py -k "parse or aggregate or rank or recovery or report" -vv

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

~~~bash
git add tools/summarize_pu4d_0711_tuning.py tests/test_pu4d_0711_tuning.py
git commit -m "feat: rank strict 0711 tuning results"
~~~

### Task 4: Atomic state, retry policy, resume, and budget

**Files:**
- Modify: tools/tune_pu4d_0711_strict.py
- Modify: tests/test_pu4d_0711_tuning.py

**Interfaces:**
- Produces: TaskRecord dataclass
- Produces: atomic_write_json(path: Path, value: Mapping[str, object]) -> None
- Produces: load_task_record(path: Path) -> TaskRecord
- Produces: classify_failure(returncode: int, log_text: str) -> Literal["transient", "permanent"]
- Produces: can_launch(deadline, now, estimated_seconds, reserve_seconds) -> bool
- Produces: should_skip(record, log_path) -> bool

- [ ] **Step 1: Write failing state-machine tests**

Cover pending to running to succeeded, stale running to resumed attempt, one transient retry, no permanent retry, successful record with missing metric is invalid, atomic replacement leaves valid JSON, and budget refusal.

~~~python
def test_budget_refuses_task_inside_reserve(tuner):
    assert not tuner.can_launch(
        deadline=10_000.0,
        now=9_000.0,
        estimated_seconds=900.0,
        reserve_seconds=300.0,
    )


@pytest.mark.parametrize("text", ["CUDA initialization error", "CUDA driver error"])
def test_cuda_startup_failure_is_transient(tuner, text):
    assert tuner.classify_failure(1, text) == "transient"


@pytest.mark.parametrize("text", ["FileNotFoundError", "ValueError: Strict 0711 requires", "AssertionError"])
def test_protocol_and_data_failures_are_permanent(tuner, text):
    assert tuner.classify_failure(1, text) == "permanent"
~~~

- [ ] **Step 2: Run state tests and verify failure**

Run: python -m pytest tests/test_pu4d_0711_tuning.py -k "state or retry or budget or skip" -vv

Expected: missing-symbol failures.

- [ ] **Step 3: Implement state and retry primitives**

Write JSON to a sibling temporary file, flush, fsync, and os.replace. Store candidate id, stage, task, overrides, stream seed, command argv, timestamps, status, attempt, return code, log path, parsed metrics, and failure class. Treat running records as stale only when no matching process is owned by the orchestrator; never signal another process.

- [ ] **Step 4: Run state tests**

Run: python -m pytest tests/test_pu4d_0711_tuning.py -k "state or retry or budget or skip" -vv

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

~~~bash
git add tools/tune_pu4d_0711_strict.py tests/test_pu4d_0711_tuning.py
git commit -m "feat: persist resumable 0711 tuning state"
~~~

### Task 5: Recovery, coordinate search, stability, freeze, and final stages

**Files:**
- Modify: tools/tune_pu4d_0711_strict.py
- Modify: tools/summarize_pu4d_0711_tuning.py
- Modify: tests/test_pu4d_0711_tuning.py

**Interfaces:**
- Produces: execute_task(run_dir, config, stage, candidate, task, stream_seed) -> TaskRecord
- Produces: run_recovery_gate(...) -> RecoveryResult
- Produces: run_coordinate_search(...) -> dict[str, object]
- Produces: run_stability_selection(...) -> dict[str, object]
- Produces: freeze_best_config(path, overrides, metadata) -> None
- Produces: run_final_validation(...) -> list[TaskRecord]

- [ ] **Step 1: Write failing orchestration tests with a fake subprocess**

Test exact stage order, recovery failure preventing candidate launch, anchor advancement only after three complete development tasks, failed candidate exclusion, stream seed 2026 stability runs, immutable best_config.yaml after held-out scores, and nine held-out tasks only during final validation.

~~~python
def test_recovery_failure_prevents_search(tuner, fake_runner, valid_config, tmp_path):
    fake_runner.metrics[(3, 1)] = 31.00
    with pytest.raises(RuntimeError, match="recovery gate"):
        tuner.run_pipeline(valid_config, tmp_path, runner=fake_runner)
    assert fake_runner.candidate_calls == []


def test_final_stage_uses_only_heldout_tasks(tuner, fake_runner, valid_config, tmp_path):
    fake_runner.seed_successful_recovery()
    tuner.run_final_validation(
        run_dir=tmp_path,
        config=valid_config,
        frozen_overrides={"Opt.lr_tar": 0.018},
        runner=fake_runner,
    )
    assert tuple(fake_runner.tasks) == tuple(map(tuple, valid_config["tasks"]["heldout"]))
~~~

- [ ] **Step 2: Run orchestration tests and verify failure**

Run: python -m pytest tests/test_pu4d_0711_tuning.py -k "pipeline or stage or freeze or final" -vv

Expected: missing-symbol failures.

- [ ] **Step 3: Implement subprocess execution**

Use subprocess.Popen with stdout and stderr redirected to the candidate-task log. Set CUDA_VISIBLE_DEVICES=0 in a copied environment. Write the shell-safe command record with shlex.join for inspection, but execute the argv list directly. Parse metrics only after return code 0.

- [ ] **Step 4: Implement the four stages**

Recovery runs all 12 baseline tasks at stream seed 2025. Coordinate search reuses recovery development results for the anchor and searches groups in YAML order. Stability evaluates baseline and top two finalists at stream seed 2026. Freeze writes best_config.yaml before final held-out launch. Final validation runs only the nine held-out tasks and combines them with frozen development results.

- [ ] **Step 5: Run orchestration tests**

Run: python -m pytest tests/test_pu4d_0711_tuning.py -k "pipeline or stage or freeze or final" -vv

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

~~~bash
git add tools/tune_pu4d_0711_strict.py tools/summarize_pu4d_0711_tuning.py tests/test_pu4d_0711_tuning.py
git commit -m "feat: orchestrate strict 0711 tuning stages"
~~~

### Task 6: Preflight launcher and dry-run contract

**Files:**
- Create: run_pu4d_0711_tuning.sh
- Modify: tests/test_pu4d_0711_tuning.py

**Interfaces:**
- Consumes: command-line interface from tools/tune_pu4d_0711_strict.py
- Produces: one-command preflight and launch on physical GPU 0

- [ ] **Step 1: Write failing CLI and launcher tests**

Assert --dry-run creates a timestamped run directory with commands for 12 recovery tasks, no subprocess training calls, no CWRU token, GPU 0 only, and all strict invariants. Assert the launcher references focused tests, py_compile, bash syntax, PU4D cache, all four source checkpoints, and nvidia-smi GPU 0.

- [ ] **Step 2: Run CLI tests and verify failure**

Run: python -m pytest tests/test_pu4d_0711_tuning.py -k "cli or dry_run or launcher" -vv

Expected: CLI or launcher does not exist.

- [ ] **Step 3: Add CLI and Bash launcher**

The Python CLI accepts --config, --run-dir, --dry-run, --stage, --task, and --resume. The optional --task value uses the exact form 0to1 and is valid only for a recovery probe; every other stage rejects it. The shell launcher uses set -euo pipefail, resolves the project directory from the script path, runs focused tests and syntax checks, validates PU4D_CACHE and four audited checkpoints, verifies physical GPU 0, creates logs/PU4D_0711_STRICT_TUNING_<UTC timestamp>, and invokes the Python CLI without nohup.

- [ ] **Step 4: Run focused verification**

Run: python -m pytest tests/test_pu4d_0711_tuning.py -vv

Run: python -m py_compile tools/tune_pu4d_0711_strict.py tools/summarize_pu4d_0711_tuning.py

Run: bash -n run_pu4d_0711_tuning.sh

Run: ./run_pu4d_0711_tuning.sh --dry-run

Expected: all tests and syntax checks pass; dry-run writes only state and command artifacts.

- [ ] **Step 5: Commit**

~~~bash
git add run_pu4d_0711_tuning.sh tools/tune_pu4d_0711_strict.py tests/test_pu4d_0711_tuning.py
git commit -m "feat: launch strict PU4D 0711 tuning"
~~~

### Task 7: Real PU4D recovery gate

**Files:**
- Create at runtime: logs/PU4D_0711_STRICT_TUNING_<timestamp>/
- Do not modify source code during this task.

**Interfaces:**
- Consumes: verified launcher and audited source checkpoints
- Produces: recovery metrics for all 12 tasks and recovery_report.md

- [ ] **Step 1: Record environment and checkpoint hashes**

Run the launcher preflight and save Python, PyTorch, CUDA, GPU 0, Git HEAD, dirty-worktree summary, configuration hash, and the four source checkpoint SHA-256 hashes in the run directory.

- [ ] **Step 2: Run the 0→1 recovery probe**

Run: ./run_pu4d_0711_tuning.sh --stage recovery --task 0to1

Expected: Strict Online Accuracy is 40.18% ±0.05, passes=1, fixed random stream seed 2025, 1253 batches, and trainable-parameter output contains only band_scale, band_bias, and warp_ctrl.

- [ ] **Step 3: Run the complete recovery gate**

Run: ./run_pu4d_0711_tuning.sh --stage recovery --resume

Expected: all remaining 11 tasks succeed, every task is within ±0.05 of the YAML historical value, and exact mean is 62.7342% ±0.05.

- [ ] **Step 4: Stop on recovery failure**

If the gate fails, preserve artifacts, write recovery_report.md with task deltas and diagnostics, invoke superpowers:systematic-debugging, and do not launch tuning candidates.

- [ ] **Step 5: Mark recovery complete**

Only after the gate passes, atomically write recovery_gate.json with passed=true, completed task records, exact mean, task deltas, and timestamp.

### Task 8: Real tuning, freeze, final validation, and regression verification

**Files:**
- Create at runtime: state/, commands/, logs/, metrics.csv, leaderboard.csv, best_config.yaml, final_12task.csv, report.md under the run directory
- Do not overwrite main_tta_0711_strict_online.py or source checkpoints.

**Interfaces:**
- Consumes: passed recovery_gate.json from Task 7
- Produces: frozen universal configuration and complete 12-task report

- [ ] **Step 1: Start or resume the 24-hour coordinate search**

Run: ./run_pu4d_0711_tuning.sh --stage tune --resume

Expected: GPU 0 only; each group retains the anchor; only complete three-task candidates enter ranking; no launch occurs inside the configured deadline reserve.

- [ ] **Step 2: Run stability selection**

Run: ./run_pu4d_0711_tuning.sh --stage stability --resume

Expected: baseline and top two finalists run on the three development tasks with stream seed 2026; regression guards select a finalist or retain baseline.

- [ ] **Step 3: Inspect the frozen configuration before held-out launch**

Verify best_config.yaml contains one universal override mapping, selection metrics for both stream seeds, source checkpoint hashes, configuration hash, and frozen_at timestamp. Verify no held-out task record predates frozen_at.

- [ ] **Step 4: Run final held-out validation**

Run: ./run_pu4d_0711_tuning.sh --stage final --resume

Expected: exactly nine held-out tasks run with stream seed 2025 and combine with the three frozen development scores.

- [ ] **Step 5: Run fresh focused and PU4D regression tests**

Run: python -m pytest tests/test_pu4d_0711_tuning.py tests/test_fixed_random_stream.py tests/test_adaptation_ema.py tests/test_physical_fault_evidence.py tests/test_pu4d_vanilla_protocol.py tests/test_pu4d_vanilla_runner_contracts.py tests/test_runner_contracts.py -q

Expected: all tests pass.

- [ ] **Step 6: Generate and audit the final report**

Run: python tools/summarize_pu4d_0711_tuning.py <run-directory>

Verify report.md includes the recovery table, full search history, rejected candidates, failure records, frozen configuration, 12-task Before/Strict Online/Post-stream table, exact and rounded means, baseline delta, checkpoint hashes, environment, protocol invariants, and recommendation decision.

- [ ] **Step 7: Apply the acceptance criterion**

Mark recommended=true only when all 12 frozen-config tasks succeeded, exact mean Strict Online Accuracy is greater than 62.7342%, strict invariants passed, and regression tests passed. Otherwise mark recommended=false and retain the 0.015 baseline as official.

- [ ] **Step 8: Commit implementation artifacts but not bulky runtime logs**

~~~bash
git add Configs/Experiments/PU4D0711_strict_tuning.yaml tools/tune_pu4d_0711_strict.py tools/summarize_pu4d_0711_tuning.py tests/test_pu4d_0711_tuning.py run_pu4d_0711_tuning.sh docs/superpowers/plans/2026-08-21-pu4d-0711-strict-tuning.md
git commit -m "feat: complete strict PU4D 0711 tuning workflow"
~~~
