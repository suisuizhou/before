# PU4D 0711 Strict-Online Tuning Design

Date: 2026-08-21 (UTC)

## Goal

Improve the PU4D 12-task mean Strict Online Accuracy of the 0711 method beyond the reproduced 62.73% baseline within a 24-hour tuning budget, while preserving the method and strict-online protocol defined by `0711方法系统概述.md`.

## Baseline

The comparison baseline is the existing ResNet18_1D_SDE strict-online run with source checkpoints trained at seed 2025. Its 12 task scores sum to 752.81 percentage points, for a mean of 62.7342%:

| Task | Strict Online |
|---|---:|
| 0→1 | 40.18% |
| 0→2 | 94.93% |
| 0→3 | 68.45% |
| 1→0 | 65.58% |
| 1→2 | 63.18% |
| 1→3 | 45.65% |
| 2→0 | 95.61% |
| 2→1 | 43.52% |
| 2→3 | 70.20% |
| 3→0 | 67.09% |
| 3→1 | 30.15% |
| 3→2 | 68.27% |

The development tasks are 0→1, 0→3, and 1→0. Their baseline mean is 58.07%.

Before any tuning candidate is evaluated, rerun the baseline on all 12 PU4D
transfer tasks with the audited source checkpoints and `stream_seed=2025`.
Every task must reproduce its historical Strict Online Accuracy within 0.05
percentage points, and the exact 12-task mean must reproduce 62.7342% within
0.05 percentage points. This recovery gate is a prerequisite outside the
24-hour tuning budget. If it fails, stop and diagnose recovery; do not tune.

## Non-Negotiable Protocol Constraints

- Use `main_tta_0711_strict_online.py` and one universal configuration for all 12 transfer tasks.
- Keep `TTA0711.mode=full`, `TTA0711.passes=1`, pre-update online scoring, a fixed random target stream, and no revisit of prior or future target batches.
- Keep ResNet18_1D_SDE, batch size 128, source seed 2025, and the four already audited source checkpoints fixed.
- Freeze the backbone and classifier. Only `band_scale`, `band_bias`, and `warp_ctrl` remain trainable.
- Target labels may be consumed by the existing accuracy meter and by the offline development-task scorer only. They must never affect gradients, pseudo-labels, routing, memory, early stopping, or per-batch parameter choices.
- Do not tune sampling rate, FFT size, physical fault frequencies, harmonics, sideband definitions, or spectrum length.
- Do not overwrite the baseline runner, checkpoints, logs, or recovery artifacts.
- Use GPU 0 by default. Do not claim or terminate work on GPU 1 or GPU 2.

## Architecture

The tuning system is an external orchestration layer around the existing strict runner. A declarative YAML file defines the immutable protocol, development tasks, parameter groups, acceptance guards, and execution budget. A Python orchestrator expands candidates one coordinate group at a time, invokes the existing runner through Hydra overrides, and records a durable status for every task. A separate summarizer parses completed logs, ranks valid candidates, applies regression guards, and writes machine-readable and human-readable results.

No 0711 algorithm module is changed by default. The orchestration layer passes only parameters that the strict runner already supports. The best configuration remains a separate experiment artifact until the full 12-task validation passes.

## Files

- Create `Configs/Experiments/PU4D0711_strict_tuning.yaml`: immutable protocol, search groups, task split, thresholds, budget, and output settings.
- Create `tools/tune_pu4d_0711_strict.py`: validation, command construction, stage execution, task state persistence, retry, resume, and budget enforcement.
- Create `tools/summarize_pu4d_0711_tuning.py`: metric parsing, candidate aggregation, ranking, regression guards, and report generation.
- Create `tests/test_pu4d_0711_tuning.py`: unit and integration-contract tests for the tuning system.
- Create `run_pu4d_0711_tuning.sh`: preflight checks and one-command launch.
- Create a timestamped directory under `logs/PU4D_0711_STRICT_TUNING_*` at runtime. It contains state, commands, logs, metrics, leaderboards, frozen best configuration, and the final report.

## Data Flow

1. The launcher validates Python syntax, shell syntax, focused tests, PU4D cache, source checkpoints, and GPU 0 availability.
2. The orchestrator loads and validates the tuning YAML. Immutable strict-protocol overrides are injected by code rather than accepted from candidate definitions.
3. The orchestrator reruns the baseline on all 12 PU4D transfer tasks and enforces the recovery gate before starting the 24-hour tuning budget.
4. Every candidate-task pair receives a stable identifier derived from the normalized parameter mapping, task, source seed, and stream seed.
5. The orchestrator writes the exact command and changes the task state from `pending` to `running` before starting the subprocess.
6. The strict runner writes an isolated log. On successful exit, the orchestrator extracts Before, Strict Online, Post-stream, batch count, runtime, and diagnostics.
7. Only completed candidates with all required development tasks enter ranking. Failed or partial candidates cannot be averaged.
8. At the end of each coordinate group, the accepted winner becomes the anchor for the next group.
9. After stability selection, the configuration is frozen before any held-out task is launched.
10. The final summarizer combines the frozen candidate's three development scores and nine held-out scores into the 12-task table.

## Search Procedure and Budget

Runtime planning assumes approximately 16 minutes per task on GPU 0. The orchestrator enforces a 24-hour wall-clock deadline and will not launch a task that cannot reasonably finish inside the remaining reserve.

### Stage A: Baseline and Adaptation Dynamics

Reuse the three development-task results from the successful 12-task recovery
gate as the Stage A baseline. Then perform coordinate search, retaining the
current anchor as an option in every group:

1. `Opt.lr_tar`: 0.008, 0.012, 0.015, 0.018, 0.024.
2. `TTA0711.warp_lr_scale`: 0.05, 0.10, 0.20, using the accepted learning rate.
3. `TTA0711.ema_beta`: 0.990, 0.995, 0.999, using the accepted optimizer settings.

Estimated budget: 7–8 hours.

### Stage B: Routing, Memory, and Loss Balance

Starting from the Stage A anchor, search these groups in order:

1. `warmup_batches`: 0, 10, 25.
2. `aux_ramp_batches`: 10, 20, 40.
3. `min_reliability`: 0.10, 0.20, 0.30.
4. Loss profile:
   - conservative: `lambda_mt=0.01`, `lambda_pcl=0.01`, `lambda_ncl=0.005`;
   - current: `lambda_mt=0.02`, `lambda_pcl=0.02`, `lambda_ncl=0.01`;
   - enhanced: `lambda_mt=0.04`, `lambda_pcl=0.04`, `lambda_ncl=0.02`.
5. `memory_per_class`: 32, 64, 128.
6. Joint `pcl_temperature` and `ncl_temperature`: 0.10, 0.20, 0.30.

The following remain at their current values unless a later design is separately approved: view construction strengths, physical evidence mask parameters, adapter bounds, regularization weights, and model architecture.

Estimated budget: 9–10 hours.

### Stage C: Stability Selection

Select the top two valid configurations from the full development leaderboard. Run the baseline and both finalists on the same three development tasks with `stream_seed=2026`. Rank the finalists by their mean improvement over the matching baseline across both stream seeds. Use the minimum task-level improvement as the tie-breaker.

A finalist is rejected if either condition holds:

- its three-task mean improvement at `stream_seed=2025` is below 0.30 percentage points;
- any development task drops by more than 0.50 percentage points relative to its matching baseline.

If both finalists are rejected, the baseline remains the selected configuration.

Estimated budget: 2–3 hours.

### Stage D: Frozen 12-Task Validation

Freeze the selected universal configuration and run only the nine held-out transfer tasks with `stream_seed=2025`. Combine those scores with the already completed three development scores. Do not revisit the search after observing held-out results.

Estimated budget: 3 hours, leaving time for tests, interruption recovery, and reporting.

## Candidate Ranking

The primary development score is the arithmetic mean Strict Online Accuracy across the three development tasks. A candidate must have all three successful task runs. Higher mean ranks first; higher minimum task-level improvement ranks second; lower wall time ranks third. Post-stream accuracy is diagnostic only and never participates in selection.

All comparisons use unrounded parsed values when available. Tables display two decimal places. The final result reports both the exact aggregate and the rounded 12-task mean.

## Failure Handling and Resume

- Persist one atomic JSON state record per candidate-task pair with `pending`, `running`, `succeeded`, or `failed` status.
- Preserve the command, start/end times, exit code, log path, metric parse result, and retry count.
- Retry a nonzero subprocess once only when the failure is classified as transient, such as a CUDA initialization failure. Configuration, checkpoint, data, assertion, or protocol failures are not retried.
- On restart, skip only records marked `succeeded` whose log and parsed metric still validate. Treat stale `running` records as interrupted and eligible for one resumed attempt.
- A missing metric after exit code 0 is a failed run.
- Stop before tuning if any of the 12 recovery-baseline tasks differs from its historical Strict Online Accuracy by more than 0.05 percentage points or if the 12-task mean differs from 62.7342% by more than 0.05 percentage points.
- Never include a failed or incomplete candidate in ranking.

## Testing

Development follows test-driven development. Tests cover:

- YAML schema and rejection of unknown or protocol-breaking keys;
- deterministic candidate identifiers and coordinate expansion;
- exact command construction with forced strict invariants;
- parsing valid and malformed runner logs;
- exclusion of partial candidates from means;
- mean ranking, tie-breaks, 0.30-point gain threshold, and 0.50-point regression guard;
- atomic task-state transitions, stale-run recovery, retry classification, and successful-run skipping;
- wall-clock budget refusal for tasks that exceed the remaining reserve;
- report and CSV column contracts.

Verification proceeds in this order:

1. focused unit tests;
2. Python compilation and shell `bash -n`;
3. orchestrator `--dry-run`, confirming commands without launching training;
4. one full 0→1 baseline run, expected Strict Online Accuracy 40.18% ± 0.05;
5. the complete 12-task PU4D recovery gate, expected mean 62.7342% ± 0.05 and every task within ±0.05 of its historical score;
6. staged tuning;
7. frozen 12-task validation;
8. fresh focused and existing PU4D regression tests before completion is claimed.

## Outputs

The tuning run directory contains:

- `state/` task status JSON files;
- `commands/` exact shell-safe command records;
- `logs/` runner stdout/stderr;
- `metrics.csv` for every successful task;
- `leaderboard.csv` for complete development candidates;
- `best_config.yaml` with the frozen universal candidate;
- `final_12task.csv` with Before, Strict Online, Post-stream, delta, status, and runtime;
- `report.md` with baseline comparison, search history, rejected candidates, failures, checkpoint hashes, environment, and final conclusion.

## Acceptance Criteria

The tuned configuration is marked recommended only if all of the following hold:

- all 12 strict-online tasks complete successfully under one universal configuration;
- the exact 12-task mean Strict Online Accuracy exceeds 62.7342%;
- no strict-protocol invariant is changed;
- all focused and existing PU4D regression tests pass;
- the run is reproducible from the saved YAML, commands, source checkpoint hashes, source seed, and stream seed.

If these conditions are not satisfied, the project keeps the original 0.015-learning-rate strict configuration as the official baseline and records the search as a negative result. No production runner is silently modified.
