# HUST DtCC and 0711 Strict-Online Comparison Design

Date: 2026-08-23 (UTC)

## Goal

Implement a reproducible HUST bearing-domain benchmark for DtCC and the full
0711 method, tune one universal 0711 target configuration, and report both a
full-pipeline comparison and a common-source target-adaptation ablation. The
primary metric is pre-update Strict Online Accuracy. The preferred source-only
target difficulty is a `Beginning Acc T` between 25% and 70% for every task,
subject to preserving a physically meaningful domain split.

## Existing-State Findings

The current HUST artifacts are not suitable as the formal benchmark:

- `Dataset/HUST_CACHE` contains four bearing domains and seven classes, but
  stores 1025-bin spectra without shaft-frequency metadata.
- the top-level `HUST_CACHE` duplicates those tensors and additionally stores
  `fs.pt`, but the active dataset loader points at the top-level cache through
  a hard-coded absolute path;
- the existing 0711 HUST wrapper reuses the CWRU physical-evidence runner, so
  its evidence masks use the wrong bearing geometry and frequency metadata;
- the old DtCC HUST run used a different source path and produced Beginning
  Accuracies from 8.59% to 36.34%, with several tasks below the preferred
  range;
- the existing workspace contains substantial historical and user-owned
  changes, so new work must not overwrite or clean unrelated files.

All new caches, checkpoints, logs, and reports therefore use isolated HUST
strict-protocol paths.

## Primary Dataset Protocol

### Domains and Classes

The primary experiment uses bearing identity as the domain variable:

| Domain | Bearing |
|---|---|
| 0 | 6205 |
| 1 | 6206 |
| 2 | 6207 |
| 3 | 6208 |

The fixed label map is:

| Label | Condition |
|---:|---|
| 0 | N |
| 1 | I |
| 2 | O |
| 3 | B |
| 4 | IB |
| 5 | IO |
| 6 | OB |

Bearing 6204 is excluded from the four-domain primary experiment because the
published dataset is missing its B and IB cases. Each primary domain contains
all seven conditions at 0 W, 200 W, and 400 W.

### Signal Processing and Balancing

The cache is rebuilt from the original MAT files under `Dataset/HUST`:

- vibration sampling rate: 51,200 Hz;
- window length: 2,048 samples;
- window stride: 1,024 samples;
- transform: 2,048-point FFT magnitude divided by 2,048, with no per-window
  maximum normalization;
- retained representation: the first 512 frequency bins;
- frequency resolution: 25 Hz per retained bin;
- model-input normalization: the existing per-sample mean/std transform used
  identically by both methods;
- deterministic equal sampling across bearing, class, and load;
- fixed preprocessing seed: 2025.

Balancing operates per raw recording. Every bearing/class/load cell contributes
the same number of windows, chosen from the global minimum available cell count
with a deterministic generator. This removes class and load-count priors while
retaining all seven classes and three loads in every domain.

Each cached sample retains its label for offline metrics and the following
non-label metadata: domain, bearing identifier, load, shaft frequency in Hz,
raw recording identifier, and window offset. A metadata manifest records the
label map, domain map, signal-processing parameters, balance count, seed,
source-file hashes, output tensor shapes, and class/load counts.

### Online Stream

For every source-target pair, the target cache is traversed exactly once using
one fixed random permutation. The primary stream seed is 2025. Prediction is
recorded before the current target batch updates the model. A batch cannot be
revisited, and neither prior nor future target batches may be replayed.

Target labels are permitted only in accuracy/F1 accumulators and in the
offline development-task scorer. They may not affect gradients, pseudo-labels,
evidence masks, routing, memory, early stopping, per-batch hyperparameters, or
any other adaptation decision.

## Source Checkpoints and Comparison Questions

Both source routes use the `ResNet18_1D_SDE` architecture, identical
bottleneck/classifier dimensions, source seed 2025, the same processed
source-domain data, 50 epochs, batch size 128, AdamW with initial learning rate
0.001 and weight decay 0.0001, and label smoothing 0.1. The fixed epoch-50
checkpoint is used; source accuracy is diagnostic and does not select an
epoch. They intentionally produce two different checkpoints.

### DtCC Source Checkpoint

The DtCC checkpoint uses ordinary supervised source training without SSP-lite,
SDE, or learnable spectral correction. Target-only carrier parameters remain
at identity and are frozen during source training. Epoch selection and model
saving use only fixed source-side rules and source metrics.

### 0711 Source Checkpoint

The 0711 checkpoint uses the source pre-robustification defined by
`0711方法系统概述.md`: SSP-lite amplitude-style views, SDE frequency-axis
deformations, supervised clean/augmented objectives, and the specified
cross-view consistency terms. Target-only adapter and F-Warp parameters remain
at identity during source training. Epoch selection and model saving use only
fixed source-side rules and source metrics.

### Full-Pipeline Main Comparison

The main comparison answers which complete method works better:

- DtCC target adaptation starts from the ordinary DtCC source checkpoint;
- 0711 target adaptation starts from the SSP-lite + SDE 0711 checkpoint.

The performance difference may contain both source pre-robustification and
target-adaptation contributions. The report must state this explicitly.

### Common-Source Ablation

The controlled ablation isolates target adaptation:

- DtCC and 0711 both start from the ordinary DtCC source checkpoint;
- both methods run all 12 transfer tasks;
- the input data, target permutation, batch size, source seed, stream seed, and
  metric implementation are identical.

The DtCC result is shared between the full-pipeline table and common-source
table because DtCC uses the ordinary checkpoint in both. No duplicate DtCC
run is required when the exact artifact and protocol are identical.

## HUST Physical Fault Evidence

The 0711 HUST runner uses a dedicated physical-evidence module. It must not
import CWRU characteristic-frequency tables.

For each target bearing, pitch diameter is estimated from the documented inner
and outer diameters. With documented ball diameter and ball count, zero contact
angle, and the cached per-recording shaft frequency, the module computes BPFO,
BPFI, and BSF. Harmonics and class-appropriate sidebands are projected to the
25 Hz FFT grid and converted into smooth, bounded masks.

The condition-to-evidence mapping is:

- I: BPFI bands;
- O: BPFO bands;
- B: BSF bands and permitted sidebands;
- IB: union of I and B evidence;
- IO: union of I and O evidence;
- OB: union of O and B evidence;
- N: physical fault evidence is not applicable and no synthetic fault band is
  invented.

For composite conditions, the union is capped by the same maximum mask-ratio
guard as a single condition. A destructive view replaces active bands with a
local smooth background instead of zeroing them. The model's predicted class,
not the true class, selects the intervention mask.

## Target Adaptation Methods

### DtCC

DtCC follows the existing verified implementation and the referenced paper:

- dynamic certain/uncertain division from spectral entropy and prediction
  confidence using adaptive batch thresholds;
- class-wise prototype memory;
- PCL for certain samples;
- NCL for uncertain samples;
- smooth entropy minimization;
- BN affine parameters are the only trainable target parameters.

DtCC uses one fixed paper/project-default configuration for HUST and is not
tuned using HUST target labels.

### 0711

0711 follows `0711方法系统概述.md`:

- clean and task-preserving teacher views;
- EMA over target adaptation parameters;
- confidence, view agreement, and HUST intervention evidence;
- class-balanced reliability routing;
- evidence-gated, reliability-weighted memory;
- PCL, NCL, reliability-weighted SEM, and mean-teacher distillation;
- only `band_scale`, `band_bias`, and `warp_ctrl` are trainable;
- backbone, classifier, BN affine parameters, and BN running statistics remain
  frozen;
- one update schedule and one universal configuration apply to all tasks.

## Beginning-Accuracy Difficulty Audit

After source training, both checkpoint families are frozen before any target
tuning. All 12 source-target pairs are evaluated to obtain `Beginning Acc T`.

The preferred result is 12 of 12 tasks in the 25%-70% range. The primary split
is still acceptable when at least 9 of 12 tasks are in range and no task is
below 15% or above 80%. This is a soft difficulty criterion, not permission to
select checkpoints using target labels.

If the acceptable condition is not met, the four-bearing result is preserved
and reported. A supplementary three-domain audit then uses load (0 W, 200 W,
400 W) as the natural domain variable. The load split cannot silently replace
the bearing split, and the final DtCC/0711 winner cannot be used to decide which
split to publish.

## 0711 Tuning Protocol

### Task Split

The four development tasks are fixed before new target runs:

- 0->1;
- 1->2;
- 2->3;
- 3->0.

They cover every source and target once. The other eight directed pairs are
held out during candidate selection.

### Recovery Baseline

The orchestrator first runs the untuned 0711 configuration and fixed DtCC
configuration on all 12 tasks. Cache manifests, checkpoint tensors and hashes,
trainable-parameter sets, target permutation, and metric parsing must pass
before tuning begins.

### Search Stages

Coordinate groups are searched from a declared anchor while retaining the
current anchor in every group:

1. target learning rate: 0.008, 0.015, 0.024, 0.036;
2. adapter learning-rate scale: 0.5, 1.0, 2.0;
3. warp learning-rate scale: 0.05, 0.10, 0.20;
4. EMA beta: 0.990, 0.995, 0.999;
5. warm-up batches: 0, 5, 10;
6. auxiliary-loss ramp batches: 10, 20, 40;
7. minimum reliability: 0.10, 0.20, 0.30;
8. loss profile: conservative (`lambda_mt=0.01`, `lambda_pcl=0.01`,
   `lambda_ncl=0.005`), current (`0.02`, `0.02`, `0.01`), or enhanced
   (`0.04`, `0.04`, `0.02`);
9. memory per class: 32, 64, 128;
10. joint PCL/NCL temperature: 0.10, 0.20, 0.30.

The frozen PU4D recommendation is included as one initial candidate but is not
assumed to transfer optimally to HUST. Sampling rate, FFT construction,
bearing geometry, characteristic-frequency formulae, label map, target
architecture, trainable-parameter allowlist, number of target passes, and
pre-update scoring cannot be candidate parameters.

Every candidate is evaluated on all four development tasks with stream seed
2025. Only complete candidates enter ranking. The top two or three candidates,
together with the baseline, are rerun on the same development tasks with
stream seed 2026.

Candidates rank by mean Strict Online Accuracy across both stream seeds, then
by minimum task-level improvement, then by lower runtime. A candidate is
eligible only when its seed-2025 development mean improves over the untuned
0711 baseline by at least 0.30 percentage points and no development task drops
by more than 1.00 percentage point.

### Frozen Validation

One universal configuration is frozen before any held-out task is launched.
The eight held-out tasks run with stream seed 2025 and cannot trigger another
search. The untuned baseline and frozen candidate then both complete all 12
tasks with stream seed 2026; the four development-task seed-2026 results may be
reused after artifact validation. The primary table contains the complete
seed-2025 12-task result, and the stability table contains matched complete
seed-2026 baseline and frozen-candidate results.

The tuned configuration is marked recommended only if its exact 12-task mean
Strict Online Accuracy exceeds the untuned 0711 mean. Whether it exceeds DtCC
is reported separately and does not alter the acceptance rule. If selection or
held-out validation fails, the untuned 0711 configuration remains official and
the tuning search is reported as a negative result.

## Parallel Execution and Resume

The launcher discovers GPUs at run time and uses only idle or low-load devices.
It never terminates, resets, or takes ownership of an existing process. Each
GPU runs at most one experiment subprocess at a time. Parallel execution may
change wall-clock completion time but not seeds, task definitions, candidate
ranking, target order, or numerical acceptance rules.

The tuning wall-clock budget is 24 hours. The orchestrator records one atomic
state file per candidate-task pair with command, GPU, timestamps, exit code,
retry count, log path, and parsed metrics. It skips only previously successful
runs whose artifacts still validate. Interrupted runs may resume once;
configuration, cache, checkpoint, assertion, and protocol failures are not
silently retried.

## Metrics and Reporting

The primary metric is pre-update Strict Online Accuracy. Every task also
reports:

- Beginning Accuracy;
- Strict Online macro precision, recall, and F1;
- post-stream full-target accuracy and macro metrics as diagnostics only;
- runtime per batch and per task;
- peak GPU memory;
- memory size and class coverage;
- certain/uncertain routing proportions;
- offline pseudo-label purity and class confusion matrix as non-adaptive
  diagnostics.

Offline diagnostics consume labels only after the decisions they describe have
already been made. They cannot be exposed to adaptation code.

The final report contains separate full-pipeline and common-source tables,
unrounded aggregates, per-task values, Beginning-Accuracy range violations,
search history, rejected candidates, failure records, cache and checkpoint
hashes, source/stream seeds, exact commands, environment details, and the final
recommendation or negative result.

## Components and Artifact Isolation

Implementation introduces focused HUST strict-protocol components:

- an isolated strict cache builder and manifest;
- a strict cached HUST dataset loader;
- a HUST physical-fault-evidence module;
- ordinary and pre-robustified source-training entry points;
- DtCC, 0711 full-pipeline, and 0711 common-source target routes;
- a declarative HUST tuning configuration;
- a resumable tuning orchestrator and result summarizer;
- focused unit and runner-contract tests;
- timestamped runtime directories for commands, states, logs, tables, frozen
  configuration, and report.

Existing HUST caches, source models, logs, archives, and unrelated dirty
worktree files are never overwritten, moved, or deleted.

## Failure Handling

- Invalid MAT metadata, an unknown filename, a missing class/load cell, or a
  non-finite/non-positive shaft frequency fails cache construction before
  output is promoted.
- A cache-manifest or tensor-shape mismatch fails preflight.
- A checkpoint with missing, unexpected, non-identity carrier, or wrong-class
  tensors fails strict loading.
- A trainable parameter outside the method-specific allowlist aborts the task.
- A target run with a missing final metric is failed even when its subprocess
  exits with status zero.
- Failed or partial candidates cannot enter averages or rankings.
- GPU unavailability delays pending work; it does not authorize CPU fallback
  for formal results or takeover of an occupied GPU.

## Testing and Verification

Tests cover:

- longest-prefix parsing for IB, IO, and OB filenames;
- exclusion of incomplete 6204 from the primary split;
- class/load balancing, deterministic sampling, tensor shapes, metadata, and
  source-file hashes;
- 512-bin FFT output and the 25 Hz frequency grid;
- numerical BPFO, BPFI, and BSF calculations for every primary bearing;
- composite-mask unions, maximum mask ratio, and N-class non-applicability;
- use of predicted class and cached shaft metadata without target-label access;
- distinct DtCC/0711 source routes and the 0711 common-source checkpoint route;
- fixed one-pass random streams and pre-update metrics;
- DtCC BN-only and 0711 adapter/F-Warp-only trainable sets;
- tuning schema, candidate identifiers, command invariants, parsing, complete
  candidate filtering, stability ranking, regression guards, held-out freeze,
  atomic state transitions, and resume;
- a small synthetic CPU integration test and one full GPU smoke task;
- relevant existing PU4D/CWRU/common-source regression tests.

Verification order is focused tests, Python compilation and shell syntax,
cache dry-run, cache build and audit, source smoke runs, source training,
Beginning-Accuracy audit, one DtCC and one 0711 full target smoke task, 12-task
baselines, staged tuning, frozen held-out validation, stability runs, and a
fresh final regression suite.

## Acceptance Criteria

Work is complete only when:

- the primary strict cache and manifest validate;
- both checkpoint families are reproducible and strictly audited;
- DtCC ordinary-source, 0711 robust-source, and 0711 common-source routes each
  complete all 12 primary tasks;
- every route uses a fixed one-pass target stream with pre-update scoring;
- one universal frozen 0711 configuration is used for final validation;
- the full-pipeline and common-source conclusions are reported separately;
- Beginning-Accuracy soft-constraint violations are visible rather than
  hidden;
- focused and relevant regression tests pass;
- saved configuration, commands, manifests, hashes, seeds, logs, and reports
  are sufficient to reproduce the result;
- no existing user artifact is overwritten or removed.
