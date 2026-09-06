# Final integrity / selection remediation report

Date: 2026-08-24 UTC

Base HEAD: `9a92254`

Scope: HUST formal selection, freeze authority, stage/report completeness, V2 cache identity, load-cache gate, and immutable worker count. No GPU training/adaptation job was launched.

## Systematic diagnosis

The validated failures shared three trust boundaries:

1. Selection trusted recent group winners and reapplied the development eligibility gate to seed 2026, instead of globally ranking every complete seed-2025 candidate and using seed 2026 only as matched ranking evidence.
2. Freeze/report trusted self-hashed summaries without recomputing the experiment, cache tensors, source artifacts, complete coordinate inventory, stability evidence, or pre-freeze boundary.
3. Formal stages and reports treated partial inventories as usable; cache identity stopped at the manifest/shape layer; load-cache creation and `num_workers` were not immutable parts of the formal command/config identity.

## RED evidence

The behavior tests failed before production changes in the following ways:

- V2 cache tests observed `version == 1`; same-shape tensor tampering, missing files, and swapped files were not content-bound.
- source/target/load command tests found `num_workers=4` absent, and source preparation accepted another worker count.
- selection tests raised `TypeError` for `eligibility_seeds`; global ranking and override-space validation functions did not exist.
- an arbitrary self-hashed `best_config.yaml` was accepted as a freeze.
- incomplete `--write-final` produced artifacts, and formal source/Beginning/baseline/final stage seams returned failed record lists without raising.
- the reporter had no strict-final mode and the orchestrator had no conditional V2 load-cache gate.
- the first explicit untuned-fallback regression produced 24 duplicate seed-2026 jobs instead of one exact 12-task inventory.

## GREEN implementation

### Selection semantics

- Seed-2025 development eligibility is exactly mean gain `>= 0.30` and every task delta `>= -1.00`.
- Every complete eligible coordinate candidate from all ten groups is globally deduplicated and ranked; the global top three (or top two when fewer than three exist) become finalists.
- Matched two-seed ranking uses both-seed mean, then minimum delta, runtime, and candidate ID. Seed 2026 is not subjected to the seed-2025 eligibility guard.
- Final recommendation is true only when the exact complete frozen 12-task seed-2025 mean is strictly greater than the exact untuned 12-task mean. Seed-2025 development and seed-2026 diagnostic guards are reported separately.
- Explicit untuned fallback produces one bound inventory: four baseline development rows plus eight held-out rows and one 12-task seed-2026 stability set.

### Authoritative freeze

- `search_history.json` is written before the non-circular `search_proof.json`; `best_config.yaml` binds the proof afterward.
- Verification recomputes the official experiment file hash, V2 manifest/content/tensor identities, eight exact route/source/checkpoint/summary identities, all ten coordinate-group inventories and winners, global finalists, matched stability inventory, final ranking, selected rank/ID/overrides, and pre-freeze absence.
- Candidate IDs and override values are checked against the closed declared search space. Formal candidate commands and composed config hashes are recomputed from the bound configuration.
- Existing freezes are fully reverified on tune skip, before final planning, and before strict reporting.
- Every selected held-out/final state binds both `best_config.yaml` and `search_proof.json` by file SHA and internal authority hash, and must start after the freeze with exact frozen overrides.

### Formal completeness and reporting

- Source, Beginning, baseline, tune, and final stages raise on any failed or partial required job inventory; failed state remains persisted for diagnosis.
- Strict final reporting preflights before publication writes: 8 primary sources, 24 primary Beginning records, three exact 12-task seed-2025 baselines, exact frozen 4+8 seed-2025 rows, seed-2026 untuned/frozen inventories, authoritative proof/freeze, and formal state/log/command/artifact hashes.
- Non-final reporting is explicitly titled `DRAFT`; orchestrator `report` and the end of `all` use strict-final mode.

### V2 cache and runtime contract

- V2 manifests contain a SHA-256 for every `domain_<n>.pt` plus a canonical cache content hash. Validation rejects value tampering, missing files, extra/swapped domain files, and semantic mismatches.
- Formal source metadata and every orchestrator state bind the manifest, content identity, and every domain tensor.
- Future formal paths use `Dataset/HUST_STRICT_CACHE_V2`, `Dataset/HUST_STRICT_LOAD_CACHE_V2`, `TTA_Model_HUST_STRICT_V2`, and `TTA_Model_HUST_STRICT_LOAD_V2`. V1 files/artifacts were not changed or deleted.
- `STAGE=all` builds and validates the V2 load cache only when the completed primary Beginning audit requires supplementation; builder failure occurs before any load-audit GPU job.
- Formal source/target/load commands, preparation functions, source metadata, and composed config identities force/bind `num_workers=4`.

## Verification

- Focused HUST suite:
  - `pytest -q tests/test_hust_strict_cache.py tests/test_hust_physical_evidence.py tests/test_hust_source_protocol.py tests/test_hust_runner_contracts.py tests/test_hust_tuning.py`
  - Result: `367 passed`.
- Full plan ten-file suite:
  - `pytest -q tests/test_hust_strict_cache.py tests/test_hust_physical_evidence.py tests/test_hust_source_protocol.py tests/test_hust_runner_contracts.py tests/test_hust_tuning.py tests/test_dtcc_resnet18_common.py tests/test_fixed_random_stream.py tests/test_runner_contracts.py tests/test_pu4d_vanilla_runner_contracts.py tests/test_cwru_runner_contracts.py`
  - Result: `398 passed`.
- `py_compile` for the eleven launcher-listed Python modules: passed.
- `bash -n run_hust_dtcc_0711_strict.sh`: passed.
- `git diff --check` for the explicit remediation file set: passed.
- Direct orchestrator dry run: `224` jobs, `224` unresolved input plans, `launched=false`, every command contains `num_workers=4`, no `.pt` output.
- Launcher `STAGE=all DRY_RUN=1`: exactly two commands printed; no run directory created.

## Concern / handoff

The remediation intentionally did not build the real V2 caches or launch source/TTA GPU jobs. The next formal run must first materialize/validate the new V2 primary cache (or use `STAGE=cache`) and will create the V2 load cache conditionally during `STAGE=all`. Historical V1 runtime artifacts remain untouched and are not authoritative inputs for the future V2 run.

## Targeted review fix round

Review date: 2026-08-24 UTC

Three additional review gaps were reproduced RED and closed:

1. **Crash after proof publication, before freeze publication.** A valid existing `search_proof.json` with no `best_config.yaml` previously raised `FileExistsError`. Resume now fully re-verifies the existing proof, compares every non-timestamp semantic field with freshly recomputed tuning evidence, preserves the original proof bytes/timestamp, and proceeds to atomic freeze publication. A corrupted proof and a valid-but-mismatched proof both fail closed.
2. **Forged strict-final checkpoint manifest.** An arbitrary mapping with eight keys previously passed the inventory-length guard and failed only later on unrelated formal fields. Strict final preflight now requires exact equality among the supplied/CLI manifest, the authoritative proof manifest, and a fresh manifest independently reconstructed from the exact eight source states. The normal saved-evidence pass then independently verifies every declared source artifact hash before any final publication.
3. **Untuned fallback recommendation.** A numeric rerun jitter could previously make an explicit `candidate_id: untuned` fallback appear recommended. Recommendation is now always false for that fallback, `report_summary.json` records `recommendation_reason: untuned_fallback`, and the human report explicitly states that no tuned candidate is recommended regardless of numeric jitter. A non-fallback frozen candidate remains recommended exactly when its complete 12-task mean strictly exceeds the untuned complete 12-task mean.

Targeted RED result: three failing tests (`FileExistsError`, forged-manifest preflight bypass, and `recommended == true` for untuned jitter). Targeted GREEN result: `3 passed`.

Post-fix verification:

- Focused HUST suite: `369 passed`.
- Full ten-file plan suite: `400 passed`.
- `py_compile`, `bash -n`, and scoped `git diff --check`: passed.
- No GPU job, cache build, checkpoint write, or formal runtime mutation was performed.
