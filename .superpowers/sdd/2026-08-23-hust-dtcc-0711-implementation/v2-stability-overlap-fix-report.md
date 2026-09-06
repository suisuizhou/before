# V2 seed-2026 stability overlap remediation

## Scope and safety

- Investigated the strict-report failure in the immutable formal run
  `logs/HUST_DTCC_0711_STRICT_20260824_081610` at baseline HEAD `76ecb48`.
- Did not edit or delete any formal state, command, log, checkpoint, cache, freeze,
  proof, or runtime report artifact. CPU compatibility reports were written only
  to an isolated temporary directory whose `state/` was a read-only reference to
  the formal run.

## Root cause

The final-stage planner scheduled all 12 seed-2026 tasks for the untuned and
frozen candidates even though the four development tasks had already completed
before freeze as authoritative `tune_stability` evidence. The reporter then
treated the later 12-task `stability` stage as authoritative and required the
four duplicated metric records to be exactly equal to the pre-freeze records.
The duplicated frozen 3→0 execution differed by 0.023457658925636338 accuracy
percentage points because of runtime nondeterminism, so an identity-valid run
failed at an invalid metric-equality requirement.

## TDD evidence

The new regression tests were first run against the old implementation and all
three failed for the intended reasons:

- final planning returned 32 jobs instead of 24;
- untuned fallback returned 12 final stability jobs instead of 8;
- a metric-only redundant rerun drift raised the old overlap-mismatch error.

The production change then made those tests pass. The tests additionally prove
that a redundant identity/config mismatch still fails and that a missing
authoritative `tune_stability` development row still fails.

## Implemented protocol

- A non-fallback final plan contains exactly 24 jobs: eight seed-2025 held-out
  jobs, eight untuned seed-2026 non-development jobs, and eight frozen seed-2026
  non-development jobs. The four seed-2026 development tasks are never rerun.
- The authoritative seed-2026 table for each reported candidate is exactly four
  pre-freeze `tune_stability` development rows plus eight post-freeze
  `stability` non-development rows.
- Candidate, route, variant, seed, config hash, overrides, source identity, and
  source-checkpoint hash must agree. Strict preflight also binds the four
  development rows to the validated search-proof evidence paths and config
  hashes.
- Every authoritative and redundant seed-2026 row must also share the exact
  proof-authoritative V2 cache identity: resolved manifest path, manifest
  SHA-256, content SHA-256, and the complete tensor-path/SHA-256 mapping after
  canonical absolute-path normalization. An internally valid but different
  cache cannot be mixed into either the authoritative table or redundant audit.
- Historical extra `stability` development rows are preserved as explicit
  non-authoritative audit evidence. Metric drift is allowed and reported in
  `stability_seed2026_redundant_audit.csv`; those rows are excluded from the
  per-task authoritative table, leaderboard means, stability means, guards, and
  recommendation.

## Formal-run compatibility evidence

An isolated CPU strict-report replay over the existing 258 immutable states
completed successfully:

- authoritative `stability_seed2026.csv`: 24 data rows;
- redundant audit CSV: 8 data rows;
- maximum absolute accuracy drift: 0.023457658925636338 pp;
- maximum absolute macro-F1 drift: 0.015314171187913672 pp;
- authoritative seed-2026 gain: 2.124872936748698 pp;
- authoritative worst-task delta: -0.8679333802486511 pp;
- seed-2026 stability guard: passed;
- recommendation: true.

A read-only reconstruction of the repeated final-stage validity gate reported
`planned=24`, `valid_skips=24`, `launches=0`, `heldout=8`, and
`stability_nondev=16`. Thus a repeated final stage reuses every authoritative
state and leaves the eight historical redundant audit rows immutable.

## Verification

- Initial overlap-focused RED/GREEN tests: 3 passed after the fix.
- Cache-binding review follow-up: 12 mutation cases first failed against the
  incomplete implementation, then 13 focused overlap/cache tests passed. The
  mutations cover each cache path/hash/tensor field on both authoritative
  non-development and redundant development rows, plus internally consistent
  alternate-cache identities rejected against the proof.
- Full `tests/test_hust_tuning.py`: 181 passed.
- Plan regression suite: 412 passed.
- Final verification also covers Python compilation, shell syntax, and
  `git diff --check`; see the task handoff for the fresh command output.
