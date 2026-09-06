# Primary PU / WTPG / UO implementation backup

This branch preserves the code used for the current primary experiments:

- PU4D: four-domain 0711 split, 0711 BN-open runner, and vanilla-source DtCC runner.
- WTPG: four-domain split, strict 0711/BN-open runners, and DtCC runner.
- UO: four-domain split, robust-source 0711 runners, BN-open runner, and DtCC support.

The backup intentionally excludes raw datasets, generated caches, source/model
checkpoints, wandb artifacts, Python bytecode, and large historical log trees.
Those artifacts remain in the local workspace and are referenced by the local
experiment reports, but are not suitable for a source-code Git repository.

## Canonical local result records

- PU4D 0711 BN-open: `logs/PU4D_0711_BN_OPEN_20260905/`
- PU4D DtCC 0711-domain rerun: `logs/PU4D_DTCC_0711DOMAIN_RERUN_20260905/`
- WTPG four-domain final comparison: `logs/WTPG_REDIV_FINAL_TUNE_20260904/`
- WTPG DtCC tuning report: `logs/WTPG_DTCC_TUNED_FINAL_20260831/DTCC_WTPG_TUNING_REPORT.md`
- UO 0711 robust profile: `logs/UO_ROBUST_PROFILE_B_STRICT_TTA_20260905_FULL/`
- UO BN-open: `logs/UO_BN_OPEN_CURRENT_SOURCE_20260905/`
- UO DtCC: `logs/UO_DTCC_0711_20260903/` and `logs/UO_DTCC_0711_TUNE_20260904/`

## Reproduction note

The source checkpoint roots and cache paths in the runners are local-machine
paths. Before running on another machine, change the dataset/cache and source
checkpoint roots to local equivalents. The four-domain task definitions and
target-side parameter policies are kept in the dataset configs and runners.
