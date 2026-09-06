# HUST DtCC and 0711 Strict-Online Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, run, tune, and report a physically grounded HUST strict-online benchmark comparing DtCC with the full 0711 pipeline and a 0711 common-source ablation.

**Architecture:** Build a versioned four-bearing cache with per-sample shaft metadata, load it through an isolated dataset class, and add a metadata hook to the existing strict 0711 runner so HUST can supply its own physical masks without changing PU4D behavior. Train two source routes into isolated checkpoint trees, expose dedicated DtCC/0711 HUST target entry points, and drive all 12-task comparisons and coordinate tuning through a resumable multi-GPU orchestrator.

**Tech Stack:** Python 3.13, PyTorch, NumPy, SciPy, Hydra/OmegaConf, PyYAML, pytest, Bash, NVIDIA CUDA.

**Spec:** `docs/superpowers/specs/2026-08-23-hust-dtcc-0711-design.md`

## Global Constraints

- Primary domains are bearing IDs 6205, 6206, 6207, and 6208; 6204 is excluded from the primary split.
- Labels are exactly `N=0, I=1, O=2, B=3, IB=4, IO=5, OB=6`.
- Cache preprocessing is 51,200 Hz, window 2,048, stride 1,024, 2,048-point FFT magnitude divided by 2,048, first 512 bins, and deterministic per-recording balancing with seed 2025.
- Both source routes use `ResNet18_1D_SDE`, 50 epochs, batch size 128, AdamW `lr=0.001`, weight decay `0.0001`, label smoothing `0.1`, and source seed 2025.
- DtCC source is ordinary supervised training; 0711 source adds SSP-lite + SDE pre-robustification.
- Formal target runs are fixed-random, one pass, pre-update scoring; target labels never influence adaptation.
- DtCC target trains BN affine parameters only; 0711 target trains only `band_scale`, `band_bias`, and `warp_ctrl`.
- Main comparison uses separate ordinary/robust checkpoints; common-source 0711 uses the ordinary checkpoint.
- One universal 0711 target configuration is frozen before eight held-out tasks.
- Existing HUST caches, checkpoints, logs, archives, and unrelated dirty worktree files are never overwritten, moved, deleted, or included in task commits.
- Formal runs use only idle/low-load GPUs and never terminate or reset another process.

---

## File Structure

### New files

- `Lib/hust_strict_protocol.py`: constants, filename parsing, manifest schema, cache validation, checkpoint paths/hashes, and strict checkpoint loading.
- `tools/build_hust_strict_cache.py`: deterministic raw-MAT-to-cache builder with atomic promotion.
- `Dataset/HUSTStrict.py`: normalized tensor dataset with stable indices and shaft metadata.
- `Configs/Dataset/HUSTStrict.yaml`: primary cache location and four-domain dataset configuration.
- `Lib/hust_physical_fault_evidence.py`: HUST geometry, HUST-specific evidence config, characteristic frequencies, class/composite masks, and applicability.
- `Lib/hust_source_training.py`: shared ordinary/robust source training loop that never evaluates a target domain during training.
- `main_src_dtcc_hust_strict.py`: ordinary source CLI.
- `main_src_0711_hust_strict.py`: SSP-lite + SDE source CLI.
- `main_tta_dtcc_hust_strict.py`: DtCC target CLI and strict ordinary-checkpoint route.
- `main_tta_0711_hust_strict.py`: full/common-source 0711 CLI and HUST evidence hooks.
- `Configs/Experiments/HUST0711_strict_tuning.yaml`: immutable protocol, routes, task split, search space, guards, and GPU policy.
- `tools/tune_hust_0711_strict.py`: validation, multi-GPU scheduling, task state, resume, search, stability, and frozen final execution.
- `tools/summarize_hust_dtcc_0711.py`: log parsing, route/task aggregation, difficulty audit, tables, and report.
- `run_hust_dtcc_0711_strict.sh`: tested one-command staged launcher.
- `tests/test_hust_strict_cache.py`: parser, FFT, balance, manifest, and dataset tests.
- `tests/test_hust_physical_evidence.py`: frequency and mask tests.
- `tests/test_hust_source_protocol.py`: source-route objective/checkpoint tests.
- `tests/test_hust_runner_contracts.py`: checkpoint route, parameter allowlist, metadata hook, and online-order tests.
- `tests/test_hust_tuning.py`: configuration, scheduling, resume, ranking, freeze, and reporting tests.

### Modified files

- `Dataset/__init__.py`: register `HUSTStrict` only.
- `main_tta_0711_strict_randomstream.py`: add optional sample-metadata/evidence-mask/diagnostic hooks while retaining identical PU4D defaults.

### Runtime artifacts (not committed)

- `Dataset/HUST_STRICT_CACHE_V1/`
- `TTA_Model_HUST_STRICT/{ordinary,robust}/source_<n>/seed_2025/`
- `logs/HUST_DTCC_0711_STRICT_<UTC timestamp>/`

---

### Task 1: Strict HUST cache contract and deterministic builder

**Files:**
- Create: `Lib/hust_strict_protocol.py`
- Create: `tools/build_hust_strict_cache.py`
- Create: `tests/test_hust_strict_cache.py`

**Interfaces:**
- Produces: `parse_hust_filename(name: str) -> HUSTRecording`, `fft_window(signal: np.ndarray) -> np.ndarray`, `build_cache(raw_root: Path, output_root: Path, seed: int, split: Literal["bearing", "load"] = "bearing") -> dict`, `validate_cache(root: Path) -> dict`.
- Produces cache files `domain_0.pt` through `domain_3.pt` with keys `data`, `label`, `shaft_hz`, `load_w`, `recording_id`, `window_offset` and a `manifest.json`.

- [ ] **Step 1: Write failing filename, FFT, and balancing tests**

```python
# tests/test_hust_strict_cache.py
from pathlib import Path
import numpy as np
import pytest
from scipy.io import savemat

from Lib.hust_strict_protocol import LABEL_MAP, parse_hust_filename
from tools.build_hust_strict_cache import build_cache, fft_window, balanced_window_indices


@pytest.fixture
def synthetic_raw_hust(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    signal = np.sin(2 * np.pi * 8 * np.arange(4096) / 2048).astype(np.float32)
    for bearing in range(4, 9):
        for fault in LABEL_MAP:
            if bearing == 4 and fault in {"B", "IB"}:
                continue
            for load_code in ("00", "02", "04"):
                savemat(root / f"{fault}{bearing}{load_code}.mat", {"data": signal[:, None], "fs": [[24.0]]})
    return root


@pytest.mark.parametrize(
    ("name", "fault", "bearing", "load"),
    [
        ("IB504.mat", "IB", 6205, 400),
        ("IO602.mat", "IO", 6206, 200),
        ("OB700.mat", "OB", 6207, 0),
        ("N800.mat", "N", 6208, 0),
    ],
)
def test_parse_hust_filename_uses_longest_prefix(name, fault, bearing, load):
    row = parse_hust_filename(name)
    assert (row.fault, row.label, row.bearing, row.load_w) == (
        fault,
        LABEL_MAP[fault],
        bearing,
        load,
    )


def test_parse_rejects_runup_or_unknown_names():
    with pytest.raises(ValueError):
        parse_hust_filename("IB50.mat")


def test_fft_window_has_exact_grid_and_scale():
    n = np.arange(2048)
    signal = np.sin(2 * np.pi * 8 * n / 2048).astype(np.float32)
    spectrum = fft_window(signal)
    assert spectrum.shape == (1, 512)
    assert spectrum.dtype == np.float32
    assert int(spectrum[0].argmax()) == 8
    assert spectrum[0, 8] == pytest.approx(0.5, rel=1e-5)


def test_balanced_indices_are_deterministic_and_equal():
    available = {"a": 499, "b": 430, "c": 470}
    first = balanced_window_indices(available, seed=2025)
    second = balanced_window_indices(available, seed=2025)
    assert first.keys() == second.keys()
    assert all(np.array_equal(first[k], second[k]) for k in first)
    assert {len(v) for v in first.values()} == {430}


def test_load_split_contract_has_three_complete_domains(tmp_path, synthetic_raw_hust):
    manifest = build_cache(synthetic_raw_hust, tmp_path / "load", seed=2025, split="load")
    assert manifest["split"] == "load"
    assert manifest["domain_map"] == {"0": "0W", "1": "200W", "2": "400W"}
    for row in manifest["domains"].values():
        assert set(row["class_counts"]) == {"0", "1", "2", "3", "4", "5", "6"}
        assert len(set(row["class_counts"].values())) == 1
```

- [ ] **Step 2: Run the focused test and verify import failures**

Run: `pytest -q tests/test_hust_strict_cache.py`

Expected: FAIL because `Lib.hust_strict_protocol` and `tools.build_hust_strict_cache` do not exist.

- [ ] **Step 3: Implement constants, strict parsing, hashing, FFT, balance, and atomic cache promotion**

```python
# Lib/hust_strict_protocol.py
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re
import torch

LABEL_MAP = {"N": 0, "I": 1, "O": 2, "B": 3, "IB": 4, "IO": 5, "OB": 6}
DOMAIN_TO_BEARING = {0: 6205, 1: 6206, 2: 6207, 3: 6208}
BEARING_TO_DOMAIN = {value: key for key, value in DOMAIN_TO_BEARING.items()}
LOAD_CODE_TO_WATTS = {"00": 0, "02": 200, "04": 400}
FILENAME_RE = re.compile(r"^(IB|IO|OB|N|I|O|B)([4-8])(00|02|04)\.mat$")


@dataclass(frozen=True)
class HUSTRecording:
    fault: str
    label: int
    bearing: int
    load_w: int


def parse_hust_filename(name: str) -> HUSTRecording:
    match = FILENAME_RE.fullmatch(Path(name).name)
    if match is None:
        raise ValueError(f"unsupported HUST filename: {name}")
    fault, bearing_digit, load_code = match.groups()
    return HUSTRecording(fault, LABEL_MAP[fault], 6200 + int(bearing_digit), LOAD_CODE_TO_WATTS[load_code])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_cache(root: Path) -> dict:
    manifest_path = Path(root) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["version"] == 1
    assert manifest["label_map"] == LABEL_MAP
    for domain_text in manifest["domain_map"]:
        domain = int(domain_text)
        obj = torch.load(Path(root) / f"domain_{domain}.pt", map_location="cpu", weights_only=True)
        n = int(obj["label"].numel())
        assert obj["data"].shape == (n, 1, 512)
        assert set(obj) == {"data", "label", "shaft_hz", "load_w", "recording_id", "window_offset"}
        assert torch.isfinite(obj["data"]).all() and torch.isfinite(obj["shaft_hz"]).all()
        assert (obj["shaft_hz"] > 0).all()
    return {**manifest, "manifest_sha256": sha256_file(manifest_path)}
```

In `tools/build_hust_strict_cache.py`, implement `fft_window()` as `abs(np.fft.fft(window, n=2048))[:512] / 2048.0`, validate MAT keys `data` and scalar positive `fs`, and select with a stable seed based on the sorted recording index. For `split="bearing"`, exclude 6204 and derive the global minimum window count across the 84 primary recordings. For `split="load"`, include 6204-6208, aggregate recordings within each load/class, and deterministically downsample every load/class to the global minimum aggregate count so missing 6204 B/IB recordings cannot create a class prior. Write into a `tempfile.mkdtemp(dir=output.parent)` directory, call `validate_cache()`, and promote with `os.replace()`. Refuse to replace an existing non-identical output directory; if it already validates, exit successfully and print its manifest hash.

- [ ] **Step 4: Run focused tests and a raw-data dry run**

Run: `pytest -q tests/test_hust_strict_cache.py`

Expected: PASS.

Run: `python tools/build_hust_strict_cache.py --raw-root Dataset/HUST --output Dataset/HUST_STRICT_CACHE_V1 --seed 2025 --dry-run`

Expected: prints 84 primary recordings, 4 domains, 7 classes, 3 loads, the global per-recording balance count, and writes nothing.

- [ ] **Step 5: Commit the cache contract**

```bash
git add Lib/hust_strict_protocol.py tools/build_hust_strict_cache.py tests/test_hust_strict_cache.py
git commit -m "feat: define strict HUST cache protocol"
```

### Task 2: Strict cached dataset loader and Hydra registration

**Files:**
- Create: `Dataset/HUSTStrict.py`
- Create: `Configs/Dataset/HUSTStrict.yaml`
- Modify: `Dataset/__init__.py`
- Modify: `tests/test_hust_strict_cache.py`

**Interfaces:**
- Produces: `HUSTStrictTensorDataset`, exposing `shaft_hz`, `load_w`, `recording_id`, and `window_offset` tensors while returning `(normalized_x, label, stable_index)`.
- Produces: `HUSTStrict(**cfg.Dataset).data_generator() -> (source_dataset, target_dataset)`.

- [ ] **Step 1: Add failing dataset normalization and metadata tests**

```python
def test_hust_strict_dataset_returns_stable_index_and_metadata(tmp_path):
    import torch
    from Dataset.HUSTStrict import HUSTStrictTensorDataset

    obj = {
        "data": torch.arange(1024, dtype=torch.float32).reshape(2, 1, 512),
        "label": torch.tensor([1, 4]),
        "shaft_hz": torch.tensor([24.8, 23.1]),
        "load_w": torch.tensor([0, 400]),
        "recording_id": torch.tensor([7, 8]),
        "window_offset": torch.tensor([0, 1024]),
    }
    ds = HUSTStrictTensorDataset(obj)
    x, y, index = ds[1]
    assert (y, index) == (4, 1)
    assert x.shape == (1, 512)
    assert float(x.mean()) == pytest.approx(0.0, abs=1e-5)
    assert float(x.std(unbiased=False)) == pytest.approx(1.0, rel=1e-5)
    assert ds.shaft_hz[index].item() == pytest.approx(23.1)
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `pytest -q tests/test_hust_strict_cache.py::test_hust_strict_dataset_returns_stable_index_and_metadata`

Expected: FAIL because `Dataset.HUSTStrict` does not exist.

- [ ] **Step 3: Implement the dataset class and safe normalization**

```python
# Dataset/HUSTStrict.py
from pathlib import Path
import torch
from torch.utils.data import Dataset
from Lib.hust_strict_protocol import validate_cache


class HUSTStrictTensorDataset(Dataset):
    def __init__(self, obj):
        self.data = obj["data"].float()
        self.label = obj["label"].long()
        self.shaft_hz = obj["shaft_hz"].float()
        self.load_w = obj["load_w"].long()
        self.recording_id = obj["recording_id"].long()
        self.window_offset = obj["window_offset"].long()

    def __len__(self):
        return int(self.label.numel())

    def __getitem__(self, index):
        x = self.data[index]
        mean = x.mean()
        std = x.std(unbiased=False).clamp_min(1e-8)
        return (x - mean) / std, int(self.label[index]), int(index)


class HUSTStrict:
    inputchannel = 1
    num_classes = 7

    def __init__(self, data_path, TL_Task=(0, 1), **_kwargs):
        self.root = Path(data_path)
        self.source, self.target = (int(TL_Task[0]), int(TL_Task[1]))
        validate_cache(self.root)

    def _load(self, domain):
        obj = torch.load(self.root / f"domain_{domain}.pt", map_location="cpu", weights_only=True)
        return HUSTStrictTensorDataset(obj)

    def data_generator(self):
        return self._load(self.source), self._load(self.target)
```

Register `from .HUSTStrict import HUSTStrict` in `Dataset/__init__.py`. Set `Configs/Dataset/HUSTStrict.yaml` to `data_name: HUSTStrict`, `data_path: Dataset/HUST_STRICT_CACHE_V1`, `TL_list: [0, 1, 2, 3]`, `input_kind: fft`, and `norm_kind: mean-std`.

- [ ] **Step 4: Run cache and dataset tests**

Run: `pytest -q tests/test_hust_strict_cache.py`

Expected: PASS.

- [ ] **Step 5: Commit the loader**

```bash
git add Dataset/HUSTStrict.py Dataset/__init__.py Configs/Dataset/HUSTStrict.yaml tests/test_hust_strict_cache.py
git commit -m "feat: load strict HUST cache with shaft metadata"
```

### Task 3: HUST bearing-frequency evidence

**Files:**
- Create: `Lib/hust_physical_fault_evidence.py`
- Create: `tests/test_hust_physical_evidence.py`

**Interfaces:**
- Produces: `HUST_GEOMETRIES: dict[int, BearingGeometry]` and `HUSTPhysicalEvidenceConfig`.
- Produces: `characteristic_frequencies(shaft_hz: torch.Tensor, geometry: BearingGeometry) -> dict[str, torch.Tensor]`.
- Produces: `build_hust_physical_masks(pseudo_labels: torch.Tensor, target_domain: int, shaft_hz: torch.Tensor, config: HUSTPhysicalEvidenceConfig) -> tuple[torch.Tensor, torch.Tensor]`.
- Reuses: `PhysicalEvidenceConfig`, `construct_physical_destructive_view`, `class_margin`, `robust_fault_evidence_score`, and `mask_active_ratio` from `Lib.physical_fault_evidence`.

- [ ] **Step 1: Write failing geometry, single-fault, composite, and normal tests**

```python
# tests/test_hust_physical_evidence.py
import torch
import pytest
from Lib.hust_physical_fault_evidence import (
    HUST_GEOMETRIES,
    HUSTPhysicalEvidenceConfig,
    build_hust_physical_masks,
    characteristic_frequencies,
)


def test_6205_characteristic_frequencies():
    values = characteristic_frequencies(torch.tensor([25.0]), HUST_GEOMETRIES[0])
    ratio = 7.8 / ((25.0 + 52.0) / 2.0)
    assert values["bpfo"].item() == pytest.approx(0.5 * 9 * 25 * (1 - ratio))
    assert values["bpfi"].item() == pytest.approx(0.5 * 9 * 25 * (1 + ratio))
    assert values["bsf"].item() == pytest.approx((38.5 / (2 * 7.8)) * 25 * (1 - ratio**2))


def test_normal_is_not_applicable_and_composites_union_components():
    cfg = HUSTPhysicalEvidenceConfig(sampling_rate_hz=51200, fft_size=2048, spectrum_length=512)
    shaft = torch.tensor([24.8, 24.8, 24.8, 24.8])
    labels = torch.tensor([0, 1, 3, 4])  # N, I, B, IB
    masks, applicable = build_hust_physical_masks(labels, 0, shaft, cfg)
    assert applicable.tolist() == [False, True, True, True]
    assert torch.count_nonzero(masks[0]) == 0
    expected_union = torch.maximum(masks[1], masks[2])
    assert torch.equal(masks[3] > 0, expected_union > 0)
    assert (masks[3] >= cfg.mask_activity_threshold).sum() <= int(cfg.max_mask_ratio * 512)
```

- [ ] **Step 2: Run the tests and verify missing-module failure**

Run: `pytest -q tests/test_hust_physical_evidence.py`

Expected: FAIL because the HUST evidence module does not exist.

- [ ] **Step 3: Implement documented geometry and predicted-class masks**

```python
from dataclasses import dataclass
from Lib.physical_fault_evidence import PhysicalEvidenceConfig


@dataclass(frozen=True)
class BearingGeometry:
    rolling_elements: int
    rolling_element_diameter_mm: float
    pitch_diameter_mm: float
    contact_angle_deg: float = 0.0


HUST_GEOMETRIES = {
    0: BearingGeometry(9, 7.8, (25.0 + 52.0) / 2.0),
    1: BearingGeometry(9, 9.0, (30.0 + 62.0) / 2.0),
    2: BearingGeometry(9, 11.0, (35.0 + 72.0) / 2.0),
    3: BearingGeometry(9, 12.0, (40.0 + 80.0) / 2.0),
}
@dataclass(frozen=True)
class HUSTPhysicalEvidenceConfig(PhysicalEvidenceConfig):
    ball_sideband_orders: tuple[int, ...] = (0, 1, 2)

FAULT_COMPONENTS = {
    0: (), 1: ("inner",), 2: ("outer",), 3: ("ball",),
    4: ("inner", "ball"), 5: ("inner", "outer"), 6: ("outer", "ball"),
}
```

Implement vectorized characteristic frequencies from shaft Hz. Build per-sample Gaussian masks on the 25 Hz grid using inner sidebands around BPFI, outer sidebands around BPFO, and ball sidebands around BSF with FTF/shaft offsets. Merge composite component masks with `torch.maximum`, then apply a shared `_cap_active_bins()` after union so composite masks obey `max_mask_ratio`. Reject invalid domains, labels, shape mismatches, and non-positive/non-finite shaft values.

- [ ] **Step 4: Run focused evidence tests**

Run: `pytest -q tests/test_hust_physical_evidence.py`

Expected: PASS.

- [ ] **Step 5: Commit the evidence module**

```bash
git add Lib/hust_physical_fault_evidence.py tests/test_hust_physical_evidence.py
git commit -m "feat: add HUST physical fault evidence"
```

### Task 4: Add a metadata hook to the strict 0711 runner and create the HUST route

**Files:**
- Modify: `Lib/hust_strict_protocol.py`
- Modify: `main_tta_0711_strict_randomstream.py`
- Create: `main_tta_0711_hust_strict.py`
- Create: `tests/test_hust_runner_contracts.py`
- Modify: existing relevant PU4D runner tests only if they need to exercise the new optional argument.

**Interfaces:**
- Adds: `Strict0711ResNetTrainer.evidence_batch_metadata(sample_indices) -> object | None`.
- Adds: `Strict0711ResNetTrainer.build_evidence_masks(pseudo_labels, batch_metadata) -> tuple[Tensor, Tensor]`.
- Adds: no-op `offline_diagnostics_update(...)` and `offline_diagnostics_finalize()` extension hooks.
- Changes compatibly: `teacher_reliability(x, iter_num, batch_metadata=None)`.
- Produces: `HUST0711Trainer(..., source_variant: ordinary|robust)`.

- [ ] **Step 1: Write failing hook and route-contract tests**

```python
# tests/test_hust_runner_contracts.py
from pathlib import Path
import ast


def test_generic_0711_runner_has_optional_metadata_hook():
    tree = ast.parse(Path("main_tta_0711_strict_randomstream.py").read_text())
    names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert {"evidence_batch_metadata", "build_evidence_masks"} <= names


def test_hust_runner_routes_two_checkpoint_variants_and_hust_evidence():
    text = Path("main_tta_0711_hust_strict.py").read_text()
    assert "build_hust_physical_masks" in text
    assert 'source_variant not in {"ordinary", "robust"}' in text
    assert "TTA_Model_HUST_STRICT" in text
    assert "shaft_hz" in text


def test_hust_runners_support_non_adaptive_beginning_only_mode():
    for path in ("main_tta_dtcc_hust_strict.py", "main_tta_0711_hust_strict.py"):
        text = Path(path).read_text()
        assert "beginning_only" in text


def test_hust_0711_trainable_allowlist_remains_adapter_and_warp_only():
    text = Path("main_tta_0711_strict_randomstream.py").read_text()
    assert 'allowed = ("band_scale", "band_bias", "warp_ctrl")' in text
```

- [ ] **Step 2: Run contract tests and verify failure**

Run: `pytest -q tests/test_hust_runner_contracts.py`

Expected: FAIL because hooks and HUST runner are missing.

- [ ] **Step 3: Introduce backward-compatible hooks in the PU4D runner**

```python
# main_tta_0711_strict_randomstream.py, inside Strict0711ResNetTrainer
def evidence_batch_metadata(self, sample_indices):
    return None


def build_evidence_masks(self, pseudo_labels, batch_metadata=None):
    return build_physical_masks(
        pseudo_labels=pseudo_labels,
        target_domain=self.target_domain,
        config=self.physical_config,
        geometry=self.geometry,
    )


def offline_diagnostics_update(self, labels, reliability_info):
    return None


def offline_diagnostics_finalize(self):
    return None
```

Change `teacher_reliability` to call `self.build_evidence_masks(pseudo, batch_metadata)`. In `adapt`, retain the third batch field as `sample_indices`, resolve metadata before reliability, call `teacher_reliability(..., batch_metadata=metadata)`, and then call the write-only diagnostic hook after routing is fixed. Call the finalize hook after the stream. All default hooks are no-ops, so PU4D masks and numerical behavior remain unchanged.

- [ ] **Step 4: Implement the HUST subclass and strict checkpoint route**

```python
class HUST0711Trainer(base.Strict0711ResNetTrainer):
    def _checkpoint_path(self):
        variant = str(getattr(self.cfg, "source_variant", "robust"))
        if variant not in {"ordinary", "robust"}:
            raise ValueError(f"invalid source_variant: {variant}")
        return resolve_hust_checkpoint(
            root=Path(getattr(self.cfg, "hust_checkpoint_root", "TTA_Model_HUST_STRICT")),
            variant=variant,
            source=int(self.cfg.Dataset.TL_Task[0]),
            seed=int(self.cfg.seed_run),
            model_name=self.cfg.model_name,
        )

    def _build_physical_config(self):
        return HUSTPhysicalEvidenceConfig(
            sampling_rate_hz=51200.0,
            fft_size=2048,
            spectrum_length=512,
            harmonics=int(_get(self.tcfg, "physical_harmonics", 8)),
            outer_sideband_orders=tuple(_get(self.tcfg, "outer_sideband_orders", [0, 1])),
            inner_sideband_orders=tuple(_get(self.tcfg, "inner_sideband_orders", [0, 1, 2])),
            ball_sideband_orders=tuple(_get(self.tcfg, "ball_sideband_orders", [0, 1, 2])),
            mask_sigma_bins=float(_get(self.tcfg, "mask_sigma_bins", 1.0)),
            background_width_bins=int(_get(self.tcfg, "physical_background_width", 7)),
            max_mask_ratio=float(_get(self.tcfg, "max_mask_ratio", 0.18)),
            mask_activity_threshold=float(_get(self.tcfg, "mask_activity_threshold", 0.10)),
            exclude_dc=True,
        )

    def evidence_batch_metadata(self, sample_indices):
        dataset = self.datasets["target_data"]
        return {"shaft_hz": dataset.shaft_hz[sample_indices.long()].to(self.device)}

    def build_evidence_masks(self, pseudo_labels, batch_metadata=None):
        if not batch_metadata or "shaft_hz" not in batch_metadata:
            raise RuntimeError("HUST shaft metadata missing")
        return build_hust_physical_masks(
            pseudo_labels, self.target_domain, batch_metadata["shaft_hz"], self.physical_config
        )
```

Add `hust_checkpoint_dir()` and `resolve_hust_checkpoint()` to `Lib/hust_strict_protocol.py` so this task compiles independently. Force `Dataset=HUSTStrict`, `Model=ResNet18_1D_SDE`, `use_spectral_adapter=True`, `band_num=256`, `input_len=512`, `min_pcl_classes=3`, `min_ncl_classes=5`, `min_ncl_entries=20`, fixed-random stream, and `passes=1` in the HUST entry point. Implement `beginning_only=True` as initialize, evaluate, print Beginning metrics, and return before optimizer creation.

- [ ] **Step 5: Run HUST and PU4D contract regressions**

Run: `pytest -q tests/test_hust_runner_contracts.py tests/test_fixed_random_stream.py tests/test_runner_contracts.py tests/test_pu4d_vanilla_runner_contracts.py`

Expected: PASS.

- [ ] **Step 6: Commit the runner hook and route**

```bash
git add Lib/hust_strict_protocol.py main_tta_0711_strict_randomstream.py main_tta_0711_hust_strict.py tests/test_hust_runner_contracts.py
git commit -m "feat: route HUST metadata into strict 0711 evidence"
```

### Task 5: Two source-training routes and strict checkpoint contract

**Files:**
- Modify: `Lib/hust_strict_protocol.py`
- Create: `Lib/hust_source_training.py`
- Create: `main_src_dtcc_hust_strict.py`
- Create: `main_src_0711_hust_strict.py`
- Create: `tests/test_hust_source_protocol.py`

**Interfaces:**
- Produces: `hust_checkpoint_dir(root, variant, source, seed) -> Path`.
- Produces: `resolve_hust_checkpoint(...) -> Path` and `strict_load_hust_checkpoint(model, path) -> dict`.
- Produces: `train_hust_source(cfg, source: int, variant: Literal["ordinary", "robust"]) -> dict`.

- [ ] **Step 1: Write failing objective and checkpoint tests**

```python
# tests/test_hust_source_protocol.py
import torch
from Lib.hust_source_training import source_loss


def test_ordinary_loss_uses_only_clean_logits():
    clean = torch.tensor([[2.0, 0.0]], requires_grad=True)
    loss, terms = source_loss("ordinary", clean, torch.tensor([0]), num_classes=2)
    assert set(terms) == {"clean"}
    assert loss.item() == terms["clean"].item()


def test_robust_loss_exposes_all_0711_terms():
    logits = torch.tensor([[2.0, 0.0]], requires_grad=True)
    loss, terms = source_loss(
        "robust",
        logits,
        torch.tensor([0]),
        num_classes=2,
        style_logits=logits + 0.1,
        warp_logits=logits - 0.1,
        style_warp_logits=logits + 0.2,
        clean_feature=torch.tensor([[1.0, 0.0]]),
        augmented_features=[torch.tensor([[0.9, 0.1]])] * 3,
    )
    assert {"clean", "style", "warp", "style_warp", "symmetric_kl", "feature"} == set(terms)
    assert torch.isfinite(loss)


def test_checkpoint_paths_separate_variants(tmp_path):
    from Lib.hust_strict_protocol import hust_checkpoint_dir
    assert hust_checkpoint_dir(tmp_path, "ordinary", 0, 2025) != hust_checkpoint_dir(tmp_path, "robust", 0, 2025)
```

- [ ] **Step 2: Run tests and verify missing source module**

Run: `pytest -q tests/test_hust_source_protocol.py`

Expected: FAIL because `Lib.hust_source_training` does not exist.

- [ ] **Step 3: Implement one source loop with route-specific views**

Build the model with the target carrier enabled, then call `reset_and_freeze_adaptation_carrier(model)` from `Lib.pu4d_vanilla_protocol`. For `ordinary`, compute clean label-smoothed CE only. For `robust`, port the pure tensor logic of `spectral_style_augment`, `spectral_warp_augment`, `symmetric_kl`, and `feature_consistency_loss` into this library module; do not import a training entry point that mutates global trainer classes. Use the approved weights and strengths:

```python
ROBUST_SOURCE_DEFAULTS = {
    "ssp_style_prob": 0.7,
    "ssp_style_strength": 0.15,
    "ssp_style_knots": 8,
    "ssp_lambda_style": 0.5,
    "sde_warp_prob": 0.7,
    "sde_warp_knots": 16,
    "sde_warp_max": 2.0,
    "sde_lambda_warp": 0.5,
    "sde_lambda_style_warp": 0.25,
    "sde_lambda_cons": 0.03,
    "sde_lambda_feat": 0.02,
}
```

The loop loads only the source dataset returned by `HUSTStrict`, never creates a target evaluation loader, trains exactly 50 epochs, saves the epoch-50 state dict plus `source_training_summary.json`, and writes its SHA-256. Metadata must record `target_labels_consumed: false`, route, hyperparameters, carrier identity check, tensor count, source accuracy, and elapsed time.

- [ ] **Step 4: Implement thin Hydra entry points**

Both CLIs parse `only_source`, force the global source constraints, and call `train_hust_source`; their only semantic difference is `variant="ordinary"` versus `variant="robust"`. Reject a missing or out-of-range source and refuse to overwrite a checkpoint whose saved metadata/hash differs. A `smoke_mode=True, smoke_epochs=1` override is accepted only when the checkpoint root name ends in `_SMOKE`; formal roots always force 50 epochs.

- [ ] **Step 5: Run source tests and compile entry points**

Run: `pytest -q tests/test_hust_source_protocol.py tests/test_hust_runner_contracts.py`

Expected: PASS.

Run: `python -m py_compile Lib/hust_source_training.py main_src_dtcc_hust_strict.py main_src_0711_hust_strict.py`

Expected: exit 0.

- [ ] **Step 6: Commit source routes**

```bash
git add Lib/hust_strict_protocol.py Lib/hust_source_training.py main_src_dtcc_hust_strict.py main_src_0711_hust_strict.py tests/test_hust_source_protocol.py
git commit -m "feat: train isolated HUST source checkpoint routes"
```

### Task 6: Strict DtCC HUST target route and offline diagnostics

**Files:**
- Create: `main_tta_dtcc_hust_strict.py`
- Modify: `main_tta_0711_hust_strict.py`
- Modify: `tests/test_hust_runner_contracts.py`

**Interfaces:**
- Produces: `HUSTDtCCTrainer`, inheriting the verified common-source DtCC update order but strictly loading the ordinary checkpoint.
- Produces log records `[HUST OFFLINE DIAGNOSTICS] pseudo_purity=... certain_purity=...` that are never returned to adaptation logic.

- [ ] **Step 1: Add failing checkpoint and trainable-parameter tests**

```python
def test_dtcc_hust_route_uses_only_ordinary_checkpoint():
    text = Path("main_tta_dtcc_hust_strict.py").read_text()
    assert 'variant="ordinary"' in text
    assert "strict_load_hust_checkpoint" in text
    assert "configure_bn_only" in text
    assert "source_variant" not in text


def test_offline_diagnostics_are_write_only():
    for path in ("main_tta_dtcc_hust_strict.py", "main_tta_0711_hust_strict.py"):
        text = Path(path).read_text()
        assert "OfflineDiagnosticsAccumulator" in text
        assert "diagnostics.update" in text
        assert "diagnostics.metrics()" in text


def test_beginning_only_returns_before_optimizer_creation():
    for path in ("main_tta_dtcc_hust_strict.py", "main_tta_0711_hust_strict.py"):
        text = Path(path).read_text()
        assert "beginning_only" in text
```

- [ ] **Step 2: Run and verify missing DtCC route**

Run: `pytest -q tests/test_hust_runner_contracts.py`

Expected: FAIL because the DtCC route does not exist.

- [ ] **Step 3: Implement strict ordinary checkpoint loading and DtCC route**

Implement `HUSTDtCCTrainer` with the verified DtCC update order from `main_tta_dtcc_resnet18_common.py`, but keep it isolated in the new HUST file so the existing untracked/common-source implementation is not rewritten. Load `resolve_hust_checkpoint(..., variant="ordinary")` through `strict_load_hust_checkpoint()`. Force `Dataset=HUSTStrict`, `Model=ResNet18_1D_SDE`, adapter carrier enabled but frozen by `configure_bn_only`, batch size 128, stream seed 2025, `optim_steps=2`, `filter_k=50`, `neighbor_k=5`, `alpha=2.0`, and `ncl_temperature=0.1`. When `beginning_only=True`, evaluate and return before `configure_bn_only()` or optimizer creation.

- [ ] **Step 4: Add write-only diagnostic accumulation**

Implement a small accumulator that stores detached CPU `truth`, `pseudo`, and `certain_mask`; `update()` returns `None`. Call it only after pseudo-label/routing decisions have been computed. Print purity and a 7x7 confusion matrix after the stream; do not read its `metrics()` result inside any loss, optimizer, routing, memory, or scheduler branch.

- [ ] **Step 5: Run runner and DtCC utility regressions**

Run: `pytest -q tests/test_hust_runner_contracts.py tests/test_dtcc_resnet18_common.py tests/test_fixed_random_stream.py`

Expected: PASS.

- [ ] **Step 6: Commit DtCC route and diagnostics**

```bash
git add main_tta_dtcc_hust_strict.py main_tta_0711_hust_strict.py tests/test_hust_runner_contracts.py
git commit -m "feat: add strict HUST DtCC target route"
```

### Task 7: HUST tuning schema, multi-GPU scheduler, and summarizer

**Files:**
- Create: `Configs/Experiments/HUST0711_strict_tuning.yaml`
- Create: `tools/tune_hust_0711_strict.py`
- Create: `tools/summarize_hust_dtcc_0711.py`
- Create: `tests/test_hust_tuning.py`

**Interfaces:**
- Produces: `validate_config`, `discover_idle_gpus`, `candidate_id`, `expand_coordinate_group`, `build_command`, `execute_parallel_tasks`, `rank_candidates`, `freeze_candidate`, and `resume_pipeline`.
- Produces route names `dtcc_ordinary`, `0711_robust`, `0711_common`.
- Produces atomic task state compatible with status values `pending`, `running`, `succeeded`, `failed`.

- [ ] **Step 1: Write failing schema, GPU, command, ranking, and freeze tests**

```python
# tests/test_hust_tuning.py
from pathlib import Path
import yaml
import pytest
from tools.tune_hust_0711_strict import (
    build_command, discover_idle_gpus, freeze_candidate, load_frozen_candidate,
    rank_candidates,
)


@pytest.fixture
def config():
    return yaml.safe_load(Path("Configs/Experiments/HUST0711_strict_tuning.yaml").read_text())


def test_development_and_heldout_partition_all_tasks(config):
    assert config["tasks"]["development"] == [[0, 1], [1, 2], [2, 3], [3, 0]]
    all_tasks = {tuple(x) for x in config["tasks"]["development"] + config["tasks"]["heldout"]}
    assert all_tasks == {(s, t) for s in range(4) for t in range(4) if s != t}


def test_auto_gpu_filter_uses_only_idle_rows():
    rows = "0, RTX, 16311, 15, 0\n1, RTX, 32760, 16247, 99\n2, RTX, 32760, 200, 3\n"
    assert discover_idle_gpus(rows, max_utilization=10, max_memory_fraction=0.10) == [0, 2]


def test_0711_commands_force_strict_invariants(config):
    command = build_command(config, "0711_robust", {"Opt.lr_tar": 0.024}, (0, 1), 2025, gpu=2)
    joined = " ".join(command)
    assert "main_tta_0711_hust_strict.py" in joined
    assert "source_variant=robust" in joined
    assert "TTA0711.passes=1" in joined
    assert "TTA0711.sampling_rate_hz=51200" in joined
    assert "TTA0711.fft_size=2048" in joined
    assert "only_task=[0,1]" in joined


def test_incomplete_candidate_never_ranks():
    rows = [
        {"candidate_id": "a", "task": task, "status": "succeeded", "strict_online": 50.0}
        for task in [(0, 1), (1, 2), (2, 3)]
    ]
    assert rank_candidates(rows, required_tasks={(0, 1), (1, 2), (2, 3), (3, 0)}, baseline={}) == []


def test_heldout_observation_cannot_change_frozen_candidate(tmp_path):
    frozen = freeze_candidate(tmp_path, "abc123", {"Opt.lr_tar": 0.024})
    (tmp_path / "heldout_observation.txt").write_text("0to2,0.0\n")
    assert load_frozen_candidate(frozen)["candidate_id"] == "abc123"
```

- [ ] **Step 2: Run tests and verify missing modules**

Run: `pytest -q tests/test_hust_tuning.py`

Expected: FAIL because tuning and summary modules do not exist.

- [ ] **Step 3: Implement immutable YAML and exact search groups**

The YAML must encode the spec's four development/eight held-out tasks, seeds 2025/2026, 24-hour budget, 0.30 mean-gain guard, 1.00 maximum task regression, route/checkpoint roots, `gpu_policy: auto_idle`, and the ten exact coordinate groups. Fixed overrides include batch 128, passes 1, `sampling_rate_hz=51200`, `fft_size=2048`, `spectrum_length=512`, physical harmonics/sidebands, adapter/F-Warp bounds, `min_pcl_classes=3`, `min_ncl_classes=5`, and `min_ncl_entries=20`.

- [ ] **Step 4: Implement a resumable multi-GPU task engine**

Use `nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits`. Treat a GPU as eligible only when utilization is at most 10% and used memory is at most 10% of total at scheduling time. `--gpus 0,2` overrides auto discovery but still refuses a GPU above either threshold unless `--dry-run`.

Use `ThreadPoolExecutor(max_workers=len(gpus))`; assign one task per GPU, set `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=<physical id>`, and pass `gpu_id=0` inside the isolated process. Write command and `running` state before launch. Implement source, Beginning-only, DtCC, robust-0711, common-source-0711, and supplementary load-audit command builders in the same module. Preserve the PU4D orchestrator's atomic JSON, log validation, transient-CUDA single retry, stale-run resume, complete-candidate filtering, deadline reserve, and no-metric failure behavior.

- [ ] **Step 5: Implement route-aware parsing and reports**

Parse Beginning, Strict Online, macro P/R/F1, post-stream, batches, runtime, peak memory, memory coverage, routing ratios, and offline purity. Write `metrics.csv`, `leaderboard.csv`, `beginning_audit.csv`, `full_pipeline_12task.csv`, `common_source_12task.csv`, `stability_seed2026.csv`, `best_config.yaml`, `checkpoint_manifest.json`, and `report.md`. The report recommendation is true only when the frozen 12-task mean exceeds the untuned 0711 mean; DtCC comparison is a separate field.

- [ ] **Step 6: Run tuning tests and dry-run command generation**

Run: `pytest -q tests/test_hust_tuning.py`

Expected: PASS.

Run: `python tools/tune_hust_0711_strict.py --config Configs/Experiments/HUST0711_strict_tuning.yaml --stage all --dry-run --gpus 0,2`

Expected: writes only isolated command/state previews, covers all required source/baseline gates, and launches no subprocess.

- [ ] **Step 7: Commit tuning and reporting infrastructure**

```bash
git add Configs/Experiments/HUST0711_strict_tuning.yaml tools/tune_hust_0711_strict.py tools/summarize_hust_dtcc_0711.py tests/test_hust_tuning.py
git commit -m "feat: orchestrate strict HUST comparison and tuning"
```

### Task 8: One-command launcher and preflight integration

**Files:**
- Create: `run_hust_dtcc_0711_strict.sh`
- Modify: `tests/test_hust_runner_contracts.py`

**Interfaces:**
- Produces stages `cache`, `source`, `beginning`, `baseline`, `tune`, `final`, `report`, and `all`.
- Accepts environment variables `STAGE`, `GPUS`, `RUN_DIR`, and `DRY_RUN` without mutating unrelated project state.

- [ ] **Step 1: Write failing launcher contract test**

```python
def test_hust_launcher_has_all_gates_and_no_destructive_commands():
    text = Path("run_hust_dtcc_0711_strict.sh").read_text()
    for stage in ("cache", "source", "beginning", "baseline", "tune", "final", "report", "all"):
        assert stage in text
    assert "set -euo pipefail" in text
    assert "DRY_RUN" in text and "GPUS" in text and "RUN_DIR" in text
    assert "rm -rf" not in text
    assert "nvidia-smi" in text
```

- [ ] **Step 2: Run and verify missing launcher failure**

Run: `pytest -q tests/test_hust_runner_contracts.py::test_hust_launcher_has_all_gates_and_no_destructive_commands`

Expected: FAIL because the launcher does not exist.

- [ ] **Step 3: Implement preflight and stage delegation**

The launcher resolves its project directory, runs focused pytest, `py_compile`, `bash -n`, cache validation, checkpoint identity/hash audits, and GPU discovery before delegating to the Python orchestrator. It creates a new `logs/HUST_DTCC_0711_STRICT_$(date -u +%Y%m%d_%H%M%S)` only when `RUN_DIR` is absent. `DRY_RUN=1` prints shell-safe commands and never builds cache, trains, or adapts.

- [ ] **Step 4: Verify shell syntax and dry run**

Run: `bash -n run_hust_dtcc_0711_strict.sh`

Expected: exit 0.

Run: `DRY_RUN=1 STAGE=all GPUS=auto bash run_hust_dtcc_0711_strict.sh`

Expected: preflight passes, commands are printed/written, and no model/cache runtime artifact is created.

- [ ] **Step 5: Commit the launcher**

```bash
git add run_hust_dtcc_0711_strict.sh tests/test_hust_runner_contracts.py
git commit -m "feat: add strict HUST experiment launcher"
```

### Task 9: Build and audit cache; run source smoke tests

**Files:**
- Runtime create: `Dataset/HUST_STRICT_CACHE_V1/`
- Runtime create: a smoke subdirectory under `logs/HUST_DTCC_0711_STRICT_<timestamp>/`

**Interfaces:**
- Consumes all implementation from Tasks 1-8.
- Produces an audited cache manifest and verifies both source routes can take one optimizer step without target-label access.

- [ ] **Step 1: Run the complete focused test suite before material runtime work**

Run: `pytest -q tests/test_hust_strict_cache.py tests/test_hust_physical_evidence.py tests/test_hust_source_protocol.py tests/test_hust_runner_contracts.py tests/test_hust_tuning.py`

Expected: PASS.

- [ ] **Step 2: Build and independently validate the strict cache**

Run: `python tools/build_hust_strict_cache.py --raw-root Dataset/HUST --output Dataset/HUST_STRICT_CACHE_V1 --seed 2025`

Expected: four `domain_<n>.pt` files plus `manifest.json`; each domain has 7 equal classes and 3 equal loads per class, tensors are `[N,1,512]`, and validation passes.

Run: `python -c "from Lib.hust_strict_protocol import validate_cache; print(validate_cache('Dataset/HUST_STRICT_CACHE_V1')['manifest_sha256'])"`

Expected: prints one 64-character SHA-256 and exits 0.

- [ ] **Step 3: Run one ordinary and one robust source smoke epoch in isolated smoke roots**

Run: `python main_src_dtcc_hust_strict.py Model=ResNet18_1D_SDE Dataset=HUSTStrict only_source=0 ++smoke_mode=True ++smoke_epochs=1 ++hust_checkpoint_root=TTA_Model_HUST_STRICT_SMOKE gpu_id=0 process_wandb=False`

Run: `python main_src_0711_hust_strict.py Model=ResNet18_1D_SDE Dataset=HUSTStrict only_source=0 ++smoke_mode=True ++smoke_epochs=1 ++hust_checkpoint_root=TTA_Model_HUST_STRICT_SMOKE gpu_id=0 process_wandb=False`

Expected: each route completes one epoch, saves an isolated identity-carrier checkpoint and metadata with `target_labels_consumed=false`, and the robust log contains all six source-loss terms.

- [ ] **Step 4: Run one real-cache target smoke for each route using smoke checkpoints**

Run: `python main_tta_dtcc_hust_strict.py Model=ResNet18_1D_SDE Dataset=HUSTStrict ++only_task=[0,1] ++seed_runs=[2025] ++stream_seed=2025 ++hust_checkpoint_root=TTA_Model_HUST_STRICT_SMOKE process_wandb=False gpu_id=0`

Run: `python main_tta_0711_hust_strict.py Model=ResNet18_1D_SDE Dataset=HUSTStrict ++only_task=[0,1] ++seed_runs=[2025] ++TTA0711.stream_seed=2025 ++hust_checkpoint_root=TTA_Model_HUST_STRICT_SMOKE ++source_variant=robust process_wandb=False gpu_id=0`

Run: `python main_tta_0711_hust_strict.py Model=ResNet18_1D_SDE Dataset=HUSTStrict ++only_task=[0,1] ++seed_runs=[2025] ++TTA0711.stream_seed=2025 ++hust_checkpoint_root=TTA_Model_HUST_STRICT_SMOKE ++source_variant=ordinary process_wandb=False gpu_id=0`

Expected: all exit 0; DtCC trainable names are BN affine only; both 0711 routes list only adapter/F-Warp names; HUST evidence consumes shaft metadata; online metrics are emitted before updates. These one-epoch-source smoke scores are engineering diagnostics and are not copied into scientific tables.

- [ ] **Step 5: Record runtime audit without committing generated data**

Write cache hash, smoke commands, exit statuses, and logs to the timestamped run directory. Do not `git add` the cache, checkpoints, or logs.

### Task 10: Train both source families and audit Beginning Accuracy

**Files:**
- Runtime create: `TTA_Model_HUST_STRICT/ordinary/source_<0..3>/seed_2025/`
- Runtime create: `TTA_Model_HUST_STRICT/robust/source_<0..3>/seed_2025/`
- Runtime update: timestamped run state/logs and `beginning_audit.csv`.

**Interfaces:**
- Produces eight frozen source checkpoints with strict hashes.
- Produces 24 Beginning Accuracy rows: 12 ordinary and 12 robust.

- [ ] **Step 1: Discover eligible GPUs and launch eight independent source jobs**

Run: `STAGE=source GPUS=auto RUN_DIR=<timestamped-run-dir> bash run_hust_dtcc_0711_strict.sh`

Expected: at most one job per eligible GPU, no occupied GPU is used, four ordinary and four robust epoch-50 checkpoints complete, and each state record is `succeeded`.

- [ ] **Step 2: Audit checkpoint identity, tensor schemas, and hashes**

Run the orchestrator preflight for the beginning stage. Expected: 8/8 checkpoints load strictly into `ResNet18_1D_SDE`, all carrier parameters are zero/identity, checkpoint manifest records unique hashes, and no source metadata reports target-label consumption.

- [ ] **Step 3: Run the 24 Beginning evaluations without target updates**

Run: `STAGE=beginning GPUS=auto RUN_DIR=<timestamped-run-dir> bash run_hust_dtcc_0711_strict.sh`

Expected: every directed task has ordinary and robust `Beginning Acc T`; no optimizer step occurs; results populate `beginning_audit.csv`.

- [ ] **Step 4: Apply the declared soft difficulty audit**

Run the summarizer. Expected: it reports `preferred` for 12/12 in 25%-70%, `acceptable` for at least 9/12 with no score below 15% or above 80%, otherwise `supplementary_load_audit_required`. It never changes a checkpoint or primary domain based on these labels.

- [ ] **Step 5: If required, build only the declared supplementary load split**

When and only when the audit status is `supplementary_load_audit_required`, run `python tools/build_hust_strict_cache.py --raw-root Dataset/HUST --output Dataset/HUST_STRICT_LOAD_CACHE_V1 --seed 2025 --split load`, then invoke `python tools/tune_hust_0711_strict.py --config Configs/Experiments/HUST0711_strict_tuning.yaml --stage load-audit --run-dir <timestamped-run-dir> --gpus auto`. The stage overrides `Dataset.data_path`, `Dataset.TL_list=[0,1,2]`, and checkpoint root `TTA_Model_HUST_STRICT_LOAD`, trains three ordinary and three robust sources, and evaluates the six directed Beginning-only tasks for both variants. Write `supplementary_load_beginning_audit.csv`; do not replace primary bearing artifacts or tune on the load split in this implementation cycle.

### Task 11: Run baselines, tune 0711, freeze, and validate held-out tasks

**Files:**
- Runtime update only: timestamped states, commands, logs, CSV/YAML/JSON outputs.

**Interfaces:**
- Produces complete seed-2025 tables for `dtcc_ordinary`, untuned `0711_robust`, and untuned `0711_common`.
- Produces one frozen universal 0711 robust configuration, held-out results, and matched seed-2026 stability results.

- [ ] **Step 1: Run all three 12-task seed-2025 baselines**

Run: `STAGE=baseline GPUS=auto RUN_DIR=<timestamped-run-dir> bash run_hust_dtcc_0711_strict.sh`

Expected: 36 succeeded task records. Each DtCC run loads an ordinary hash; robust 0711 loads the matching robust source hash; common-source 0711 loads the matching ordinary hash. Before values for routes sharing an ordinary checkpoint match within 0.05 percentage points.

- [ ] **Step 2: Review protocol diagnostics before tuning**

Run the summarizer protocol audit. Expected: every task has one pass, pre-update metrics, exact trainable allowlists, fixed stream seed 2025, complete 7-class target coverage, finite losses, and no failed/missing parsed metric. Stop and use `superpowers:systematic-debugging` if any condition fails.

- [ ] **Step 3: Run staged coordinate tuning on four development tasks**

Run: `STAGE=tune GPUS=auto RUN_DIR=<timestamped-run-dir> bash run_hust_dtcc_0711_strict.sh`

Expected: every complete candidate has four seed-2025 development scores; group winners advance deterministically; failed/partial candidates do not rank; top finalists and baseline complete the four seed-2026 stability tasks; one candidate or the baseline is frozen in `best_config.yaml` before held-out launch.

- [ ] **Step 4: Run frozen held-out seed-2025 tasks and complete seed-2026 stability**

Run: `STAGE=final GPUS=auto RUN_DIR=<timestamped-run-dir> bash run_hust_dtcc_0711_strict.sh`

Expected: eight held-out seed-2025 tasks use the exact frozen config hash. The untuned baseline and frozen candidate both have 12 seed-2026 results. No new candidate ID appears after `best_config.yaml` creation.

- [ ] **Step 5: Resume-test the completed pipeline**

Repeat the final command with the same `RUN_DIR`. Expected: validated succeeded tasks are skipped, no formal log is overwritten, hashes remain unchanged, and summary files reproduce byte-for-byte except explicitly timestamped completion fields.

### Task 12: Final verification and evidence-backed report

**Files:**
- Runtime finalize: `report.md`, comparison CSVs, leaderboard, frozen YAML, manifests.
- Modify only if verification reveals a real defect: focused implementation/test files from Tasks 1-8, using TDD and a separate fix commit.

**Interfaces:**
- Produces the final user-facing conclusion with separate full-pipeline and common-source interpretations.

- [ ] **Step 1: Run fresh focused and regression tests**

Run: `pytest -q tests/test_hust_strict_cache.py tests/test_hust_physical_evidence.py tests/test_hust_source_protocol.py tests/test_hust_runner_contracts.py tests/test_hust_tuning.py tests/test_dtcc_resnet18_common.py tests/test_fixed_random_stream.py tests/test_runner_contracts.py tests/test_pu4d_vanilla_runner_contracts.py tests/test_cwru_runner_contracts.py`

Expected: PASS.

- [ ] **Step 2: Compile Python and validate shell syntax**

Run: `python -m py_compile Lib/hust_strict_protocol.py Lib/hust_physical_fault_evidence.py Lib/hust_source_training.py Dataset/HUSTStrict.py main_src_dtcc_hust_strict.py main_src_0711_hust_strict.py main_tta_dtcc_hust_strict.py main_tta_0711_hust_strict.py tools/build_hust_strict_cache.py tools/tune_hust_0711_strict.py tools/summarize_hust_dtcc_0711.py`

Expected: exit 0.

Run: `bash -n run_hust_dtcc_0711_strict.sh`

Expected: exit 0.

- [ ] **Step 3: Run artifact completeness and protocol audit**

Expected checks: 12/12 seed-2025 rows for all three routes; complete seed-2026 baseline/frozen rows; one frozen config hash; cache and eight source hashes; no labels in adaptive decisions; exact parameter allowlists; complete commands/logs/state; difficulty audit; full/common-source tables; failures and retries explicitly listed.

- [ ] **Step 4: Generate the final report**

Run: `python tools/summarize_hust_dtcc_0711.py --run-dir <timestamped-run-dir> --write-final`

Expected: `report.md` states exact per-task and mean Strict Online Accuracy, macro-F1, Beginning range, tuned-versus-untuned gain, 0711-versus-DtCC full-pipeline result, 0711-versus-DtCC common-source result, stability, runtime/memory, rejected candidates, and whether the tuned configuration is recommended.

- [ ] **Step 5: Invoke verification-before-completion and hand off artifacts**

Re-run the most important tests and report-generation command immediately before claiming success. Link the design, implementation plan, strict cache manifest, best config, final CSVs, and report. State any tasks outside 25%-70%, any negative tuning result, and any incomplete supplementary load audit without softening the conclusion.

---

## Execution Notes

- Never stage runtime cache tensors, checkpoints, logs, or W&B artifacts in Git.
- Before each commit, run `git diff --check` and `git diff --cached --name-only` to ensure unrelated dirty files are excluded.
- If an existing test fails before the related change, record the baseline failure and do not claim it as a regression caused by this work.
- Any unexpected runtime bug triggers `superpowers:systematic-debugging` before a fix; any feature or bugfix implementation follows `superpowers:test-driven-development`.
- Before the final success claim, use `superpowers:verification-before-completion` and cite fresh command output.
