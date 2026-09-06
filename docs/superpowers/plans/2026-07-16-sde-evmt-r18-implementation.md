# SDE-EVMT-R18 Isolated Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不修改当前 BN-first 最佳路线的前提下，实现 ResNet18-1D-GN 的 Source SDE 训练和 R2–R6 严格单遍 EVMT，并完成 PU4D 0→1 实验。

**Architecture:** 新增 `sde_evmt_r18` 独立包。模型内部组合稳健频谱归一化、门控 F-Warp、门控 Spectral Adapter、GN ResNet18 和残差 Feature Adapter；Source 入口训练 R0/R1，Target 入口加载 R1 检查点并通过配置运行 R2–R6。当前 `evmt/` 和 `main_tta_evmt.py` 只作为结果对照，不被导入或修改。

**Tech Stack:** Python 3.10、PyTorch、Hydra/OmegaConf、unittest、JSONL、PU4D `.pt` FFT cache。

## Global Constraints

- 当前最佳文件 `evmt/*.py`、`main_tta_evmt.py`、`Configs/TTA/evmt.yaml` 不得修改。
- 新路线只读取 `Dataset/PU4D_CACHE`，不修改缓存。
- 基础模型固定为 ResNet18-1D-GN；GroupNorm 默认 8 组。
- 输入固定为非负 FFT 幅值 `[B,512]` 或 `[B,1,512]`。
- 目标阶段严格单遍，目标标签仅进入 detached 指标。
- Source/TTA 输出分别写入 `TTA_Model/SDE_EVMT_R18` 和 `outputs/sde_evmt_r18`。
- 每个功能遵循 RED→GREEN→REFACTOR；每个任务独立提交。
- 首轮只运行 source domain 0 和 target 0→1；不运行 0→2 或全部 12 任务。

---

### Task 1: PU4D cache loader and GN model

**Files:**
- Create: `sde_evmt_r18/__init__.py`
- Create: `sde_evmt_r18/data.py`
- Create: `sde_evmt_r18/model.py`
- Create: `tests/sde_evmt_r18/test_model_data.py`

**Interfaces:**
- Produces: `load_pu4d_domain(cache_root, domain, input_kind="fft") -> TensorDataset`
- Produces: `RobustSpectrumNorm(eps=1e-6)`
- Produces: `GatedFrequencyWarp(input_len=512, knots=16, max_warp=2.0, initial_gate=0.05)`
- Produces: `GatedSpectralAdapter(input_len=512, bands=64, delta=0.1, initial_gate=0.1)`
- Produces: `ResidualFeatureAdapter(dim=512, hidden_dim=64, initial_gate=0.01)`
- Produces: `SDEEVMTResNet18(num_classes=32, bottleneck_dim=256, ...)`
- Produces: `model.forward_parts(x, use_feature_adapter=True) -> (feature, logits)`

- [ ] **Step 1: Write failing model/data tests**

```python
def test_robust_norm_is_finite_and_sample_centered(self):
    norm = RobustSpectrumNorm()
    y = norm(torch.rand(4, 512) * 100)
    self.assertEqual(tuple(y.shape), (4, 1, 512))
    self.assertTrue(torch.isfinite(y).all())
    self.assertTrue(torch.allclose(y.median(-1).values, torch.zeros(4, 1), atol=1e-5))

def test_gated_modules_start_near_identity(self):
    x = torch.rand(2, 1, 512)
    warp = GatedFrequencyWarp(initial_gate=0.05)
    adapter = GatedSpectralAdapter(initial_gate=0.1)
    self.assertTrue(torch.allclose(warp(x), x, atol=1e-6))
    self.assertTrue(torch.allclose(adapter(x), x, atol=1e-6))

def test_model_uses_groupnorm_and_returns_expected_shapes(self):
    model = SDEEVMTResNet18(32)
    feature, logits = model.forward_parts(torch.rand(3, 512))
    self.assertEqual(tuple(feature.shape), (3, 256))
    self.assertEqual(tuple(logits.shape), (3, 32))
    self.assertFalse(any(isinstance(m, nn.BatchNorm1d) for m in model.modules()))
```

- [ ] **Step 2: Run RED test**

Run: `../.conda/bin/python -m unittest tests.sde_evmt_r18.test_model_data -v`

Expected: FAIL because `sde_evmt_r18` does not exist.

- [ ] **Step 3: Implement cache loader and model**

`load_pu4d_domain` maps domains 0–3 to the four cache filenames, loads `x/y` without the legacy dataset-wide normalization and validates shape, finiteness and labels 0–31. `RobustSpectrumNorm` applies `log1p(clamp_min(0))`, sample median and IQR.

The model owns all gates and classifier. `forward_parts` returns bottleneck features and logits; `forward` returns logits only. Feature Adapter output linear layer and Warp/Spectral control tensors start at zero, making the initial transform exactly identity even though gate probabilities are nonzero.

- [ ] **Step 4: Run GREEN test**

Run: `../.conda/bin/python -m unittest tests.sde_evmt_r18.test_model_data -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sde_evmt_r18/__init__.py sde_evmt_r18/data.py sde_evmt_r18/model.py tests/sde_evmt_r18/test_model_data.py
git commit -m "feat: add isolated GN ResNet SDE-EVMT model"
```

---

### Task 2: Source augmentations and objective

**Files:**
- Create: `sde_evmt_r18/augment.py`
- Create: `sde_evmt_r18/source.py`
- Create: `tests/sde_evmt_r18/test_source_objective.py`

**Interfaces:**
- Produces: `SourceAugmenter(cfg, generator).views(x) -> dict[str, Tensor]`
- Produces: `symmetric_kl(clean_logits, view_logits) -> Tensor`
- Produces: `feature_consistency(clean_feature, view_feature) -> Tensor`
- Produces: `source_objective(model, x, y, augmenter, cfg) -> SourceLosses`

- [ ] **Step 1: Write failing tests**

Tests assert clean/style/warp/noise shapes, fixed-generator reproducibility, finite gradients and that R0 only uses clean classification while R1 includes all configured branches.

```python
def test_r1_objective_is_finite_and_backward_safe(self):
    losses = source_objective(model, x, y, augmenter, cfg_r1)
    losses.total.backward()
    self.assertTrue(torch.isfinite(losses.total))
    self.assertGreater(float(losses.pred_cons), 0.0)
```

- [ ] **Step 2: Run RED test**

Run: `../.conda/bin/python -m unittest tests.sde_evmt_r18.test_source_objective -v`

Expected: FAIL due to missing modules.

- [ ] **Step 3: Implement augmentations and loss**

Implement probabilistic smooth style, per-sample smooth warp and one randomly selected noise family. Apply one shared mixup permutation/lambda to every view. Source total uses label-smoothed CE, view weights 0.5, symmetric KL 0.2 and normalized feature consistency 0.05.

- [ ] **Step 4: Run GREEN and Task 1 regression**

Run: `../.conda/bin/python -m unittest discover -s tests/sde_evmt_r18 -p 'test_*.py' -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sde_evmt_r18/augment.py sde_evmt_r18/source.py tests/sde_evmt_r18/test_source_objective.py
git commit -m "feat: add generalized source SDE objective"
```

---

### Task 3: Source training entry and checkpoint contract

**Files:**
- Create: `Configs/Model/ResNet18_1D_GN_EVMT.yaml`
- Create: `Configs/SDE_EVMT_R18/source.yaml`
- Create: `main_src_sde_evmt_r18.py`
- Create: `tests/sde_evmt_r18/test_source_entry.py`

**Interfaces:**
- Produces: `train_source(cfg) -> dict`
- Produces checkpoint keys: `model`, `epoch`, `source_accuracy`, `config`, `model_signature`
- Produces output: `TTA_Model/SDE_EVMT_R18/<dataset>/source_<d>/seed_<s>/<variant>/`

- [ ] **Step 1: Write failing checkpoint/output tests**

Use an in-memory tiny dataset and one epoch. Assert unique R0/R1 directories, reload equality, SHA256 and summary fields.

- [ ] **Step 2: Run RED test**

Run: `../.conda/bin/python -m unittest tests.sde_evmt_r18.test_source_entry -v`

Expected: FAIL because entry/config is missing.

- [ ] **Step 3: Implement source entry**

Build deterministic train/eval loaders from the source cache. Train with AdamW and cosine schedule, evaluate the full source stream after every epoch, save the highest source accuracy checkpoint and JSON history. `+smoke_batches=N` limits batches only in smoke mode and is recorded in the summary.

- [ ] **Step 4: Run entry smoke and tests**

Run: `../.conda/bin/python main_src_sde_evmt_r18.py +variant=R0 +source=0 +epochs=1 +smoke_batches=2 process_wandb=false`

Run: `../.conda/bin/python -m unittest discover -s tests/sde_evmt_r18 -p 'test_*.py' -v`

Expected: exit 0 and all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add Configs/Model/ResNet18_1D_GN_EVMT.yaml Configs/SDE_EVMT_R18/source.yaml main_src_sde_evmt_r18.py tests/sde_evmt_r18/test_source_entry.py
git commit -m "feat: add SDE-EVMT-R18 source trainer"
```

---

### Task 4: Target views, evidence and reliability

**Files:**
- Create: `sde_evmt_r18/target_views.py`
- Create: `sde_evmt_r18/evidence.py`
- Create: `sde_evmt_r18/reliability.py`
- Create: `tests/sde_evmt_r18/test_target_reliability.py`

**Interfaces:**
- Produces: `make_teacher_views(x, cfg, generator) -> list[Tensor]` with four views
- Produces: `verify_margin_evidence(forward_fn, spectrum, logits, pseudo, cfg) -> EvidenceResult`
- Produces: `EvidenceReliabilityRouter.route(q_views, margin_drop) -> RouteResult`

- [ ] **Step 1: Write failing tests**

Assert exactly four reproducible task-preserving views, destructive view exclusion from `q`, positive margin behavior, classwise median routing, 3/4 agreement and small predicted classes excluded from memory eligibility.

- [ ] **Step 2: Run RED test**

Run: `../.conda/bin/python -m unittest tests.sde_evmt_r18.test_target_reliability -v`

- [ ] **Step 3: Implement target evidence pipeline**

Use clean Teacher logits for saliency, smooth with kernel 9, select two non-overlapping contiguous bands totaling at most 10%, replace by local median, and compute logit-margin drop. MAD normalization must be finite for constant inputs. Reliability is `confidence*exp(-5*JS)*evidence`.

- [ ] **Step 4: Run GREEN tests**

Run: `../.conda/bin/python -m unittest discover -s tests/sde_evmt_r18 -p 'test_*.py' -v`

- [ ] **Step 5: Commit**

```bash
git add sde_evmt_r18/target_views.py sde_evmt_r18/evidence.py sde_evmt_r18/reliability.py tests/sde_evmt_r18/test_target_reliability.py
git commit -m "feat: add evidence-verified teacher reliability"
```

---

### Task 5: Target memory, prior, losses and scheduler

**Files:**
- Create: `sde_evmt_r18/memory.py`
- Create: `sde_evmt_r18/target_losses.py`
- Create: `sde_evmt_r18/scheduler.py`
- Create: `tests/sde_evmt_r18/test_target_state.py`

**Interfaces:**
- Produces: `ClassBalancedMemory(num_classes, capacity, feature_dim)`
- Produces: `EMATeacherPrior(num_classes, momentum=0.9)`
- Produces: `TargetStage` values `WARMUP`, `EVIDENCE_BUILD`, `FULL`
- Produces: `OnlineScheduler.update(step, candidate_stats, memory_stats, reliability_history)`
- Produces: `target_objective(...) -> TargetLosses`

- [ ] **Step 1: Write failing tests**

Test FIFO balance, reliability weighted prototypes, neighbor similarity gate, nonuniform EMA prior, warm-up for exactly 5 batches and monotonic stage transitions.

- [ ] **Step 2: Run RED test**

Run: `../.conda/bin/python -m unittest tests.sde_evmt_r18.test_target_state -v`

- [ ] **Step 3: Implement state and losses**

Implement MT, weighted Tsallis SEM, KL to EMA prior, certain PCL, uncertain NCL and three adapter regularizers. Every empty branch returns a device-local scalar connected safely to the graph.

- [ ] **Step 4: Run GREEN tests**

Run: `../.conda/bin/python -m unittest discover -s tests/sde_evmt_r18 -p 'test_*.py' -v`

- [ ] **Step 5: Commit**

```bash
git add sde_evmt_r18/memory.py sde_evmt_r18/target_losses.py sde_evmt_r18/scheduler.py tests/sde_evmt_r18/test_target_state.py
git commit -m "feat: add staged SDE-EVMT target state"
```

---

### Task 6: Strict online runner and parameter gates

**Files:**
- Create: `sde_evmt_r18/runner.py`
- Create: `tests/sde_evmt_r18/test_online_runner.py`

**Interfaces:**
- Produces: `configure_adaptation(model, variant, stage) -> list[str]`
- Produces: `build_target_optimizer(model, cfg) -> Optimizer`
- Produces: `SDEEVMTOnlineRunner.step(x, y_for_metrics=None) -> BatchMetrics`

- [ ] **Step 1: Write failing strict-order tests**

Use spies to assert score occurs before optimizer/memory/EMA, optimizer steps at most once, labels do not change state, R2–R6 parameter sets are exact, and a non-finite loss skips all writes.

- [ ] **Step 2: Run RED test**

Run: `../.conda/bin/python -m unittest tests.sde_evmt_r18.test_online_runner -v`

- [ ] **Step 3: Implement runner**

Score with `torch.no_grad()` before any current-Batch update. Warm-up disables feature adapter gradients and memory. Evidence-build collects candidate stats. Full enables configured R5/R6 branches. Teacher EMA updates only adaptable parameters and never classifier or convolution weights.

- [ ] **Step 4: Run full unit suite**

Run: `../.conda/bin/python -m unittest discover -s tests/sde_evmt_r18 -p 'test_*.py' -v`

- [ ] **Step 5: Commit**

```bash
git add sde_evmt_r18/runner.py tests/sde_evmt_r18/test_online_runner.py
git commit -m "feat: enforce strict SDE-EVMT online updates"
```

---

### Task 7: Target entry, variants and isolated outputs

**Files:**
- Create: `Configs/SDE_EVMT_R18/target.yaml`
- Create: `main_tta_sde_evmt_r18.py`
- Create: `scripts/run_pu4d_sde_evmt_r18.sh`
- Create: `tests/sde_evmt_r18/test_target_entry.py`

**Interfaces:**
- Produces variants `R2`, `R3`, `R4`, `R5`, `R6`
- Produces output `outputs/sde_evmt_r18/<task>/seed_<seed>/<variant>/`
- Produces `config.yaml`, `batches.jsonl`, `summary.json`, `run.log`

- [ ] **Step 1: Write failing variant/output tests**

Assert exact R2–R6 switches, unknown variant rejection, checkpoint signature verification and unique output directories.

- [ ] **Step 2: Run RED test**

Run: `../.conda/bin/python -m unittest tests.sde_evmt_r18.test_target_entry -v`

- [ ] **Step 3: Implement entry and script**

Load only the R1 Source checkpoint with matching model signature. Record checkpoint SHA256, 160,299 seen samples, 314 Batch rows, one pass, updated/skipped counts, Accuracy, Macro-F1, final stage, gate values, elapsed time and peak GPU memory. Remove stale summary before starting.

- [ ] **Step 4: Run smoke and complete regression**

Run: `../.conda/bin/python main_tta_sde_evmt_r18.py +variant=R2 +only_task='[0,1]' +smoke_batches=2`

Run: `bash -n scripts/run_pu4d_sde_evmt_r18.sh`

Run: `../.conda/bin/python -m unittest discover -s tests/sde_evmt_r18 -p 'test_*.py' -v`

- [ ] **Step 5: Commit**

```bash
git add Configs/SDE_EVMT_R18/target.yaml main_tta_sde_evmt_r18.py scripts/run_pu4d_sde_evmt_r18.sh tests/sde_evmt_r18/test_target_entry.py
git commit -m "feat: add reproducible SDE-EVMT-R18 variants"
```

---

### Task 8: Source training, 0→1 experiment and report

**Files:**
- Create: `logs/实验日志_SDE_EVMT_R18.md`

- [ ] **Step 1: Verify code before GPU work**

Run: `../.conda/bin/python -m unittest discover -s tests/sde_evmt_r18 -p 'test_*.py' -v`

Run: `../.conda/bin/python -m compileall -q sde_evmt_r18 main_src_sde_evmt_r18.py main_tta_sde_evmt_r18.py`

Expected: all tests PASS and compileall exit 0.

- [ ] **Step 2: Run Source smoke matrix**

Run R0 and R1 for 2 epochs/20 batches. Stop if loss is non-finite, source accuracy does not exceed random chance, or checkpoint reload differs.

- [ ] **Step 3: Train full R1 Source 0**

Run 80 epochs. Monitor train loss, full-source accuracy and gate values. Continue only while finite; save best source checkpoint without target feedback.

- [ ] **Step 4: Run R2–R6 sequentially on 0→1**

After each run verify 314 JSONL rows, 160,299 samples, one pass, finite metrics and no more than 10% skipped batches. Stop later variants on structural failure.

- [ ] **Step 5: Freeze decision and reveal metrics**

Use only unlabeled stability, effective classes, reliability, memory coverage, gate trajectories, runtime and skipped updates to identify stable variants. Then report Accuracy/Macro-F1 and compare with BN-first Full 45.0964%/44.3556%. Do not tune using target labels.

- [ ] **Step 6: Write experiment log and verify summaries**

Record exact commands, commits, checkpoint SHA256, Source curve, R2–R6 metrics, failures and decision. Parse each JSONL back into its summary totals.

- [ ] **Step 7: Commit report**

```bash
git add logs/实验日志_SDE_EVMT_R18.md scripts/run_pu4d_sde_evmt_r18.sh
git commit -m "exp: record SDE-EVMT-R18 PU4D results"
```
