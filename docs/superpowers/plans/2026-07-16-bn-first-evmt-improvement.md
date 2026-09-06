# BN-first EVMT Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 PU4D 的 ResNet18_1D_SDE 上实现严格滞后一批的 BN-first EVMT，并完成 0→1 单变量矩阵以及通过稳定性门槛后的 0→2 校准。

**Architecture:** 新增 `TargetBNController` 独占源/目标 BN 统计，在线运行器以“历史统计预测当前批、当前批只更新未来状态”的顺序执行。Student 只训练 BN affine、Adapter 和 Warp；阶段控制器根据 batch 数、预测类别覆盖和目标记忆覆盖依次启用 BN warm-up、证据建库和完整 PCL/NCL，并由独立塌缩保护器决定是否允许更新。

**Tech Stack:** Python 3.10、PyTorch、Hydra/OmegaConf、unittest/pytest、JSONL、PU4D FFT cache。

## Global Constraints

- 基础模型固定为 `ResNet18_1D_SDE`，不使用 ViT1D 或 prompt。
- PU4D 固定为32类四域 `[0,1,2,3]`，旧三域 PU 不参与本轮实验。
- 目标数据严格单遍；当前 batch 必须先用截至上一批的状态预测计分，再更新 BN 历史和参数。
- 目标标签不得参与梯度、BN统计、路由、阈值、阶段切换、回滚或参数选择。
- Student 只训练 BN `weight/bias`、`band_scale`、`band_bias`、`warp_ctrl`。
- 首轮只运行 0→1；无标签稳定性通过且配置冻结后才运行 0→2，不运行全12任务。
- 保留 dirty worktree 中的用户内容；每次提交只包含本计划列出的文件。
- 每项实现遵循 TDD：失败测试、最小实现、通过测试、独立提交。

---

## File Map

- Create `evmt/bn.py`: BN 层发现、源统计快照、在线目标矩合并和融合统计安装。
- Create `evmt/stages.py`: 三阶段状态机及切换条件。
- Create `evmt/guards.py`: 无标签预测塌缩检测。
- Create `evmt/regularization.py`: BN-anchor、Adapter 和 Warp 正则。
- Modify `evmt/ema.py`: 可适配状态扩展到 BN affine，同时排除 BN running statistics。
- Modify `evmt/reliability.py`: 支持无 evidence 的 bootstrap 路由。
- Modify `evmt/memory.py`: 暴露目标记忆覆盖和条目统计。
- Modify `evmt/runner.py`: 严格预测顺序、BN控制、三阶段、保护、正则和扩展指标。
- Modify `main_tta_evmt.py`: 构建 BN 控制器、多参数组优化器、方法开关和唯一输出目录。
- Modify `Configs/TTA/evmt.yaml`: BN-first 默认值与各分支开关。
- Create `scripts/run_pu4d_bnfirst_calibration.sh`: 0→1 单变量矩阵及条件式 0→2 命令。
- Create/modify `tests/evmt/*.py`: 单元、协议和入口集成测试。
- Modify `logs/实验日志_EVMT_DtCC.md`: 诊断证据、命令、结果和决策。

---

### Task 1: TargetBNController and lagged statistics

**Files:**
- Create: `evmt/bn.py`
- Create: `tests/evmt/test_bn_controller.py`

**Interfaces:**
- Produces: `TargetBNController(model, blend_batches=5, max_target_weight=0.9, eps=1e-5)`
- Produces: `prediction_stats() -> context manager`
- Produces: `observe_batch(model, x, forward_fn) -> None`
- Produces: `summary() -> dict[str, float | int]`

- [ ] **Step 1: Write failing tests for source immutability and online moments**

```python
class BNControllerTests(unittest.TestCase):
    def test_online_moments_match_concatenated_tensor(self):
        model = nn.Sequential(nn.BatchNorm1d(2))
        ctl = TargetBNController(model, blend_batches=1, max_target_weight=1.0)
        x1 = torch.tensor([[[1., 3.]], [[3., 5.]]])
        x2 = torch.tensor([[[5., 7.]], [[7., 9.]]])
        ctl.observe_batch(model, x1, lambda m, x: m(x))
        ctl.observe_batch(model, x2, lambda m, x: m(x))
        expected = torch.cat([x1, x2], 0).transpose(0, 1).reshape(2, -1)
        state = ctl.layer_state("0")
        self.assertTrue(torch.allclose(state.target_mean, expected.mean(1)))
        self.assertTrue(torch.allclose(state.target_var, expected.var(1, unbiased=False)))

    def test_current_batch_does_not_change_prediction_context(self):
        model = nn.Sequential(nn.BatchNorm1d(2))
        ctl = TargetBNController(model)
        before = ctl.fused_state("0").mean.clone()
        with ctl.prediction_stats():
            _ = model(torch.randn(8, 2, 4))
        self.assertTrue(torch.equal(before, ctl.fused_state("0").mean))
        self.assertTrue(torch.equal(ctl.source_state("0").mean,
                                    model[0].running_mean))
```

- [ ] **Step 2: Run tests and confirm import failure**

Run: `../.conda/bin/python -m pytest -q tests/evmt/test_bn_controller.py`

Expected: FAIL because `evmt.bn` does not exist.

- [ ] **Step 3: Implement layer state and parallel-moment merge**

```python
@dataclass
class LayerMoments:
    source_mean: Tensor
    source_var: Tensor
    target_mean: Tensor | None = None
    target_m2: Tensor | None = None
    target_count: int = 0

def merge_moments(mean, m2, count, batch_mean, batch_var, batch_count):
    if count == 0:
        return batch_mean, batch_var * batch_count, batch_count
    total = count + batch_count
    delta = batch_mean - mean
    new_mean = mean + delta * (batch_count / total)
    new_m2 = m2 + batch_var * batch_count + delta.square() * count * batch_count / total
    return new_mean, new_m2, total
```

Implement `observe_batch` with temporary forward hooks on every `_BatchNorm` input. Hooks reduce all dimensions except channel; they update controller-owned tensors only after the forward finishes. Restore every BN module's original training flag and running statistics in `finally`.

- [ ] **Step 4: Implement lagged fused-stat context**

```python
def target_weight(self):
    return min(self.max_target_weight,
               self.max_target_weight * self.seen_batches / self.blend_batches)

@contextmanager
def prediction_stats(self):
    snapshots = self._module_snapshots()
    try:
        self._install_fused_stats()  # only observations from earlier batches
        yield
    finally:
        self._restore_module_snapshots(snapshots)
```

Use total-variance fusion:
`var = (1-w)*(src_var + (src_mean-mix_mean)^2) + w*(tar_var + (tar_mean-mix_mean)^2)`.

- [ ] **Step 5: Run Task 1 tests**

Run: `../.conda/bin/python -m pytest -q tests/evmt/test_bn_controller.py`

Expected: all tests PASS.

- [ ] **Step 6: Commit Task 1**

```bash
git add evmt/bn.py tests/evmt/test_bn_controller.py
git commit -m "feat: add lagged target BN controller"
```

---

### Task 2: Adaptable BN state, optimizer groups, and regularizers

**Files:**
- Create: `evmt/regularization.py`
- Modify: `evmt/ema.py`
- Modify: `main_tta_evmt.py`
- Create: `tests/evmt/test_bn_adaptation.py`

**Interfaces:**
- Produces: `adaptable_state(model, include_bn_affine=True) -> dict[str, Parameter]`
- Produces: `freeze_bn_adapter_warp(model) -> list[str]`
- Produces: `build_bnfirst_optimizer(model, cfg) -> Optimizer`
- Produces: `bn_anchor_loss(model, anchor)`, `adapter_reg_loss(model)`, `warp_reg_loss(model)`

- [ ] **Step 1: Write failing trainable-state and EMA tests**

```python
def test_only_bn_affine_adapter_and_warp_are_trainable(self):
    names = freeze_bn_adapter_warp(self.model)
    self.assertTrue(any(name.endswith("bn1.weight") for name in names))
    self.assertIn("0.band_scale", names)
    self.assertIn("0.warp_ctrl", names)
    self.assertFalse(self.model[0].stem[0].weight.requires_grad)
    self.assertFalse(self.model[2].fc.weight.requires_grad)

def test_ema_updates_bn_affine_but_not_running_stats(self):
    before = self.teacher[0].stem[1].running_mean.clone()
    self.student[0].stem[1].weight.data.add_(1)
    ema_update_(self.teacher, self.student, beta=0.5, include_bn_affine=True)
    self.assertTrue(torch.equal(before, self.teacher[0].stem[1].running_mean))
    self.assertFalse(torch.equal(self.student[0].stem[1].weight,
                                 self.teacher[0].stem[1].weight))
```

- [ ] **Step 2: Run focused test and confirm failure**

Run: `../.conda/bin/python -m pytest -q tests/evmt/test_bn_adaptation.py`

Expected: FAIL because new signatures/functions are absent.

- [ ] **Step 3: Extend adaptable state without matching running buffers**

Recognize BN affine by enumerating named BN modules and adding `<module>.weight` and `<module>.bias`; continue recognizing names ending in `band_scale`, `band_bias`, `warp_ctrl`. `ema_update_` must compare exact parameter-key sets and never iterate buffers.

- [ ] **Step 4: Build three optimizer groups**

```python
groups = [
    {"params": bn_params, "lr": base_lr * cfg.bn_lr_scale, "weight_decay": wd},
    {"params": adapter_params, "lr": base_lr * cfg.adapter_lr_scale, "weight_decay": wd},
    {"params": warp_params, "lr": base_lr * cfg.warp_lr_scale, "weight_decay": wd},
]
return torch.optim.AdamW([group for group in groups if group["params"]])
```

- [ ] **Step 5: Implement regularizers and anchors**

Capture immutable BN affine anchors before TTA. Adapter regularizer uses the actual transformed scale/bias; Warp regularizer uses interpolated `tanh(warp_ctrl)` with L2 plus first-difference smoothness. Every empty branch must return a scalar tensor on the model device.

- [ ] **Step 6: Run Task 2 and existing EMA tests**

Run: `../.conda/bin/python -m pytest -q tests/evmt/test_bn_adaptation.py tests/evmt/test_ema_views.py`

Expected: all tests PASS.

- [ ] **Step 7: Commit Task 2**

```bash
git add evmt/ema.py evmt/regularization.py main_tta_evmt.py tests/evmt/test_bn_adaptation.py
git commit -m "feat: adapt BN affine with anchored regularization"
```

---

### Task 3: Stage controller, evidence bypass, and memory statistics

**Files:**
- Create: `evmt/stages.py`
- Modify: `evmt/reliability.py`
- Modify: `evmt/memory.py`
- Create: `tests/evmt/test_stages_routing.py`

**Interfaces:**
- Produces: `AdaptationStage` enum values `BN_WARMUP`, `EVIDENCE_BUILD`, `FULL`
- Produces: `StageController.update(step, recent_coverage, memory_coverage, min_support) -> AdaptationStage`
- Produces: `ReliabilityRouter.route(q, view_js, margin_drop=None) -> RouteResult`
- Produces: `ClassBalancedMemory.stats() -> MemoryStats`

- [ ] **Step 1: Write failing transition and evidence-bypass tests**

```python
def test_stage_never_forces_full_without_memory(self):
    stages = StageController(warmup_batches=20, coverage_window=5,
                             min_pred_classes=24, min_memory_classes=16,
                             min_entries_per_class=2)
    self.assertEqual(stages.update(20, 23, 0, 0), AdaptationStage.BN_WARMUP)
    self.assertEqual(stages.update(21, 24, 0, 0), AdaptationStage.EVIDENCE_BUILD)
    self.assertEqual(stages.update(100, 32, 15, 32), AdaptationStage.EVIDENCE_BUILD)
    self.assertEqual(stages.update(101, 32, 16, 2), AdaptationStage.FULL)

def test_route_without_evidence_uses_confidence_and_agreement(self):
    out = self.router.route(self.q, self.js, margin_drop=None)
    self.assertTrue(torch.allclose(out.reliability,
                    out.confidence * out.agreement))
```

- [ ] **Step 2: Confirm tests fail**

Run: `../.conda/bin/python -m pytest -q tests/evmt/test_stages_routing.py`

Expected: FAIL because `evmt.stages` and optional evidence are absent.

- [ ] **Step 3: Implement monotonic stage transitions**

Stage transitions are one-way. Stage B requires `step > warmup_batches` and recent coverage. Stage C requires both target memory class coverage and per-used-class minimum support. If conditions disappear after a transition, retain the current stage unless the collapse guard skips the update.

- [ ] **Step 4: Add optional evidence and memory stats**

When `margin_drop is None`, set `evidence=ones_like(confidence)` and omit the positive-margin gate. `MemoryStats` returns total entries, covered classes, minimum positive queue size and maximum queue size without exposing labels.

- [ ] **Step 5: Run routing, memory, and stage tests**

Run: `../.conda/bin/python -m pytest -q tests/evmt/test_stages_routing.py tests/evmt/test_reliability_memory.py`

Expected: all tests PASS.

- [ ] **Step 6: Commit Task 3**

```bash
git add evmt/stages.py evmt/reliability.py evmt/memory.py tests/evmt/test_stages_routing.py
git commit -m "feat: gate EVMT stages by prediction and memory coverage"
```

---

### Task 4: Collapse guard and strict runner integration

**Files:**
- Create: `evmt/guards.py`
- Modify: `evmt/runner.py`
- Modify: `tests/evmt/test_online_runner.py`
- Create: `tests/evmt/test_collapse_guard.py`

**Interfaces:**
- Produces: `CollapseGuard(num_classes, min_effective_classes=8, max_class_share=0.5, max_relative_drop=0.5)`
- Changes: `EVMTOnlineRunner(..., bn_controller, source_anchors=None)`
- Changes: `BatchMetrics` adds stage, BN, coverage, memory, regularizer, gradient, displacement and skip fields.

- [ ] **Step 1: Write failing guard tests**

```python
def test_guard_rejects_single_class_collapse(self):
    logits = torch.full((64, 32), -20.0); logits[:, 0] = 20.0
    decision = CollapseGuard(32).check(logits)
    self.assertFalse(decision.allow_update)
    self.assertIn("max_class_share", decision.reasons)

def test_guard_accepts_balanced_finite_logits(self):
    logits = torch.eye(32).repeat(2, 1) * 5
    self.assertTrue(CollapseGuard(32).check(logits).allow_update)
```

- [ ] **Step 2: Extend the online-order regression test**

Use a spy BN controller whose `observe_batch` records calls. Assert the runner records `correct` before the first observation and calls the optimizer at most once. Add a second test where the guard rejects; optimizer, memory, and EMA call counts must remain zero.

- [ ] **Step 3: Run tests and confirm failure**

Run: `../.conda/bin/python -m pytest -q tests/evmt/test_collapse_guard.py tests/evmt/test_online_runner.py`

Expected: FAIL because guard and BN-aware runner are absent.

- [ ] **Step 4: Implement guard metrics**

Compute `marginal=softmax(logits).mean(0)`, `effective_classes=exp(entropy(marginal))`, `max_class_share=marginal.max()`, and an EMA stable reference. Reasons are deterministic strings; non-finite logits/loss/gradients always reject.

- [ ] **Step 5: Refactor runner into explicit phases**

```python
with self.bn.prediction_stats():
    feature, logits = forward_parts(self.student, x)
pre_logits = logits.detach()
correct = metric_only(pre_logits, y_for_metrics)
self.bn.observe_batch(self.student, x, forward_parts)
stage = self.stages.update(...)
teacher_q, route = self._teacher_route(x, stage)
losses = self._losses(x, feature, logits, route, stage)
decision = self.guard.check(pre_logits, losses.total)
if decision.allow_update:
    self._backward_step_write_memory_ema(...)
```

Recompute Student adaptation logits after `observe_batch`; do not backpropagate through the scored logits. Skip evidence completely in stage A or when `use_evidence=false`. Enable PCL/NCL only in stage C. Add BN/Adapter/Warp regularizers in every stage.

- [ ] **Step 6: Add detached logging fields**

Include `pred_histogram` length32, `pred_coverage`, `effective_classes`, `max_class_share`, `memory_entries`, `memory_coverage`, `bn_target_weight`, `bn_seen_batches`, `reg_adapter`, `reg_warp`, `reg_bn`, group gradient norms, group displacement norms, `skip_reasons`, and `updated`.

- [ ] **Step 7: Run Task 4 and full EVMT tests**

Run: `../.conda/bin/python -m pytest -q tests/evmt`

Expected: all tests PASS.

- [ ] **Step 8: Commit Task 4**

```bash
git add evmt/guards.py evmt/runner.py tests/evmt/test_collapse_guard.py tests/evmt/test_online_runner.py
git commit -m "feat: enforce strict BN-first online adaptation"
```

---

### Task 5: Configuration, entry point, and reproducible method variants

**Files:**
- Modify: `Configs/TTA/evmt.yaml`
- Modify: `main_tta_evmt.py`
- Modify: `tests/evmt/test_device_config.py`
- Create: `tests/evmt/test_method_variants.py`

**Interfaces:**
- Produces: method variants `bn_stat`, `bn_affine`, `bn_mt_adapter`, `bnfirst_full`
- Produces: one unique run directory per task/seed/method with `summary.json` and `batches.jsonl`

- [ ] **Step 1: Write failing variant tests**

```python
def test_bn_stat_has_no_optimizer_updates(self):
    cfg = method_overrides("bn_stat")
    self.assertTrue(cfg.use_bn_stats)
    self.assertFalse(cfg.train_bn_affine)
    self.assertFalse(cfg.use_mt)
    self.assertFalse(cfg.use_evidence)

def test_full_enables_state_driven_components(self):
    cfg = method_overrides("bnfirst_full")
    self.assertTrue(cfg.train_bn_affine)
    self.assertTrue(cfg.use_evidence)
    self.assertTrue(cfg.use_pcl)
    self.assertTrue(cfg.use_ncl)
```

- [ ] **Step 2: Confirm test fails**

Run: `../.conda/bin/python -m pytest -q tests/evmt/test_method_variants.py`

Expected: FAIL because variants are absent.

- [ ] **Step 3: Add explicit defaults**

Add `method`, `use_bn_stats`, `train_bn_affine`, `bn_blend_batches=5`, `bn_target_weight_max=0.9`, `bn_eps=1e-5`, `bn_lr_scale=0.1`, `lambda_bn=0.001`, `min_pred_classes=24`, `min_memory_classes=16`, `min_entries_per_class=2`, `coverage_window=5`, `min_effective_classes=8`, `max_class_share=0.5`, `max_effective_drop=0.5`.

- [ ] **Step 4: Implement method override mapping and output isolation**

`method_overrides(name)` returns a copied config fragment; reject unknown names. The run directory becomes `outputs/evmt/bnfirst_calibration/<src>-<tar>/seed_<seed>/<method>/`. A complete summary must include `method`, `seen_samples`, `passes`, `updated_batches`, `skipped_batches`, `final_stage`, `checkpoint_sha256`, elapsed time and peak GPU memory.

- [ ] **Step 5: Run entry and full tests**

Run: `../.conda/bin/python -m pytest -q tests/evmt`

Run: `../.conda/bin/python -m compileall -q evmt main_tta_evmt.py`

Expected: all tests PASS and compileall exit code0.

- [ ] **Step 6: Commit Task 5**

```bash
git add Configs/TTA/evmt.yaml main_tta_evmt.py tests/evmt/test_device_config.py tests/evmt/test_method_variants.py
git commit -m "feat: add reproducible BN-first calibration variants"
```

---

### Task 6: Calibration script, experiment log, and 0→1 run matrix

**Files:**
- Create: `scripts/run_pu4d_bnfirst_calibration.sh`
- Modify: `logs/实验日志_EVMT_DtCC.md`

**Interfaces:**
- Consumes: Task 5 variants and source checkpoint `TTA_Model/PU4D0.0001/source_0/seed_2025/best_source_ResNet18_1D_SDE2025fft_Linear.pt`
- Produces: four new 0→1 summaries and, after the gate, four 0→2 summaries.

- [ ] **Step 1: Write the shell script with exact common arguments**

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
COMMON=(Model=ResNet18_1D_SDE Dataset=PU4D
  Dataset.data_path=/home/std04/Projects/DtCC_FOA_ViT_v1/DtCC_FOA_V1/Dataset/PU4D_CACHE
  TTA=evmt Opt.lr_src=0.0001 +seed_run=2025 batch_size=512
  num_workers=4 process_wandb=false gpu_id=0)
for method in bn_stat bn_affine bn_mt_adapter bnfirst_full; do
  ../.conda/bin/python main_tta_evmt.py "${COMMON[@]}" \
    +only_task='[0,1]' TTA.method="$method"
done
```

The script must skip only when a parsed summary exists with `seen_samples=160299`, `passes=1`, matching method and checkpoint hash. Do not infer success from directory existence.

- [ ] **Step 2: Validate shell and run all tests before GPU work**

Run: `bash -n scripts/run_pu4d_bnfirst_calibration.sh`

Run: `../.conda/bin/python -m pytest -q tests/evmt`

Expected: shell syntax clean and all tests PASS.

- [ ] **Step 3: Run 0→1 methods sequentially and monitor every run**

Run: `bash scripts/run_pu4d_bnfirst_calibration.sh`

After each method verify: exit0, 314 JSONL rows, 160,299 samples, one pass, finite metrics, no target-label-derived field in config or update decisions. Stop the matrix on structural failure, NaN/Inf, or more than10% skipped batches.

- [ ] **Step 4: Apply the label-free stability gate**

Freeze one configuration using only finite metrics, predicted-class coverage, effective classes, memory coverage, skipped updates, runtime and GPU memory. Then read online accuracy/macro-F1 for offline reporting. If Full is worse than `bn_mt_adapter`, report the result and do not silently tune against target labels.

- [ ] **Step 5: Conditionally run 0→2**

Only if the frozen 0→1 configuration is structurally stable and improves the declared baselines, rerun the same method matrix with `+only_task='[0,2]'`; do not alter thresholds or learning rates.

- [ ] **Step 6: Update Chinese experiment log**

Record diagnostic evidence 18.33→42.21, commits, exact commands, checkpoint SHA256, per-method metrics, protocol checks, failures and the decision whether to proceed to source retraining or global scale design.

- [ ] **Step 7: Final verification and experiment commit**

Run: `../.conda/bin/python -m pytest -q tests/evmt`

Run: `git diff --check`

Verify each completed summary against its JSONL totals with a local parser.

```bash
git add scripts/run_pu4d_bnfirst_calibration.sh logs/实验日志_EVMT_DtCC.md
git commit -m "exp: record BN-first PU4D calibration"
```
