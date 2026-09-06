# EVMT-DtCC PU4D 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 在 PU4D 四域上实现基于 ResNet18_1D_SDE 的严格单遍在线 EVMT-DtCC，重训可信源模型，并完成两任务校准与12任务正式实验。

**架构：** 保留现有 Source/TTA 文件作为基线，在 `evmt/` 中按多视图、证据、路由/记忆、损失、在线运行器拆分。新入口 `main_tta_evmt.py` 只编排模块；所有方法共用新源检查点和严格的“先预测、后更新”协议。

**技术栈：** Python 3.10、PyTorch、Hydra/OmegaConf、NumPy、pandas、pytest、PU4D FFT 缓存。

## 全局约束

- 基础模型固定为 `ResNet18_1D_SDE`，不使用 ViT1D 或 prompt。
- TTA 只训练 `band_scale`、`band_bias`、`warp_ctrl`。
- 目标数据严格单遍；每个 batch 先预测计分，再无标签更新一次。
- 目标标签不得参与适配、阈值、早停、回滚或超参数选择。
- 先固定种子在 0→1、0→2 校准；冻结参数后再跑12任务。
- 正式实验使用三个固定种子，报告校准2任务、其余10任务和全部12任务。
- 现有 dirty worktree 内容视为用户工作，不恢复、不覆盖、不纳入无关提交。
- 每个任务遵循 TDD：先失败测试，再最小实现，再完整验证，再独立提交。

---

## 文件结构

- 创建 `Configs/Model/ResNet18_1D_SDE.yaml`：ResNet18-SDE 正式模型配置。
- 创建 `Configs/TTA/evmt.yaml`：EVMT 默认超参数和方法开关。
- 创建 `evmt/views.py`：弱标签保持视图。
- 创建 `evmt/evidence.py`：显著性连续频带遮挡与 PLPD/margin drop。
- 创建 `evmt/reliability.py`：稳健归一化、门槛和类别均衡路由。
- 创建 `evmt/memory.py`：类别均衡 FIFO、近邻和原型。
- 创建 `evmt/losses.py`：加权 SEM、PCL、NCL、MT。
- 创建 `evmt/ema.py`：仅可适配状态的 EMA。
- 创建 `evmt/logging.py`：JSONL、CSV、运行清单和中文实验日志。
- 创建 `evmt/runner.py`：严格在线单 batch 编排。
- 创建 `main_tta_evmt.py`：Hydra 入口和12任务枚举。
- 修改 `main_Src_SDE.py`：支持按4个唯一源域训练和共享检查点。
- 创建 `scripts/run_pu4d_source.sh`、`scripts/run_pu4d_calibration.sh`、`scripts/run_pu4d_full.sh`：实验编排。
- 创建 `tools/summarize_evmt.py`：结果汇总。
- 创建 `tests/evmt/`：上述模块的单元与集成测试。
- 创建 `logs/实验日志_EVMT_DtCC.md`：中文实验记录。
- 创建 `docs/cleanup/pu4d-cleanup-manifest.md`：只记录清理候选，不在实现阶段自动删除。

---

### 任务1：补齐 ResNet18-SDE 配置与唯一源域训练协议

**文件：**
- 创建：`Configs/Model/ResNet18_1D_SDE.yaml`
- 修改：`main_Src_SDE.py`
- 创建：`tests/evmt/test_source_tasks.py`

**接口：**
- 产出：`build_source_tasks(domains: list[int], only_source: int | None) -> list[list[int]]`
- 约定：任务节点 `[s, s]` 只用于构造源域数据与保存路径，不以目标域指标选择检查点。

- [ ] **步骤1：编写失败测试**

```python
from main_Src_SDE import build_source_tasks

def test_builds_four_unique_sources():
    assert build_source_tasks([0, 1, 2, 3], None) == [[0, 0], [1, 1], [2, 2], [3, 3]]

def test_selects_one_source():
    assert build_source_tasks([0, 1, 2, 3], 2) == [[2, 2]]
```

- [ ] **步骤2：确认测试失败**

运行：`pytest -q tests/evmt/test_source_tasks.py`  
预期：因 `build_source_tasks` 尚不存在而失败。

- [ ] **步骤3：实现唯一源域枚举并添加模型配置**

```python
def build_source_tasks(domains, only_source=None):
    selected = domains if only_source is None else [only_source]
    if any(d not in domains for d in selected):
        raise ValueError(f"unknown PU4D source domain: {selected}")
    return [[int(d), int(d)] for d in selected]
```

```yaml
model_name: ResNet18_1D_SDE
bottleneck: true
bottleneck_num: 128
model_type: linear
temp: 1
Dropout: true
input_len: 512
in_chans: 1
drop_rate: 0.0
use_spectral_adapter: true
band_num: 256
adapter_delta: 0.1
```

在 `run()` 中增加 `source_only_mode=true` 分支，使用 `build_source_tasks`；保存目录改为 `TTA_Model/PU4D<lr>/source_<domain>/seed_<seed>/`，检查点选择仍只看 `source_eval`。

- [ ] **步骤4：验证**

运行：`pytest -q tests/evmt/test_source_tasks.py`  
预期：2 passed。

运行：`.conda/bin/python main_Src_SDE.py --cfg job Model=ResNet18_1D_SDE Dataset=PU4D source_only_mode=true only_source=0 process_wandb=false`  
预期：解析后的模型为 `ResNet18_1D_SDE`，任务为 `[0,0]`。

- [ ] **步骤5：提交**

```bash
git add Configs/Model/ResNet18_1D_SDE.yaml main_Src_SDE.py tests/evmt/test_source_tasks.py
git commit -m "feat: add reproducible PU4D source-domain training"
```

---

### 任务2：实现 EMA 可适配状态与三种标签保持视图

**文件：**
- 创建：`evmt/__init__.py`
- 创建：`evmt/ema.py`
- 创建：`evmt/views.py`
- 创建：`tests/evmt/test_ema_views.py`

**接口：**
- 产出：`adaptable_state(model) -> dict[str, Tensor]`
- 产出：`ema_update_(teacher, student, beta: float) -> None`
- 产出：`make_teacher_views(x, style_strength, warp_max, generator) -> list[Tensor]`

- [ ] **步骤1：编写失败测试**

```python
import torch
from evmt.ema import adaptable_state, ema_update_
from evmt.views import make_teacher_views

def test_ema_updates_only_adapter_state(model_pair):
    student, teacher = model_pair
    student[0].band_scale.data.fill_(2)
    frozen_before = teacher[2].fc.weight.detach().clone()
    ema_update_(teacher, student, beta=0.5)
    assert torch.allclose(teacher[0].band_scale, torch.ones_like(teacher[0].band_scale))
    assert torch.equal(teacher[2].fc.weight, frozen_before)

def test_views_keep_shape_and_clean_first():
    x = torch.rand(4, 512)
    views = make_teacher_views(x, 0.03, 0.5, torch.Generator().manual_seed(7))
    assert len(views) == 3
    assert torch.equal(views[0], x)
    assert all(v.shape == x.shape for v in views)
```

- [ ] **步骤2：确认失败**

运行：`pytest -q tests/evmt/test_ema_views.py`  
预期：模块不存在。

- [ ] **步骤3：最小实现**

```python
ADAPTABLE = ("band_scale", "band_bias", "warp_ctrl")

def adaptable_state(model):
    return {n: p for n, p in model.named_parameters() if n.endswith(ADAPTABLE)}

@torch.no_grad()
def ema_update_(teacher, student, beta):
    t, s = adaptable_state(teacher), adaptable_state(student)
    if t.keys() != s.keys():
        raise ValueError("teacher/student adaptable states differ")
    for name in t:
        t[name].mul_(beta).add_(s[name], alpha=1.0 - beta)
```

`make_teacher_views` 复用源训练的频谱风格与平滑 warp 数学形式，但接受显式 `Generator`，返回 `[x, weak_style, weak_warp]`，不修改输入。

- [ ] **步骤4：验证**

运行：`pytest -q tests/evmt/test_ema_views.py`  
预期：2 passed，固定种子结果可重复。

- [ ] **步骤5：提交**

```bash
git add evmt tests/evmt/test_ema_views.py
git commit -m "feat: add EVMT teacher EMA and spectral views"
```

---

### 任务3：实现显著性证据验证

**文件：**
- 创建：`evmt/evidence.py`
- 创建：`tests/evmt/test_evidence.py`

**接口：**
- 产出：`EvidenceResult(mask, destructive_x, plpd, margin_drop)`
- 产出：`verify_evidence(model, x, logits, pseudo, bands, width) -> EvidenceResult`

- [ ] **步骤1：编写失败测试**

```python
def test_mask_is_contiguous_and_destructive_view_is_finite(toy_classifier):
    x = torch.ones(3, 512, requires_grad=True)
    logits = toy_classifier(x)
    result = verify_evidence(toy_classifier, x, logits, logits.argmax(1), bands=2, width=8)
    assert result.mask.shape == x.shape
    assert torch.isfinite(result.destructive_x).all()
    transitions = (result.mask[:, 1:] != result.mask[:, :-1]).sum(1)
    assert torch.all(transitions <= 4)

def test_margin_drop_uses_original_pseudo_class():
    original = torch.tensor([[4.0, 2.0, 1.0]])
    destroyed = torch.tensor([[2.5, 2.0, 1.0]])
    assert torch.allclose(margin_drop(original, destroyed, torch.tensor([0])),
                          torch.tensor([1.5]))
```

- [ ] **步骤2：确认失败**

运行：`pytest -q tests/evmt/test_evidence.py`  
预期：导入失败。

- [ ] **步骤3：实现**

计算 `g = autograd.grad(logits.gather(1,pseudo[:,None]).sum(), x)[0]`，平滑 `abs(x*g)`，以非极大抑制选取 `bands` 个宽度为 `width` 的连续区间；区间内容用两侧邻域中位数替换。再次前向后计算：

```python
def class_margin(logits, pseudo):
    chosen = logits.gather(1, pseudo[:, None]).squeeze(1)
    other = logits.masked_fill(
        torch.nn.functional.one_hot(pseudo, logits.size(1)).bool(), float("-inf")
    ).amax(1)
    return chosen - other

def margin_drop(original, destroyed, pseudo):
    return class_margin(original, pseudo) - class_margin(destroyed, pseudo)
```

- [ ] **步骤4：验证**

运行：`pytest -q tests/evmt/test_evidence.py`  
预期：全部通过，且 `x.grad` 不被累积。

- [ ] **步骤5：提交**

```bash
git add evmt/evidence.py tests/evmt/test_evidence.py
git commit -m "feat: add spectral evidence verification"
```

---

### 任务4：实现可靠性路由与类别均衡记忆库

**文件：**
- 创建：`evmt/reliability.py`
- 创建：`evmt/memory.py`
- 创建：`tests/evmt/test_reliability_memory.py`

**接口：**
- 产出：`ReliabilityRouter.route(q, view_js, margin_drop) -> RouteResult`
- 产出：`ClassBalancedMemory.add(features, q, r, mask, step) -> None`
- 产出：`prototypes() -> tuple[Tensor, Tensor]`
- 产出：`neighbors(features, k) -> tuple[Tensor, Tensor, Tensor]`

- [ ] **步骤1：编写失败测试**

```python
def test_singleton_requires_global_threshold():
    router = ReliabilityRouter(num_classes=3, min_class_support=2,
                               min_conf=0.5, max_js=0.2)
    q = torch.tensor([[.9,.05,.05], [.6,.3,.1], [.1,.1,.8]])
    out = router.route(q, torch.zeros(3), torch.ones(3))
    assert out.certain.dtype == torch.bool
    assert out.certain.shape == (3,)

def test_memory_is_fifo_per_class():
    mem = ClassBalancedMemory(2, capacity_per_class=2, feature_dim=2)
    for step in range(3):
        mem.add(torch.tensor([[float(step), 0.]]), torch.tensor([[.9,.1]]),
                torch.tensor([1.]), torch.tensor([True]), step)
    assert len(mem.queues[0]) == 2
    assert mem.queues[0][0].step == 1
```

- [ ] **步骤2：确认失败**

运行：`pytest -q tests/evmt/test_reliability_memory.py`。

- [ ] **步骤3：实现**

`ReliabilityRouter` 维护 evidence median/MAD 的 EMA 回退；计算 `confidence=1-H(q)/log(C)`、`agreement=exp(-gamma*view_js)` 和 sigmoid evidence，随后应用绝对门槛及类内/全局均值阈值。返回分量必须 detached。

记忆库使用 `list[deque(maxlen=capacity)]`；只写入 `certain=True` 的归一化特征、软标签、可靠性和步数。原型按可靠性加权；近邻返回余弦 top-k、软标签与可靠性，不足时返回实际数量。

- [ ] **步骤4：验证**

运行：`pytest -q tests/evmt/test_reliability_memory.py`  
预期：路由、退化 MAD、空库、FIFO、原型和近邻测试全部通过。

- [ ] **步骤5：提交**

```bash
git add evmt/reliability.py evmt/memory.py tests/evmt/test_reliability_memory.py
git commit -m "feat: add reliability routing and balanced memory"
```

---

### 任务5：实现 EVMT-DtCC 损失

**文件：**
- 创建：`evmt/losses.py`
- 创建：`tests/evmt/test_losses.py`

**接口：**
- 产出：`weighted_sem(logits, reliability, eta, alpha) -> Tensor`
- 产出：`prototype_contrastive(features, pseudo, certain, prototypes, classes, tau) -> Tensor`
- 产出：`neighborhood_kl(logits, features, uncertain, memory, k) -> Tensor`
- 产出：`mean_teacher_kl(student_logits, teacher_q, reliability) -> Tensor`

- [ ] **步骤1：编写失败测试**

```python
@pytest.mark.parametrize("mask", [torch.tensor([False, False]), torch.tensor([True, True])])
def test_losses_are_finite_for_empty_branch(mask):
    logits = torch.randn(2, 3, requires_grad=True)
    loss = weighted_sem(logits, torch.tensor([.2,.8]), eta=.1, alpha=2.)
    loss = loss + prototype_contrastive(torch.randn(2,4), torch.tensor([0,1]),
                                        mask, torch.empty(0,4), torch.empty(0,dtype=torch.long), .1)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
```

- [ ] **步骤2：确认失败**

运行：`pytest -q tests/evmt/test_losses.py`。

- [ ] **步骤3：实现**

所有空分支返回与输入相连的 `logits.sum()*0` 或 `features.sum()*0`，避免设备和反传问题。SEM 使用 `w=eta+(1-eta)*r` 加权 Tsallis 样本项并加 `p_bar*log(p_bar)` 多样性；PCL 用归一化特征/原型交叉熵；NCL 用历史近邻的 `relu(cosine)*reliability` 形成软目标；MT 用逐样本 KL 后按可靠性归一化。

- [ ] **步骤4：验证**

运行：`pytest -q tests/evmt/test_losses.py`  
预期：正常、空子集、空记忆和单样本均通过。

- [ ] **步骤5：提交**

```bash
git add evmt/losses.py tests/evmt/test_losses.py
git commit -m "feat: add reliability-aware EVMT losses"
```

---

### 任务6：实现严格在线运行器与配置

**文件：**
- 创建：`Configs/TTA/evmt.yaml`
- 创建：`evmt/runner.py`
- 创建：`main_tta_evmt.py`
- 创建：`tests/evmt/test_online_runner.py`

**接口：**
- 产出：`EVMTOnlineRunner.step(x, y_for_metrics=None) -> BatchMetrics`
- 产出：`run_task(cfg) -> TaskSummary`

- [ ] **步骤1：编写顺序失败测试**

```python
def test_prediction_is_recorded_before_optimizer_step(runner, batch):
    before = runner.student[0].band_scale.detach().clone()
    metrics = runner.step(batch[0], batch[1])
    assert metrics.prediction_state_hash == tensor_hash(before)
    assert metrics.update_count in (0, 1)
    assert runner.seen_samples == len(batch[0])
```

另测每个样本只出现一次、Teacher 在 Student 后更新、非有限 batch 三项更新全跳过、标签改变不影响参数结果。

- [ ] **步骤2：确认失败**

运行：`pytest -q tests/evmt/test_online_runner.py`。

- [ ] **步骤3：实现配置和运行器**

`Configs/TTA/evmt.yaml` 明确写入：`passes: 1`、`ema_beta: 0.99`、三视图强度、evidence bands/width、门槛、warm-up、每类容量、`eta`、四项 loss 权重、梯度裁剪、adapter/warp 学习率倍率和各消融开关。

`step()` 严格按规格第5节执行。训练参数白名单由 `adaptable_state` 生成；若出现额外可训练参数立即抛错。在线指标使用更新前 logits；`y_for_metrics` 只传给指标累积器，绝不传入路由或损失。

- [ ] **步骤4：验证**

运行：`pytest -q tests/evmt/test_online_runner.py`  
预期：顺序、白名单、有限性、单遍和标签隔离全部通过。

运行：`.conda/bin/python main_tta_evmt.py --cfg job Model=ResNet18_1D_SDE Dataset=PU4D TTA=evmt process_wandb=false only_task='[0,1]'`  
预期：配置显示 `passes=1`，无 prompt 字段参与运行。

- [ ] **步骤5：提交**

```bash
git add Configs/TTA/evmt.yaml evmt/runner.py main_tta_evmt.py tests/evmt/test_online_runner.py
git commit -m "feat: add strict-online EVMT runner"
```

---

### 任务7：实现可复现日志、汇总与中文实验日志

**文件：**
- 创建：`evmt/logging.py`
- 创建：`tools/summarize_evmt.py`
- 创建：`tests/evmt/test_logging_summary.py`
- 创建：`logs/实验日志_EVMT_DtCC.md`

**接口：**
- 产出：`RunLogger.write_batch(metrics) -> None`
- 产出：`RunLogger.finalize(summary) -> Path`
- 产出：`summarize(root: Path) -> pandas.DataFrame`

- [ ] **步骤1：编写失败测试**

构造两个种子的临时 JSONL/summary，断言输出含 `calibration_mean`、`generalization10_mean`、`all12_mean`、`std`、`worst_task`，且运行清单含命令、commit、dirty、环境、数据摘要、种子和 checkpoint SHA256。

- [ ] **步骤2：确认失败**

运行：`pytest -q tests/evmt/test_logging_summary.py`。

- [ ] **步骤3：实现**

JSONL 每行至少包含：step、pre-update correct/pred、各 loss、confidence、JS、PLPD、margin drop、reliability、certain ratio、类别覆盖、memory occupancy、grad norm、adapter/warp norm、skip reason、collapse guard、耗时。CSV 只聚合已完成运行，缺任务时明确报错，不静默计算。

中文日志首条记录设计提交、实现计划提交、数据域、协议和“目标标签仅离线分析”声明。

- [ ] **步骤4：验证并提交**

运行：`pytest -q tests/evmt/test_logging_summary.py`。

```bash
git add evmt/logging.py tools/summarize_evmt.py tests/evmt/test_logging_summary.py logs/实验日志_EVMT_DtCC.md
git commit -m "feat: add reproducible EVMT experiment logging"
```

---

### 任务8：建立严格在线旧基线与全套验证

**文件：**
- 修改：`main_tta_SDE.py`
- 创建：`tests/evmt/test_strict_baseline.py`

**接口：**
- 产出：`strict_online=true` 时旧 SEM/F-Warp 与 Full 使用相同目标顺序和计分时机。

- [ ] **步骤1：编写失败测试**

用两个 batch 的 toy loader，记录 forward/optimizer 事件，断言事件序列为 `predict0, update0, predict1, update1`，且总更新数不超过 batch 数。

- [ ] **步骤2：确认失败**

运行：`pytest -q tests/evmt/test_strict_baseline.py`。

- [ ] **步骤3：最小修改**

为旧入口增加独立 `strict_online` 分支：强制 `passes=1`，删除多阶段 pass 切换，先从 logits 记录指标再反传。保留原多遍历史分支，默认不改变；正式比较命令必须显式 `strict_online=true`。

- [ ] **步骤4：运行完整测试**

运行：`pytest -q tests/evmt`  
预期：全部通过。

运行：`.conda/bin/python -m compileall -q evmt main_tta_evmt.py main_tta_SDE.py main_Src_SDE.py`  
预期：退出码0。

- [ ] **步骤5：提交**

```bash
git add main_tta_SDE.py tests/evmt/test_strict_baseline.py
git commit -m "feat: add comparable strict-online SDE baseline"
```

---

### 任务9：源模型冒烟、重训与两任务校准

**文件：**
- 创建：`scripts/run_pu4d_source.sh`
- 创建：`scripts/run_pu4d_calibration.sh`
- 修改：`logs/实验日志_EVMT_DtCC.md`

**接口：**
- 消费：任务1的4源域训练入口、任务6运行器、任务7日志。
- 产出：种子2025的4个新源检查点及 0→1、0→2 的 Source/strict baseline/MT/MT+Evidence/Full/消融结果。

- [ ] **步骤1：写脚本并做 shell 语法测试**

源脚本依次执行 `only_source=0,1,2,3`，固定 `Model=ResNet18_1D_SDE Dataset=PU4D seed_runs='[2025]' process_wandb=false`。校准脚本固定任务 `[0,1]`、`[0,2]`，按方法名分别写唯一目录。

运行：`bash -n scripts/run_pu4d_source.sh scripts/run_pu4d_calibration.sh`  
预期：退出码0。

- [ ] **步骤2：运行快速源训练冒烟**

运行：`.conda/bin/python main_Src_SDE.py Model=ResNet18_1D_SDE Dataset=PU4D source_only_mode=true only_source=0 seed_runs='[2025]' src_epoch=1 batch_size=16 process_wandb=false`  
预期：完成一个 epoch，生成独立 smoke 目录；不覆盖正式检查点。

- [ ] **步骤3：重训4个正式源模型**

运行：`bash scripts/run_pu4d_source.sh`。每个源域完成后检查 SHA256、最佳源准确率、样本数和有限 loss；任一失败则停止后续 TTA 并记录原因。

- [ ] **步骤4：运行两任务方法矩阵与消融**

运行：`bash scripts/run_pu4d_calibration.sh`。先 Source、strict baseline、MT、MT+Evidence、Full，再运行六项消融。只根据无标签稳定性、有效类别覆盖、certain/记忆覆盖、有限性和运行成本选定参数并写入日志；参数冻结后才揭示准确率和 macro-F1，此后不再改动全任务参数。

- [ ] **步骤5：验证与提交实验记录**

运行：`.conda/bin/python tools/summarize_evmt.py --root outputs/evmt/calibration --expect-tasks 0-1 0-2`。  
预期：生成校准 CSV，所有运行均为单遍且样本数与目标集一致。

```bash
git add scripts/run_pu4d_source.sh scripts/run_pu4d_calibration.sh logs/实验日志_EVMT_DtCC.md
git commit -m "exp: record PU4D source and calibration runs"
```

---

### 任务10：12任务三种子正式实验、汇总和清理清单

**文件：**
- 创建：`scripts/run_pu4d_full.sh`
- 创建：`docs/cleanup/pu4d-cleanup-manifest.md`
- 修改：`logs/实验日志_EVMT_DtCC.md`

**接口：**
- 产出：12任务×3种子的 Source-only、strict baseline、Full 结果及最终汇总。
- 本任务只生成清理清单，不自动删除候选文件。

- [ ] **步骤1：编写并检查全任务脚本**

脚本固定种子 `2025, 2026, 2027`，每个种子先确保4个源检查点存在，再枚举 `permutations([0,1,2,3],2)`，运行三种核心方法；存在完整 summary 时跳过，只有临时文件时重跑该项。

运行：`bash -n scripts/run_pu4d_full.sh`。

- [ ] **步骤2：重训其余种子的源模型并运行核心矩阵**

运行：`bash scripts/run_pu4d_full.sh`。不在运行中修改配置。失败项写入日志并单独重试，不能用成功任务替代。

- [ ] **步骤3：生成最终汇总**

运行：`.conda/bin/python tools/summarize_evmt.py --root outputs/evmt/full --seeds 2025 2026 2027 --expect-all-pu4d`。  
预期：每方法36行任务-种子记录，并输出 calibration2、generalization10、all12、std、worst-task、macro-F1、耗时和显存。

- [ ] **步骤4：生成清理清单**

用 `rg` 检查旧三域 PU、备份、`__pycache__` 和临时输出的引用。清单逐项写路径、大小、最后修改时间、引用证据、可再生方式和建议动作。无法证明无用的项标为保留；不执行删除。

- [ ] **步骤5：最终验证与提交**

运行：`pytest -q tests/evmt`。  
运行：`git diff --check`。  
核对每个正式 summary 的 `seen_samples == target_dataset_size` 且 `passes == 1`。

```bash
git add scripts/run_pu4d_full.sh tools/summarize_evmt.py logs/实验日志_EVMT_DtCC.md docs/cleanup/pu4d-cleanup-manifest.md
git commit -m "exp: complete EVMT-DtCC PU4D evaluation"
```
