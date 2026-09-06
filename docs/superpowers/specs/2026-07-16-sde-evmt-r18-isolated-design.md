# SDE-EVMT-R18 独立实现设计

## 1. 目标

按照 `docs/superpowers/plans/SDE_EVMT_R18_方案.md` 实现一条独立的 SDE-EVMT-R18 实验路线，并首先在 PU4D 四域数据集的 0→1 任务上完成 Source 训练、严格单遍 TTA 调试和结果校准。

当前最佳 BN-first Full 结构、入口、配置、检查点和 0→1 结果必须保持不变。新方案不得修改其行为，也不得覆盖 `outputs/evmt/bnfirst_calibration` 下的结果。

## 2. 隔离策略

新路线使用独立命名空间：

```text
sde_evmt_r18/
main_src_sde_evmt_r18.py
main_tta_sde_evmt_r18.py
Configs/Model/ResNet18_1D_GN_EVMT.yaml
Configs/TTA/sde_evmt_r18.yaml
```

当前最佳路线继续使用：

```text
evmt/
main_tta_evmt.py
Configs/Model/ResNet18_1D_SDE.yaml
Configs/TTA/evmt.yaml
outputs/evmt/bnfirst_calibration/
```

不创建 Git worktree。原因是当前可运行环境依赖多个尚未被 Git 跟踪的模型、数据配置和入口文件，新 worktree 无法得到完整依赖。隔离通过“只新增独立文件、禁止修改最佳入口和核心包”实现。

## 3. 模型结构

新模型为 `ResNet18-1D-GN-EVMT`：

```text
FFT512
  → log1p + sample-wise median/IQR robust normalization
  → Gated F-Warp
  → Gated Spectral Adapter
  → ResNet18-1D with GroupNorm
  → Global Average Pooling
  → zero-initialized Residual Feature Adapter
  → Bottleneck(512→256)
  → frozen Linear Classifier
```

GroupNorm 主版本固定使用 8 组；当通道数不能被 8 整除时，选择不大于 8 的最大可整除组数。

门控模块初始化接近恒等映射：

- F-Warp：16 个控制点，`max_warp=2.0`，初始 gate=0.05；
- Spectral Adapter：64 个频带，`adapter_delta=0.1`，初始 gate=0.1；
- Feature Adapter：隐藏维度 64，输出层零初始化，初始 gate=0.01。

## 4. Source 路线

Source 阶段按 R0、R1 两个可独立运行的变体实现：

- R0：GN ResNet18，仅 clean 监督；
- R1：R0 + style、warp、noise 三类概率增强 + 对称 KL 预测一致性 + 特征一致性。

R1 默认参数遵循原方案：80 epoch、batch size 64、AdamW、学习率 0.001、weight decay `1e-5`、label smoothing 0.1、mixup alpha 0.2。增强强度和开关全部来自配置，随机性由 seed 2025 控制。

Source 输出使用独立目录：

```text
TTA_Model/SDE_EVMT_R18/PU4D/source_0/seed_2025/<variant>/
```

保存最佳验证/源评估检查点、训练历史、解析配置和 SHA256。不会覆盖现有 `TTA_Model/PU4D0.0001`。

## 5. Target 路线

Target 阶段实现 R2 至 R6 的配置化变体：

- R2：门控 F-Warp + 门控 Spectral Adapter + reliability-weighted SEM；
- R3：R2 + 多视图 EMA Teacher + MT；
- R4：R3 + 连续显著性频带破坏 + margin-PLPD；
- R5：R4 + certain-only class-balanced memory + PCL；
- R6：R5 + uncertain NCL + Residual Feature Adapter。

Teacher 使用 clean、weak style、weak warp、weak noise/gain 四个任务保持视图。证据破坏视图只用于 margin-PLPD，不进入 Teacher 多视图平均。

联合可靠性固定为：

```text
reliability = confidence × view_agreement × normalized_margin_PLPD
```

其中证据分数使用 Batch MAD 稳健归一化。Certain 样本采用预测类内中位数阈值，并同时满足置信度、正 margin drop 和至少 3/4 视图类别一致。

## 6. 在线调度

目标数据严格单遍，每个 Batch 先预测计分，再执行至多一次更新。目标标签只能用于 detached 的 Accuracy、Macro-F1 和离线诊断，不能参与任何适配状态。

前 5 个 Batch 为 warm-up：

- 仅训练 Warp、Spectral Adapter 及其 gates；
- 使用 MT、小权重 SEM 和正则；
- 不写 memory，不使用 PCL/NCL，不启用 Feature Adapter。

当以下无标签条件同时满足时开启 memory 和 Feature Adapter：

- 至少 2 个预测类别存在有效 certain 样本；
- memory 总条目数达到 `2 × 当前 memory 覆盖类别数`；
- 最近可靠性均值未连续下降。

由于 memory 在启动前为空，实现时允许一个独立的 evidence-build 阶段先收集 candidate certain 统计；达到启动条件后才正式写入持久 memory。该定义消除原方案“memory 为空但要求 memory 条目数达标”的循环依赖。

完整卷积权重始终冻结。Layer4 GroupNorm affine 作为配置消融，首轮 R6 默认关闭。

## 7. 损失与优化器

Target 总损失为：

```text
L = 1.0 L_MT
  + 0.2 L_SEM
  + 0.1 L_PCL
  + 0.2 L_NCL
  + 1.0 L_div
  + 2e-4 L_warp
  + 1e-4 L_spectral_adapter
  + 1e-3 L_feature_anchor
```

`L_div` 使用 Teacher EMA 类别先验，不强迫当前 Batch 均匀。NCL 只接受平均近邻相似度不低于 0.3 的 uncertain 样本。

优化器为 AdamW：

- Warp/Spectral Adapter：`5e-4`；
- Feature Adapter：`1e-4`；
- 可选 Layer4 GN affine：`5e-5`；
- weight decay：`1e-5`；
- gradient clip：1.0。

## 8. 安全与失败处理

每个 Batch 检查 logits、loss、梯度和参数是否有限。发生非有限值时跳过优化器、memory 和 Teacher EMA，但保留更新前计分。

同时记录预测有效类别数、最大类别占比、reliability 分布、certain 比例、memory 覆盖、三个 gate、梯度范数和参数位移。首轮如果超过 10% Batch 被跳过，或有效类别数持续低于 8，则停止后续变体并诊断。

## 9. 测试与实验顺序

所有功能遵循 TDD：先写失败测试，再写最小实现。测试分为：

1. robust normalization、GN backbone、三个 gated adapter 的形状、恒等初始化和梯度测试；
2. Source 增强概率、强度、可复现性和损失测试；
3. 四 Teacher 视图、margin-PLPD、类内 certain、EMA prior 测试；
4. memory/PCL/NCL 和在线阶段切换测试；
5. Source/TTA 入口、参数冻结、严格单遍和输出隔离测试。

实验按以下顺序执行：

```text
R0 Source smoke
→ R1 Source smoke
→ R1 Source 0 完整训练
→ R2 0→1
→ R3 0→1
→ R4 0→1
→ R5 0→1
→ R6 0→1
```

只有结构稳定、314 个 Batch 完整、无非有限值且保护跳过不超过 10% 时，才揭示标签指标并与 Source-only、BN-stat、BN-affine、BN-MT-Adapter 和 BN-first Full 比较。

首轮不运行 0→2 或全部 12 个任务，也不根据 0→1 标签结果修改阈值。0→1 完成后再决定下一步。

## 10. 验收标准

- 当前最佳 BN-first 相关被跟踪文件的 Git 内容不变；
- 新模型可以对 `[B,1,512]` 输出 `[B,32]` logits 和 `[B,256]` bottleneck feature；
- Source R0/R1 可训练、保存和恢复检查点；
- Target R2-R6 严格单遍，标签不进入适配决策；
- 所有单元测试和入口冒烟测试通过；
- PU4D 0→1 产生唯一的配置、Batch JSONL、summary 和日志；
- 实验失败不会覆盖当前最佳检查点或输出。
