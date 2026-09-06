# SDE-EVMT-R18 实验日志

日期：2026-07-17
任务：PU4D 0→1，seed=1，严格单遍在线 TTA
代码提交：`0600854` → `a54a2a8` → `9ab3287` → `8cffe96` → `67b3c83` → `7645cb8` → `6909e36`

## 1. 验证与恢复

被中断点位于 Task 3（Source 入口）。恢复后完成 Task 3–7，最终相关测试为 27/27 通过；`compileall` 和 `bash -n scripts/run_pu4d_sde_evmt_r18.sh` 均退出 0。

真实数据 smoke：

```bash
../.conda/bin/python main_src_sde_evmt_r18.py +variant=R1 +source=0 +epochs=1 +smoke_batches=2 process_wandb=false
../.conda/bin/python main_tta_sde_evmt_r18.py +variant=R2 +only_task='[0,1]' +smoke_batches=2
```

Target smoke 产生 2 行 JSONL、1,024 个样本、2 次更新、0 跳过。

## 2. Source smoke 诊断

按计划运行 R0/R1 各 2 epoch × 20 batch 时，完整 Source accuracy 仍约等于 32 类随机水平：

- R0：3.1281% → 3.1188%；
- R1：3.1281% → 3.1362%。

系统化检查结果：

- Domain 0 共 160,384 个有限 FFT 样本；
- 32 类近似均衡，每类 4,994–5,106 个；
- 首个随机 batch 覆盖 27 类；
- classifier 和 stem 梯度均非零；
- 固定 batch 在 50 步达到 100% accuracy；
- 只将覆盖增加至 R0 200 batch，完整 Source accuracy 即达到 60.1120%。

结论：20 batch 只覆盖约 0.8% Source 数据，完整域 accuracy 仍随机是覆盖不足，不是数据、模型或反传故障，因此未修改实现和超参数。

## 3. 完整 R1 Source 0

命令：

```bash
../.conda/bin/python main_src_sde_evmt_r18.py +variant=R1 +source=0 +epochs=80 device=cuda:2 process_wandb=false
```

结果：

- 80/80 epoch 完成，所有 loss 有限；
- 最佳 epoch：48；
- 最佳 Source accuracy：100.0000%；
- 训练耗时：7,079.316 秒；
- 检查点：`TTA_Model/SDE_EVMT_R18/PU4D/source_0/seed_1/R1/best.pt`；
- SHA256：`9e034a7eee7870d11a375138946348435d4669e5f67337702cc8fa00d819dffc`；
- 模型参数量：4,051,251；
- 最佳检查点结构签名：`a440e6b8d6895e5a2803a265818c2f34ad9ddd90a5278f8e1a07fe4e012b55c1`。

冻结 R1 在完整 Target 1 上的 Source-only 基线：

- Accuracy：16.6021%；
- Macro-F1：12.8762%；
- 预测类别覆盖：31/32；
- 最大预测类占比：32.87%。

## 4. R2–R6 严格单遍 0→1

统一命令形式：

```bash
../.conda/bin/python main_tta_sde_evmt_r18.py +variant=R2 +only_task='[0,1]' device=cuda:2
../.conda/bin/python main_tta_sde_evmt_r18.py +variant=R3 +only_task='[0,1]' device=cuda:2
../.conda/bin/python main_tta_sde_evmt_r18.py +variant=R4 +only_task='[0,1]' device=cuda:2
../.conda/bin/python main_tta_sde_evmt_r18.py +variant=R5 +only_task='[0,1]' device=cuda:2
../.conda/bin/python main_tta_sde_evmt_r18.py +variant=R6 +only_task='[0,1]' device=cuda:2
```

所有变体均满足：314 行 JSONL、160,299 样本、1 pass、314 更新、0 跳过、314 个有限 loss batch、最终阶段 FULL、相同 Source 检查点 SHA256。

| 版本 | Accuracy | Δ vs Source | Macro-F1 | Δ vs Source | 时间(s) | memory 条目/覆盖 |
|---|---:|---:|---:|---:|---:|---:|
| Source-only | 16.6021 | — | 12.8762 | — | — | — |
| R2 | 16.5522 | -0.0499 | 12.8298 | -0.0464 | 19.061 | 0 / 0 |
| R3 | 16.6208 | +0.0187 | 12.8887 | +0.0125 | 30.782 | 0 / 0 |
| R4 | 16.6327 | +0.0306 | 12.9039 | +0.0277 | 119.421 | 0 / 0 |
| R5 | 16.6002 | -0.0019 | 12.8729 | -0.0033 | 125.824 | 1,668 / 28 |
| R6 | **16.7468** | **+0.1447** | **12.9103** | **+0.0341** | 128.894 | 1,669 / 28 |

无标签稳定性与分支激活：

| 版本 | 平均 reliability | 平均 certain 比例 | PCL 非零 batch | NCL 非零 batch | 峰值显存(bytes) |
|---|---:|---:|---:|---:|---:|
| R2 | 0.6325 | 0.5023 | 0 | 0 | 870,740,992 |
| R3 | 0.6012 | 0.5006 | 0 | 0 | 875,721,728 |
| R4 | 0.3210 | 0.4798 | 0 | 0 | 1,530,007,552 |
| R5 | 0.3208 | 0.4797 | 308 | 0 | 1,530,007,552 |
| R6 | 0.3216 | 0.4801 | 308 | 308 | 1,532,774,400 |

R4 的平均 margin-drop 为 0.7108，平均 PLPD 为 0.3844，说明 evidence 分支确实参与了可靠性计算。R6 最终 gate：warp 0.05572、spectral 0.70366、feature 0.007741。

## 5. 对照与结论

现有 BN-first Full 的同任务记录为 45.0964% Accuracy / 44.3556% Macro-F1。R6 相比它仍低：

- Accuracy：-28.3495 个百分点；
- Macro-F1：-31.4452 个百分点。

首轮结论：

1. 新路线的 Source、R2–R6、证据、memory、PCL/NCL 和严格单遍输出链路均已跑通；
2. R3/R4 提供很小正收益，R5 的 PCL 抵消该收益，R6 的 NCL + Feature Adapter 恢复并取得本路线最佳；
3. R6 对 Source-only 的提升仅 +0.1447 Accuracy / +0.0341 Macro-F1，不足以替代 BN-first Full；
4. 本轮未依据 Target 标签调整阈值或超参数，也未运行 0→2 或其他任务；
5. 下一步应先分析 GN Source 模型在 0→1 上仅 16.60% 的跨域基线，以及 memory 只覆盖 28/32 类的原因，再决定是否进入跨任务实验。

## 6. 运行产物

- Source：`TTA_Model/SDE_EVMT_R18/PU4D/source_0/seed_1/R1/`
- Target：`outputs/sde_evmt_r18/0_to_1/seed_1/R2/` 至 `R6/`
- 每个 Target 目录包含：`config.yaml`、`batches.jsonl`、`summary.json`、`run.log`。
