# WTPG 数据集上的 0711 Full Algorithm Implementation

本文档将 `PU4D_0711_Full_Algorithm_Implementation.md` 的算法结构具体化到 WTPG 齿轮箱变工况数据集。所有配置取自当前已验证的最终 profile；文档冻结后不再继续提升或替换配置。WTPG 的物理工况变量是电机转速，因此 Domain 只按转速定义，不能把不同转速随机混成一个域。

## 1. 任务定义与严格边界

给定源域和目标域：

\[
 D_s\rightarrow D_t,\qquad D_s,D_t\in\{0,1,\ldots,7\},\quad D_s\ne D_t.
\]

Source 阶段可以使用源域故障标签；目标流阶段不使用目标标签。全部 `8×7=56` 个有向迁移任务都运行，不能根据 Source Only、目标标签或中间结果筛选任务。

严格在线协议包括：

- source checkpoint 与 target adaptation 分开保存；
- target 数据由 seed=2025 生成固定随机排列，每个样本恰好访问一次；
- 每个 batch 的预测在该 batch 更新前计算并计入 Strict Online；
- 当前 batch 在损失和 optimizer step 后才写入 memory；
- target 标签只用于最终准确率、F1 和混淆矩阵统计；
- BN running statistics、BN affine、backbone 和 classifier 在 0711 target 端全部冻结。

## 2. 数据缓存与 Domain 划分

### 2.1 版本化缓存

| 项目 | 固定值 |
|---|---|
| 数据集 | WTPG / WTPGStrict |
| 原始数据 | `Dataset/WTPG` |
| 严格缓存 | `Dataset/WTPG_STRICT_CACHE_V1` |
| 缓存 manifest | `Dataset/WTPG_STRICT_CACHE_V1/manifest.json` |
| Domain 数 | 8 |
| 每域样本 | 3000 |
| 类别数 | 5 |
| 每类每域 | 600 |
| 张量形状 | `(3000, 1, 512)` |
| source/stream seed | 2025 / 2025 |
| 采样率 | 48 kHz |

manifest 对每个域都记录五类计数 600/类，因此所有 Domain 的样本规模和类别分布严格相等。

### 2.2 Domain 定义

| Domain | 电机转速 |
|---:|---:|
| D0 | 20 Hz |
| D1 | 25 Hz |
| D2 | 30 Hz |
| D3 | 35 Hz |
| D4 | 40 Hz |
| D5 | 45 Hz |
| D6 | 50 Hz |
| D7 | 55 Hz |

域间 shift 是速度变化。任务 `0→7` 表示 20 Hz source 到 55 Hz target；`7→0` 表示反向迁移。不存在随机 Domain 或隐含的任务筛选。

### 2.3 故障类别

| 标签 | 类别 |
|---:|---|
| 0 | Healthy |
| 1 | Broken tooth |
| 2 | Wear gear |
| 3 | Gear root crack |
| 4 | Missing tooth |

## 3. 预处理协议

严格缓存由 `Lib/wtpg_strict_protocol.py` 和缓存构建脚本生成，固定流程为：

1. 读取每条 WTPG MAT 记录的通道 0；
2. 采样率固定为 48,000 Hz；
3. 用长度 2048、stride 2048 的非重叠时域窗口；
4. 计算 real FFT，去掉 DC；
5. 取前 512 个频谱 bin，并做 `log1p` 幅值压缩；
6. 对每个窗口独立进行 z-score：

\[
 x'=(x-\mu_x)/(\sigma_x+\epsilon);
\]

7. 按 speed×fault 分层固定抽取 300 个窗口/记录，合并成每域 3000 个样本；
8. 保存 `x`、`y` 和 speed/recording 元数据及 SHA-256。

由于输入已经是 signed 的 per-window standardized log spectrum，WTPG 不使用对 signed bin 做简单乘法的通用 SSP；该差异是当前最佳 WTPG source 结果的关键实现约束。

## 4. 网络和适配载体

### 4.1 主网络

模型为 `ResNet18_1D_SDE`，输入 `(B,1,512)`，bottleneck=128，线性五类分类头：

\[
 q=g_\theta(f_\theta(x)),\qquad p=h_\theta(q).
\]

Source 和 target 使用相同网络结构，checkpoint contract 中保存 dataset、route、source、seed 和 hash。

### 4.2 Spectral Adapter

WTPG source 模型结构上启用 adapter 接口，但 source 训练开始前通过 `reset_and_freeze_adaptation_carrier` 将 `band_scale=0`、`band_bias=0`，并验证载体是恒等且不参与 source 更新。target 端才解冻：

\[
 A_\phi(x)_k=(1+\delta\tanh s_k)x_k+\delta b_k.
\]

### 4.3 F-Warp

`warp_ctrl` 有 16 个控制点，线性插值到 512 bin，经过 `tanh` 后乘 `max_warp=2.0`。初始化为零：

\[
 x_k^{warp}=x_{k+\Delta_\psi(k)},\qquad
 \Delta_\psi=2.0\,\operatorname{interp}_{512}(\tanh\psi).
\]

source checkpoint 必须通过 carrier identity check；target 端仅 `band_scale`、`band_bias` 和 `warp_ctrl` 有梯度。

## 5. Source-only 训练

### 5.1 路由和固定超参数

- ordinary Source（DtCC 对照）：`main_src_dtcc_wtpg_strict.py`；
- robust Source（0711）：`main_src_0711_wtpg_strict.py` / `Lib/wtpg_source_training.py`；
- checkpoint root：最终 0711 为 `TTA_Model_WTPG_STRICT_V2`，普通对照为 `TTA_Model_WTPG_STRICT_V1`；
- 每个 source speed 单独训练一个模型，目标 speed 不参与训练。

| 参数 | 值 |
|---|---:|
| epochs | 50 |
| batch size | 128 |
| optimizer | AdamW |
| learning rate | 0.001 |
| weight decay | 0.0001 |
| label smoothing | 0.1 |
| num workers | 4 |
| seed | 2025 |

### 5.2 WTPG 专属鲁棒扰动 `wtpg_speed_noise_v1`

因为输入是 log-amplitude 后的 signed z-score，当前 profile 用 additive response/noise 表达传感器响应变化：

\[
 x^{style}=x+r(k)+\epsilon,
\quad r(k)=\operatorname{interp}(u_1,\ldots,u_8),
\quad u_j\sim U(-0.10,0.10),
\quad \epsilon\sim\mathcal N(0,0.02^2),
\]

施加概率为 0.8。

速度变化既会改变响应，也会使 gear-mesh 峰沿频率轴移动，因此 warp 由一个全局 dilation 和一个局部连续 warp 组成：

\[
 s\sim U(0.88,1.12),\qquad
 x^{scale}(k)=x(k/s),\qquad |\Delta_{local}(k)|\le1.0.
\]

全局 scale 概率=0.8，局部 warp 使用 16 控制点、最大 1.0 bin、概率=0.8。`style_warp` 先执行 speed scale+local warp，再叠加 response/noise。四个视图 clean/style/warp/style_warp 共享源标签。

### 5.3 Source 损失

WTPG robust trainer 调用 `Lib/hust_source_training.py::source_loss`。设 `CE_v` 为 label smoothing=0.1 的交叉熵，`SKL` 为 clean 与三增强 logits 的平均 symmetric KL，`FCons` 为 clean 与三增强 bottleneck 的平均特征一致性：

\[
\begin{aligned}
 L_{src}={}&CE_{clean}+0.5CE_{style}+0.5CE_{warp}\\
 &+0.25CE_{style\_warp}+0.03SKL+0.02FCons.
\end{aligned}
\]

ordinary DtCC Source 只使用 `CE_clean`。两条 source 路由独立训练、独立保存，不能把 robust checkpoint 用作 DtCC 对照。

## 6. 0711 严格目标适配

实现：`main_tta_0711_wtpg_strict.py`，共享严格 trainer：`main_tta_0711_strict_randomstream.py`；配置：`Configs/Experiments/WTPG0711_strict.yaml` 的 `final_0711_wtpg` 节。

### 6.1 最终配置

| 参数 | 值 |
|---|---:|
| `Opt.lr_tar` | 0.12 |
| adapter lr scale | 8.0 |
| F-Warp lr scale | 1.0 |
| EMA beta | 0.98 |
| mode | `full` |
| passes | 1 |
| warmup batches | 5 |
| auxiliary ramp batches | 5 |
| minimum reliability | 0.12 |
| memory per class | 64 |
| PCL/NCL temperature | 0.10 / 0.10 |
| PCL/NCL weight | 0.02 / 0.01 |
| MT weight | 0.02 |
| stream | fixed permutation, seed=2025 |
| scoring | pre-update |

实际 AdamW 学习率为 `0.12×8=0.96`（adapter）和 `0.12×1=0.12`（F-Warp），weight decay 使用 strict trainer 默认的 target 值。高 adapter scale 是在开发任务上冻结的最佳配置，不应在复现实验中再次调参。

### 6.2 单遍流与参数控制

target loader 先用 seed=2025 生成固定随机排列，再以 batch=128 顺序读取。每个 batch 执行：

```text
读取固定流 batch
→ EMA teacher 计算伪标签/可靠性
→ student 前向并记录更新前预测（Strict Online）
→ 计算 SEM/MT/PCL/NCL 与正则
→ 只更新 band_scale、band_bias、warp_ctrl
→ 更新 adaptation EMA
→ 将 certain 样本加入 memory
```

student 在 eval 模式运行，BN 统计和 affine 参数完全不变。严格 trainer 使用 `AdaptationEMA`，只保存上述 adaptation 参数的 EMA 状态；主干和分类器不会被 EMA 或 optimizer 修改。

### 6.3 六视图 teacher

WTPG 严格版本的 teacher 对每个 batch 计算六个无标签视图：

1. clean；
2. weak-style（strength=0.05, knots=8）；
3. weak-warp（max_warp=0.5, knots=8）；
4. weak gain（±3%）；
5. weak baseline（±0.02）；
6. weak noise（std=0.01）。

每个视图都使用 adaptation EMA 参数执行前向，概率取平均：

\[
 \bar p=\frac{1}{6}\sum_{v=1}^6p_v,
 \qquad \tilde y=\arg\max_c\bar p_c.
\]

### 6.4 可靠性和 WTPG 证据

样本可靠性为

\[
 r_i=c_i\cdot a_i\cdot e_i,
\]

其中 `c_i` 是 teacher 置信度，`a_i=exp(-5·JS)` 是六视图一致性，`e_i` 是故障证据。

WTPG 当前实现不把目标标签或人工故障频率写入路由。`build_evidence_masks` 对 teacher clean 输入计算 `abs(gradient)×abs(input)` saliency，在频谱上选择每类 top contiguous bands：`saliency_bands=8`、`half_width=3`、最大 mask ratio=0.18。将选中频带替换为局部背景后，比较伪类 margin（配置 `evidence_metric=margin`）的下降：

\[
 e_i=RobustScore\bigl(M(\bar p_i)-M(p_i^{destroy})\bigr).
\]

前 5 个 batch 为 warmup，之后按 evidence interval 执行破坏性证据。物理证据配置对象仍记录 48 kHz、FFT=2048、512 spectrum 等协议元数据，但 WTPG 的实际 mask 由 saliency 实现，不能报告成 PU4D 的 BPFO/BPFI 物理谐波 mask。

`min_reliability=0.12` 后执行 class-balanced certain mask；certain 样本进入 memory，uncertain 样本仍按连续可靠性参与 SEM/MT。

### 6.5 Memory、PCL/NCL 和调度

`EvidenceMemoryBank` 每类最多 64 个历史 teacher feature/probability/reliability。当前 batch 写入发生在 optimizer step 之后。

- `PCL`：certain feature 与历史同类 prototype 对齐，temperature=0.10；
- `NCL`：uncertain feature 与历史近邻关系对齐，neighbors=3，temperature=0.10；
- 前 5 个 warmup batch 不启用 auxiliary loss；之后 5 个 batch 线性 ramp；
- PCL 至少需要 2 个已覆盖类别，NCL 至少需要 4 个类别及 20 个 memory entry（WTPG config 中的 strict minimum）。

### 6.6 Target 损失

\[
\begin{aligned}
 L_{component}={}&L_{SEM}+0.02L_{MT}+0.02L_{PCL}+0.01L_{NCL},\\
 L_{tar}={}&L_{component}+10^{-3}L_A+2\times10^{-4}L_W.
\end{aligned}
\]

`L_A` 约束 adapter 偏离恒等，`L_W` 是 warp 位移 L2 加平滑正则（smooth weight=2.0）。所有损失只由当前/历史无标签目标数据和 teacher 伪标签构成。

## 7. DtCC 对照实现

DtCC 使用 `main_tta_dtcc_wtpg_strict.py` 及 ordinary source checkpoint，保持同一缓存、任务列表、stream seed、batch size 和 pre-update 计分。它的 target 端只更新 BN affine，不能复用 0711 的 adapter/F-Warp checkpoint。DtCC 的 post-stream 塌缩现象必须单独报告，不能用 0711 的 post-stream 逻辑替代。

## 8. 已冻结的全任务结果

结果来源：`docs/WTPG_DTCC_0711_ALL_TASKS_FINAL_SUMMARY_20260826.md`；全量日志：`logs/WTPG_ALL_TASKS_20260826`。

### 8.1 总体

| 方法 | Source Only 平均 | Strict Online | Macro-F1 | Post-stream |
|---|---:|---:|---:|---:|
| DtCC ordinary | 64.43% | 69.96% | 69.41% | 26.55% |
| 0711 WTPG-robust | **68.86%** | **76.05%** | **75.88%** | **77.27%** |
| 0711 − DtCC | +4.44 | +6.10 | +6.47 | +50.72 |

### 8.2 按 Domain 距离

| 距离 | 任务数 | DtCC | 0711 | 0711−DtCC |
|---:|---:|---:|---:|---:|
| 1 | 14 | 88.01 | 95.45 | +7.44 |
| 2 | 12 | 80.62 | 87.72 | +7.11 |
| 3 | 10 | 69.76 | 77.04 | +7.27 |
| 4 | 8 | 59.90 | 66.15 | +6.26 |
| 5 | 6 | 49.53 | 54.02 | +4.48 |
| 6 | 4 | 41.03 | 41.75 | +0.72 |
| 7 | 2 | 40.02 | 39.72 | −0.30 |

### 8.3 逐任务 Strict Online

| Task | DtCC Source Only | 0711 Source Only | DtCC Online | 0711 Online |
|---|---:|---:|---:|---:|
| 0→1 | 97.90 | 98.43 | 97.20 | 98.70 |
| 0→2 | 83.67 | 90.70 | 90.00 | 88.90 |
| 0→3 | 61.03 | 68.83 | 67.87 | 75.77 |
| 0→4 | 62.60 | 57.30 | 63.67 | 62.80 |
| 0→5 | 41.80 | 42.27 | 51.50 | 55.60 |
| 0→6 | 43.07 | 36.13 | 47.27 | 39.80 |
| 0→7 | 43.60 | 35.97 | 44.47 | 48.90 |
| 1→0 | 95.13 | 96.30 | 96.37 | 98.50 |
| 1→2 | 90.47 | 94.93 | 94.30 | 98.27 |
| 1→3 | 73.10 | 81.00 | 82.43 | 92.07 |
| 1→4 | 62.70 | 65.03 | 76.40 | 65.70 |
| 1→5 | 46.90 | 52.40 | 61.67 | 57.30 |
| 1→6 | 43.33 | 45.50 | 54.43 | 51.93 |
| 1→7 | 38.50 | 38.83 | 49.63 | 53.70 |
| 2→0 | 82.67 | 85.23 | 86.10 | 90.57 |
| 2→1 | 91.07 | 95.00 | 90.20 | 95.77 |
| 2→3 | 96.93 | 98.40 | 96.53 | 99.33 |
| 2→4 | 88.97 | 93.63 | 89.03 | 96.80 |
| 2→5 | 71.63 | 74.83 | 83.30 | 88.70 |
| 2→6 | 54.83 | 58.07 | 67.70 | 73.00 |
| 2→7 | 53.23 | 57.20 | 59.30 | 59.23 |
| 3→0 | 59.53 | 60.57 | 66.90 | 64.97 |
| 3→1 | 69.10 | 74.07 | 77.87 | 86.73 |
| 3→2 | 85.20 | 92.40 | 93.77 | 97.63 |
| 3→4 | 76.63 | 92.70 | 93.67 | 98.97 |
| 3→5 | 61.80 | 69.80 | 74.87 | 85.10 |
| 3→6 | 38.03 | 50.40 | 62.07 | 70.10 |
| 3→7 | 43.33 | 53.63 | 60.80 | 70.33 |
| 4→0 | 54.43 | 64.87 | 59.33 | 64.07 |
| 4→1 | 61.90 | 70.37 | 68.17 | 82.87 |
| 4→2 | 79.67 | 84.20 | 87.53 | 95.67 |
| 4→3 | 91.77 | 97.80 | 94.50 | 98.80 |
| 4→5 | 74.83 | 82.27 | 76.10 | 93.57 |
| 4→6 | 65.40 | 67.17 | 71.67 | 75.60 |
| 4→7 | 55.37 | 53.30 | 61.90 | 71.10 |
| 5→0 | 40.47 | 46.53 | 45.87 | 61.30 |
| 5→1 | 47.83 | 53.77 | 47.37 | 61.63 |
| 5→2 | 67.43 | 75.63 | 82.80 | 78.83 |
| 5→3 | 80.30 | 84.03 | 85.83 | 90.83 |
| 5→4 | 82.07 | 89.73 | 87.23 | 94.80 |
| 5→6 | 61.60 | 73.43 | 71.67 | 83.23 |
| 5→7 | 71.20 | 69.00 | 74.43 | 76.53 |
| 6→0 | 35.63 | 37.33 | 35.87 | 37.60 |
| 6→1 | 40.30 | 42.23 | 39.50 | 43.57 |
| 6→2 | 49.20 | 53.07 | 52.90 | 65.33 |
| 6→3 | 56.87 | 71.47 | 63.60 | 85.23 |
| 6→4 | 80.10 | 83.50 | 75.47 | 87.67 |
| 6→5 | 73.77 | 82.60 | 73.27 | 87.97 |
| 6→7 | 71.83 | 87.63 | 86.77 | 94.67 |
| 7→0 | 38.00 | 34.80 | 35.57 | 30.53 |
| 7→1 | 35.17 | 37.67 | 31.37 | 35.90 |
| 7→2 | 48.23 | 50.67 | 46.60 | 52.47 |
| 7→3 | 62.70 | 57.13 | 65.73 | 74.87 |
| 7→4 | 74.90 | 76.13 | 64.63 | 87.07 |
| 7→5 | 79.73 | 79.77 | 72.17 | 86.07 |
| 7→6 | 70.47 | 90.67 | 80.57 | 96.13 |

## 9. 复现入口与日志

- cache manifest：`Dataset/WTPG_STRICT_CACHE_V1/manifest.json`
- Domain/协议校验：`Lib/wtpg_strict_protocol.py`
- DtCC source：`main_src_dtcc_wtpg_strict.py`
- 0711 source：`main_src_0711_wtpg_strict.py`、`Lib/wtpg_source_training.py`
- DtCC target：`main_tta_dtcc_wtpg_strict.py`
- 0711 target：`main_tta_0711_wtpg_strict.py`
- strict trainer：`main_tta_0711_strict_randomstream.py`
- 配置：`Configs/Experiments/WTPG0711_strict.yaml`
- 全量结果：`docs/WTPG_DTCC_0711_ALL_TASKS_FINAL_SUMMARY_20260826.md`
- 全量日志目录：`logs/WTPG_ALL_TASKS_20260826`

复现时应保留 source checkpoint SHA-256、cache content SHA-256、seed、task、trainable parameter names、`pre_update_scoring=True`、`passes=1`、finite-loss 检查和 memory class coverage。

## 10. 方法使用边界

本文档只定义 WTPG 的最终方法和复现配置，不规定后续消融实验的组合。任何扩展实验都必须使用独立的 checkpoint/log 目录，不能覆盖 `TTA_Model_WTPG_STRICT_V2` 或当前全任务结果。

## 11. 结论边界

在当前严格缓存、8 个等量转速 Domain、全部 56 个有向任务和 seed=2025 协议下，WTPG 专属 robust Source + 0711 strict adapter/F-Warp 的总体 Strict Online 和 Post-stream 均超过 DtCC，且避免 DtCC 的最终状态塌缩。不能据此宣称 0711 在每个任务、每个距离或所有随机种子上都优于 DtCC；最大跨度任务仍应单独报告。
