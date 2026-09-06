# UO 数据集上的 0711 Full Algorithm Implementation

本文档是 `PU4D_0711_Full_Algorithm_Implementation.md` 在 UO 数据集上的可复现实例。文档描述的是当前已经验证过的最佳配置，而不是继续搜索中的候选配置。后续 UO 复现实验均以本文档为固定基线。

## 1. 实验目标与固定边界

任务是跨运行工况的无标签目标域测试时适配（test-time adaptation, TTA）。每个任务只有一个带标签的 source domain 和一个无标签的 target domain：

\[
  D_s \rightarrow D_t,\qquad D_s,D_t\in\{D_0,D_1,D_2,D_3\},\quad D_s\ne D_t.
\]

训练阶段使用 source 标签；target 标签只用于离线评估，不能进入伪标签、损失、路由、模型选择或超参数选择。目标流只允许从头到尾访问一次，且每个 batch 的预测必须在该 batch 更新前计算。

固定边界如下：

- 数据缓存、域编号、类别映射和随机种子固定；
- Source-only 和 target adaptation 分别使用独立的 source checkpoint；
- UO 的最终配置冻结为 `UO_ROBUST_SOURCE_STRICT_TTA_20260905_WARP04`；
- target 端冻结 BN 和主干/分类器，只更新频谱适配器和 F-Warp；
- 不再把 Profile A/B 等未超过当前 anchor 的尝试写入最终算法。

## 2. 数据集、域划分和任务

### 2.1 版本化缓存

| 项目 | 固定值 |
|---|---|
| 数据集 | UO |
| 缓存 | `Dataset/UO_EXPERIMENT_20260903/UO_CACHE_CH1_256` |
| 缓存元数据 | `Dataset/UO_EXPERIMENT_20260903/UO_CACHE_CH1_256/meta.json` |
| 文件数 | 4 个域文件 |
| 每域张量 | `(3840, 1, 512)` |
| 每类样本 | 768 |
| 类别数 | 5 |
| 每域样本总数 | 3840 |
| Source seed | 2025 |
| target stream seed | 2025 |

每个缓存文件均包含 `x/data`、`y/label/labels` 和 `meta` 字段。四个域的类别计数完全相同，因此域间比较不会因类别数量不平衡产生额外偏差。

### 2.2 Domain 定义

域按照原始 UO 记录的转速变化轨迹划分，不进行随机混域：

| Domain | 缓存文件 | 工况含义 |
|---|---|---|
| D0 | `D1_A_fft.pt` | A：increasing speed |
| D1 | `D2_B_fft.pt` | B：decreasing speed |
| D2 | `D3_C_fft.pt` | C：increasing then decreasing |
| D3 | `D4_D_fft.pt` | D：decreasing then increasing |

四个域两两组成 12 个有向迁移任务。任务列表是
`0→1, 0→2, 0→3, 1→0, 1→2, 1→3, 2→0, 2→1, 2→3, 3→0, 3→1, 3→2`，不筛选任务。

### 2.3 类别映射

| 标签 | UO 类别 |
|---:|---|
| 0 | H healthy |
| 1 | I inner-race fault |
| 2 | O outer-race fault |
| 3 | B ball fault |
| 4 | C combination fault |

## 3. 信号预处理

缓存生成阶段对原始时域记录执行以下固定流程：

1. 按通道选择 UO 的单个振动通道；
2. 采用长度 1024 的非重叠窗口；
3. 对每个窗口计算长度 512 的实数 FFT；
4. 去除 DC 分量，并保留前 512 个频谱 bin；
5. 使用 `log1p` 压缩幅值动态范围；
6. 将结果保存为 `(N,1,512)` 的 `float32` 张量；
7. 每个原始文件最多取 1012 个完整窗口，再以固定随机规则抽取每类 768 个样本。

因此，模型输入是预计算的单通道对数幅度频谱，不在 target 端重新拟合归一化统计量。训练/评估阶段只读取缓存，避免原始文件解析差异造成实验不可比。

## 4. 模型结构

### 4.1 Backbone

模型为 `ResNet18_1D_SDE`，输入长度 512，5 类输出，中间 bottleneck 维度 128，分类头为线性层。记 backbone、bottleneck 和 classifier 为 `f_θ`、`g_θ`、`h_θ`：

\[
 z=f_\theta(x),\quad q=g_\theta(z),\quad
 p=h_\theta(q),\quad \hat y=\arg\max p.
\]

### 4.2 频谱适配器

源模型中 `use_spectral_adapter=False`，因此 Source checkpoint 只学习任务判别器，适配器载体保持恒等。target 初始化时显式建立频谱适配器：

\[
 A_\phi(x)_k=(1+\delta\tanh s_k)x_k+\delta b_k,
\]

其中 `band_scale=s`、`band_bias=b` 按频谱 bin 建立，初值为零，故初始变换为恒等映射。UO 只在 target 端优化这两个参数。

### 4.3 F-Warp

F-Warp 使用 16 个控制点 `warp_ctrl` 插值到 512 个频谱 bin，经过 `tanh` 后乘以 `max_warp=2.0`，形成连续频率位移场：

\[
 x'_k=x_{k+\Delta_\psi(k)},\qquad
 \Delta_\psi=2.0\cdot \operatorname{interp}_{512}(\tanh(\psi)).
\]

`warp_ctrl` 初值为零，source checkpoint 中保持恒等，target 才允许更新。

### 4.4 严格冻结策略

target 端 `student.eval()`，所有 BN 保持 eval，冻结 running mean/variance 和 affine weight/bias；Dropout 关闭。除 `band_scale`、`band_bias`、`warp_ctrl` 外所有参数 `requires_grad=False`。这使适配容量明确限制在输入频谱坐标变换上。

## 5. Source 训练

### 5.1 当前最佳配置

实现：`main_src_0711_uo_robust_tuned.py`；checkpoint 根目录：`TTA_Model_UO_ROBUST_TUNED`。

| 参数 | 值 |
|---|---:|
| epochs | 15 |
| batch size | 256 |
| optimizer | AdamW |
| learning rate | 8e-4 |
| weight decay | 1e-4 |
| label smoothing | 0.05 |
| mixup alpha / probability | 0.2 / 0.5 |
| optimizer seed | 2025 |
| spectral adapter | disabled / identity |
| checkpoint | `ResNet18_1D_SDE2025fft_Linear.pt` |

每个 source domain 单独训练一个 checkpoint。四个 source checkpoint 的最终 source accuracy 均为 100%；target 域没有参与 source 训练或 checkpoint 选择。

### 5.2 Source 鲁棒频谱扰动

UO 缓存输入为非负对数幅度频谱，当前 profile 采用中等强度、保持类别的频谱扰动：

**SSP-lite（平滑响应扰动）**

生成 8 个控制点的平滑乘性响应，强度 0.15，以 0.8 概率施加：

\[
 x^{style}=x\odot r_\alpha(k),\quad
 \alpha=0.15,\quad P(style)=0.8.
\]

**SDE-lite（局部频率扭曲）**

用 16 个控制点的连续 warp，最大位移 1.75 bin，以 0.8 概率施加：

\[
 x^{warp}_k=x_{k+\Delta(k)},\quad
 |\Delta(k)|\le 1.75.
\]

**联合视图**

先施加 warp，再施加 style，得到 `style_warp`。clean、style、warp、style_warp 四个视图共享 source 标签。Mixup 只在监督 source batch 中使用，不进入 target 伪标签流程。

### 5.3 Source 损失

令 `CE_v` 为带 label smoothing=0.05 的交叉熵，`SKL` 为 clean 与增强视图 logits 的平均 symmetric KL，`FCons` 为 clean/增强 bottleneck 特征的平均归一化平方差，则当前实现的 robust source loss 为：

\[
\begin{aligned}
 L_{src}={}&CE_{clean}+0.35CE_{style}+0.35CE_{warp}\\
 &+0.20CE_{style\_warp}+0.03SKL+0.02FCons.
\end{aligned}
\]

该损失只使用 source 标签。`band_scale`、`band_bias`、`warp_ctrl` 在 source 训练后通过恒等检查，防止 source 阶段泄露适配载体。

## 6. Target 严格在线适配

实现：`main_tta_0711_uo_adapter_only.py` 和
`main_tta_0711_uo_adapter_only_root.py`；共享核心：`main_tta_0711.py`。

### 6.1 优化器与流程参数

| 参数 | 值 |
|---|---:|
| mode | `full` |
| passes | 1 |
| stream order | fixed random permutation |
| stream seed | 2025 |
| batch size | 256 |
| `Opt.lr_tar` | 0.20 |
| adapter lr scale | 2.0 |
| F-Warp lr scale | 0.40 |
| EMA beta | 0.98 |
| warmup passes | 0 |
| BN affine update | false |
| frequency warp | enabled |

AdamW 的两个参数组为：

\[
 \eta_A=0.20\times2.0=0.40,
 \qquad \eta_W=0.20\times0.40=0.08,
\]

weight decay 使用 `1e-4`。每个 batch 的顺序为：固定流取样 → teacher 可靠性评估 → student 前向和在线计分 → 计算无标签损失 → 更新适配器/F-Warp → EMA 更新 → 写入 memory。当前 batch 只在损失和优化器步骤结束后进入 memory，避免 self-prototype/self-neighbor 泄漏。

### 6.2 Teacher 多视图伪标签

UO 使用共享 0711 teacher 管线的三个无标签、类别保持视图：

1. clean：原始频谱；
2. weak-style：strength=0.05、knots=8；
3. weak-warp：max_warp=0.5、knots=8。

三视图概率平均得到

\[
 \bar p=\frac{p(x)+p(x^{style})+p(x^{warp})}{3},
 \qquad \tilde y=\arg\max_c\bar p_c.
\]

注意：UO 当前实现是三视图 generic pipeline，不应在报告中写成 WTPG/PU4D 的六视图或物理轴承谐波证据。

### 6.3 可靠性路由

可靠性由三项相乘：

\[
 r_i=c_i\cdot a_i\cdot e_i.
\]

- `c_i`：平均 teacher 概率的归一化置信度；
- `a_i=exp(-5·JS)`：clean/style/warp 视图一致性；
- `e_i`：破坏性 saliency 证据分数。

UO 没有可可靠映射到物理故障频率的 shaft/BPFO 元数据，因此 `main_tta_0711.py` 使用数据驱动证据：对 clean logits 求 `x·grad` saliency，平滑后选取 4 个 top contiguous bands，每段宽度 17；用局部背景替换后观察伪类概率下降，并通过 robust score 得到 `e_i`。证据参数为 smooth width=9、background width=31、evidence interval=1。

对每个预测类做 class-balanced certain mask。当前 UO 配置 `min_reliability=0`，即不额外设置绝对门槛，但仍执行按类的可靠性均衡选择。可靠样本进入 memory；不可靠样本只由 SEM/MT 以其连续可靠性权重参与。

### 6.4 Memory、PCL 与 NCL

`mode=full` 开启 `EvidenceMemoryBank`，默认每类最多保存 128 个 teacher bottleneck 特征、概率和可靠性。

- PCL：可靠样本与历史同类 prototype 的归一化特征对齐，temperature=0.1；
- NCL：不确定样本与历史邻居的邻域一致性约束，neighbors=5、temperature=0.1；
- memory 只使用过去 batch，更新发生在优化器步骤之后。

### 6.5 Target 损失

令 `L_SEM` 为 reliability-weighted semantic entropy minimization，`L_MT` 为 student logits 与 teacher soft target 的可靠性加权一致性，`L_PCL/L_NCL` 为上节定义的对比/邻域损失。适配器和 warp 正则为 `L_A`、`L_W`：

\[
 L_{tar}=L_{SEM}+0.20L_{MT}+0.05L_{PCL}+0.05L_{NCL}
 +10^{-3}L_A+2\times10^{-4}L_W.
\]

BN anchor 项保留代码默认值 `1e-3`，但因 BN 冻结，其数值恒为零。`L_W` 包含 warp 位移 L2 和平滑项（smooth weight=2.0），限制 F-Warp 发生不连续或过大形变。

### 6.6 严格在线指标

第 `t` 个 batch 的预测为更新前 student 的预测：

\[
 \hat y_t=\arg\max f_{\theta_t}(x_t),\qquad
 \theta_{t+1}=Update(\theta_t,L_t).
\]

`Strict Online Acc` 是所有目标样本的更新前预测平均值；`Post-stream Full-Target Acc` 是流结束后固定 student 在完整 target 集上的一次评估。两者不能混用。

## 7. 当前冻结结果

结果目录：`logs/UO_ROBUST_SOURCE_STRICT_TTA_20260905_WARP04`。12 个有向任务的当前最佳结果为：

| Task | Source Only | Strict Online | Post-stream |
|---|---:|---:|---:|
| 0→1 | 86.93 | 91.90 | 94.01 |
| 0→2 | 98.78 | 98.62 | 99.01 |
| 0→3 | 87.94 | 88.91 | 88.83 |
| 1→0 | 80.44 | 91.17 | 92.92 |
| 1→2 | 90.23 | 95.76 | 96.88 |
| 1→3 | 93.39 | 93.70 | 94.30 |
| 2→0 | 96.69 | 96.46 | 97.14 |
| 2→1 | 83.78 | 91.12 | 93.85 |
| 2→3 | 88.57 | 90.49 | 89.90 |
| 3→0 | 84.35 | 91.90 | 93.44 |
| 3→1 | 97.99 | 98.83 | 99.17 |
| 3→2 | 94.17 | 97.58 | 97.92 |
| 平均 | **90.27** | **93.87** | **94.78** |

Profile A/B 等后续扰动候选分别得到 90.10/93.72/94.61 和 90.35/93.79/94.58，均未同时超过当前 anchor，因此不纳入最终配置。当前文档所称“最佳”即上表对应的 frozen profile。

## 8. 复现入口与证据

- 预处理元数据：`Dataset/UO_EXPERIMENT_20260903/UO_CACHE_CH1_256/meta.json`
- Source trainer：`main_src_0711_uo_robust_tuned.py`
- UO target trainer：`main_tta_0711_uo_adapter_only.py`
- 显式 checkpoint root：`main_tta_0711_uo_adapter_only_root.py`
- 共享 source 增强/损失：`main_Src_SDE_STABLE.py`、`main_Src_stronger_SSP_lite_STABLE.py`
- 共享 target 可靠性、memory、SEM/MT/PCL/NCL：`main_tta_0711.py`
- 固定随机流：`Lib/fixed_random_stream.py`
- Source checkpoint root：`TTA_Model_UO_ROBUST_TUNED`
- 最终日志目录：`logs/UO_ROBUST_SOURCE_STRICT_TTA_20260905_WARP04`

复现实验必须记录：缓存路径及校验值、source/stream seed、source checkpoint 路由、可训练参数名、每 batch 的 pre-update 计分标志、loss 是否有限、memory 类别覆盖和完整 12 任务结果。

## 9. 方法使用边界

本文档只定义 UO 的最终方法和复现配置。任何扩展实验都应保留本文档中的数据划分、目标流、随机种子和严格在线指标定义，并在独立目录中保存，不能覆盖当前最佳 profile。
