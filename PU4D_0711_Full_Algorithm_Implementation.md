PU4D 0711-Full 测试时适应算法方案（整理版）

本文档针对 PU4D 数据集描述 0711-Full 方法的完整算法实现。
文档不讨论其他数据集，仅围绕 PU4D 的 Domain 设置、数据处理、模型结构、
Source 训练、Target 严格在线适应流程、可靠性建模、Memory、PCL/NCL
以及消融设计展开。

目标是在无标签目标域中，通过轻量频谱适配、EMA Teacher、多视图一致性、
物理故障证据验证以及可靠性驱动的对比学习，实现稳定的测试时域适应。

**\# PU4D 当前 0711-Full 算法方案与消融实验设计**

本文档对应当前代码实现，主线数据集为 PU4D，模型为
\`ResNet18_1D_SDE\`，用于后续复现、模块审计和消融实验。CWRU、HUSTgearbox
和 WTPG 的实现只作为跨数据集参考，不应把它们的专用 Source
扰动或任务配置混入 PU4D 主实验。

**\## 1. 问题定义**

PU4D 包含 4 个运行工况 Domain：

\| Domain \| 数据文件 \| 转速/负载条件（代码标识） \|

\|---\|---\|---\|

\| D0 \| \`D1_1500_0.7_1000_fft.pt\` \| 1500 rpm, 0.7 \|

\| D1 \| \`D2_900_0.7_1000_fft.pt\` \| 900 rpm, 0.7 \|

\| D2 \| \`D3_1500_0.1_1000_fft.pt\` \| 1500 rpm, 0.1 \|

\| D3 \| \`D4_1500_0.7_400_fft.pt\` \| 1500 rpm, 0.7, 不同采样/工况段 \|

每个 Domain 由同一故障类别空间组成，共 32
类：健康、外圈故障、内外圈组合故障、内圈故障。所有有向迁移任务为
\`4×3=12\` 个：每个任务使用一个 Domain 训练/加载 Source 模型，在另一个
Domain 上进行一次无标签目标流适应。

严格在线指标只统计"当前 batch
更新前"的预测；目标标签只用于离线评估，不得参与伪标签、路由、memory、梯度或早停。

**\## 2. 数据和预处理**

1\. 原始振动信号按长度 1024 的非重叠窗口切分。

2\. 对每个窗口计算 1024 点 FFT。

3\. 取幅值谱并除以窗口长度，只保留前 512 个频率 bin（代码中的
\`input_len=512\`）。

4\. PU4D cache 已保存为 \`\[N,512\]\` 频谱；Dataset 返回 \`\[N,512,1\]\`
或等价的一维输入。

5\. Source/Target 使用同样的 \`mean-std\` 归一化协议；正式恢复实验使用
\`Dataset/PU4D_CACHE\`。

6\. Source seed 固定为 2025；严格在线目标流使用固定
\`stream_seed=2025\`。若做稳定性实验，可额外使用 stream seed
2026，但不得混入主结果。

**\## 3. 模型结构**

模型是：

\`\`\`text

FFT spectrum (512 bins)

-\> ResNet18_1D_SDE backbone

-\> 128-dimensional bottleneck

-\> linear classifier (32 classes)

\`\`\`

Target 端在输入侧增加两个可学习模块：

**\### 3.1 Spectral Adapter**

对每个频率 bin 进行幅值和偏置修正：

\`\`\`text

scale = 1 + adapter_delta \* tanh(band_scale)

bias  = adapter_delta \* band_bias

x_adapt = scale \* x + bias

\`\`\`

当前 PU4D \`adapter_delta=0.1\`，\`band_scale\` 和 \`band_bias\`
从零初始化，因此初始 adapter 是恒等变换。

**\### 3.2 F-Warp / SDE target carrier**

\`warp_ctrl\` 是低维控制点，经 \`tanh\` 和线性插值生成平滑频率位移：

\`\`\`text

delta(f) = max_warp \* interpolate(tanh(warp_ctrl))

x_warp(f) = x(f + delta(f))

\`\`\`

当前 PU4D 默认 \`warp_knots=16\`、\`max_warp=2.0\`。边界采用
replicate/border，重采样使用可微 \`grid_sample\`。

**\### 3.3 严格冻结策略**

正式 PU4D Strict 0711 中：

\- 冻结 ResNet backbone；

\- 冻结 bottleneck；

\- 冻结 classifier；

\- 只更新 \`band_scale\`、\`band_bias\`、\`warp_ctrl\`；

\- Teacher 只保存这些适配参数的 EMA，不复制完整模型。

这一区别必须在消融中明确：若解冻 BN、backbone 或
classifier，应单独标为"放宽协议"，不能与 Strict 0711 主结果并列。

**\## 4. Source 预训练**

Source
阶段的作用是学习故障类别判别表示，并提高对频谱外观变化的鲁棒性。当前完整
Source recipe 来自 \`main_Src_SDE_STABLE.py\` 和
\`main_Src_stronger_SSP_lite_STABLE.py\`。

**\### 4.1 Clean 监督**

对原始 Source batch 计算 label-smoothed cross entropy：

\`\`\`text

L_clean = CE_label_smoothing(f(x), y), epsilon=0.1

\`\`\`

**\### 4.2 SSP-lite 风格扰动**

在频率轴上生成低维平滑控制点，插值得到幅值 mask：

\`\`\`text

x_style(f) = exp(interpolate(ctrl(f))) \* x(f)

\`\`\`

PU4D 默认：\`prob=0.7\`、\`strength=0.15\`、\`knots=8\`。

**\### 4.3 SDE 频率轴扰动**

随机生成平滑位移并重采样：

\`\`\`text

x_warp(f) = x(f + delta(f))

\`\`\`

PU4D 默认：\`prob=0.7\`、\`max_warp=2.0\`、\`knots=16\`。

**\### 4.4 Style+Warp 组合视图**

先执行 SDE warp，再执行 SSP-lite style，得到 \`x_style_warp\`。

**\### 4.5 Source 总损失**

\`\`\`text

L_source = L_clean

-   0.5  \* L_style

-   0.5  \* L_warp

-   0.25 \* L_style_warp

-   0.03 \* L_SymmetricKL

-   0.02 \* L_feature

\`\`\`

其中：

\- \`L_style\`、\`L_warp\`、\`L_style_warp\` 是各视图的监督 CE；

\- \`L_SymmetricKL\` 约束 clean 与增强视图的双向 KL；

\- \`L_feature\` 对 clean/增强 bottleneck 特征做 L2
一致性，特征先归一化，clean 分支 detach；

\- 部分 batch 以 \`mixup_alpha=0.2\`、\`mixup_prob=0.5\` 做监督
Mixup，增强视图继承 Mixup 标签。

Source checkpoint 选择依据是 Source 数据上的 accuracy；正式 PU4D 恢复
checkpoint 位于 \`TTA_Model/PU4D0.001/source\_{id}/seed_2025/\`。Source
训练准确率接近 100% 不是迁移性能，必须另外报告目标 Domain 上的
SourceOnly。

**\## 5. Target 严格在线流程**

对目标流的每个 batch \`B_t\`，严格执行以下顺序：

\`\`\`text

1\. Student 在更新前预测 B_t，记录 Strict Online accuracy。

2\. Teacher 对 clean + 5 个 task-preserving views 推理。

3\. 平均多视图概率，得到软伪标签和候选类别。

4\. 计算 confidence、view agreement、fault evidence。

5\. 联合可靠性 r_i 与 class-balanced certain/uncertain 路由。

6\. 仅用 certain 样本更新 EvidenceMemoryBank。

7\. 计算 SEM + MT + PCL + NCL + adapter/warp regularization。

8\. 只更新 Student 的三个轻量适配载体。

9\. 对适配参数做 EMA，供下一 batch 的 Teacher 使用。

10\. 丢弃当前 batch 的梯度上下文，进入 B\_{t+1}。

\`\`\`

目标流只遍历一次，不能 replay、不能访问未来
batch、不能用目标标签决定更新。

**\## 6. EMA Teacher 与多视图伪标签**

对每个目标样本构造 6 个视图：

1\. clean；

2\. weak smooth style；

3\. weak frequency warp；

4\. weak gain；

5\. weak baseline offset；

6\. weak Gaussian noise。

每个视图由 EMA Teacher 输出：

\`\`\`text

q_i\^v = softmax(logits_teacher(x_i\^v) / T)

q_i   = mean_v(q_i\^v)

pseudo_i = argmax(q_i)

\`\`\`

当前默认 \`teacher_temp=1.0\`、\`ema_beta=0.995\`。EMA 只作用于
\`band_scale\`、\`band_bias\` 和 \`warp_ctrl\`：

\`\`\`text

shadow \<- beta \* shadow + (1-beta) \* student_parameter

\`\`\`

**\## 7. 三个可靠性分量**

**\### 7.1 Teacher confidence**

\`\`\`text

c_i = 1 - H(q_i)/log(C)

\`\`\`

越接近 1，表示 Teacher 的类别分布越尖锐。

**\### 7.2 View agreement**

代码计算各视图与平均分布之间的 JS 型分歧：

\`\`\`text

d_i = mean_v KL(q_i\^v \|\| q_i)

a_i = exp(-gamma \* d_i)

\`\`\`

当前默认 \`view_gamma=5.0\`。

**\### 7.3 Fault evidence**

PU4D 具有明确的轴承物理先验。根据目标转速、轴承几何参数和候选类别，构造
BPFO/BPFI 谐波及边带频带：

\`\`\`text

fr   = rpm / 60

BPFO = 0.5 \* Z \* fr \* (1 - d/D\*cos(theta))

BPFI = 0.5 \* Z \* fr \* (1 + d/D\*cos(theta))

\`\`\`

当前几何参数：\`Z=8\`、滚动体直径 \`6.75 mm\`、节径 \`28.55 mm\`、接触角
\`0°\`。对每个候选类别生成高斯软频带 mask：

\- healthy 预测：无故障频带，evidence 设为中性 1；

\- outer 类：BPFO 谐波和外圈边带；

\- inner 类：BPFI 谐波和内圈边带；

\- combined 类：outer 与 inner mask 的最大值。

默认 \`harmonics=8\`，outer sideband orders \`\[0,1\]\`，inner sideband
orders
\`\[0,1,2\]\`，\`mask_sigma_bins=1.0\`，\`max_mask_ratio=0.18\`，排除
DC。

将 mask 覆盖的频率 bin 替换为局部平均背景，形成 destructive view：

\`\`\`text

x_destroyed = (1-M) \* x + M \* local_background(x)

\`\`\`

计算候选类别概率下降：

\`\`\`text

drop_i = q_i\[pseudo_i\] - q_destroyed_i\[pseudo_i\]

\`\`\`

在 fault-applicable 样本内部使用 median/MAD 稳健归一化和 sigmoid 得到
\`e_i∈\[0,1\]\`。evidence
只说明预测依赖相关频谱结构，不能单独证明伪标签正确。

**\### 7.4 联合可靠性**

\`\`\`text

r_i = clamp(c_i \* a_i \* e_i, 0, 1)

\`\`\`

当前实现中 healthy/不适用物理 evidence 的样本使用中性 evidence，但仍受
confidence 和 view agreement 约束。

**\## 8. Dynamic routing 与 memory**

对每个预测类别
\`k\`，计算该类别样本的可靠性均值，保留不低于类内均值且满足
\`min_reliability\` 的样本：

\`\`\`text

certain = {i: r_i \>= mean(r_j \| pseudo_j=pseudo_i)

and r_i \>= min_reliability}

uncertain = complement(certain)

\`\`\`

PU4D 当前正式配置中 \`min_reliability=0.2\`。

只有 certain 样本进入 class-balanced \`EvidenceMemoryBank\`。每条 memory
记录：

\`\`\`text

(teacher_feature, teacher_soft_probability, reliability, pseudo_label)

\`\`\`

每类容量默认 64；原型是按 reliability 加权的归一化特征均值。当前 batch
必须在损失计算和 Student 更新之后才写入 memory，避免当前样本对自身
PCL/NCL 形成泄漏。

**\## 9. 四类 Target 损失**

**\### 9.1 Reliability-weighted Tsallis SEM**

Student 概率为 \`p_i\`。代码使用 \`alpha=2.0\` 的 Tsallis
entropy，并依据 certain 样本类别频率进行 class-aware reweighting：

\`\`\`text

w_i = eta + (1-eta) \* r_i

L_TE = sum_i w_i \* TsallisAlpha(p_i) / sum_i w_i

L_div = sum_k p_bar_k log(p_bar_k)

L_SEM = L_TE + L_div

\`\`\`

PU4D 调优配置使用 \`eta=0.05\`，即高可靠样本权重影响较大，同时通过
diversity 项抑制类别塌缩。

**\### 9.2 Reliability-weighted Mean Teacher**

\`\`\`text

L_MT = sum_i r_i \* KL(stopgrad(q_i) \|\| p_i) / sum_i r_i

\`\`\`

Teacher soft target 和 reliability 均 detach，不向 Teacher 反传。

**\### 9.3 Certain-aware PCL**

certain 样本与 memory 原型计算 prototype classification loss：

\`\`\`text

L_PCL = CE(cos(normalize(z_i), prototypes) / temperature,

pseudo_i)

\`\`\`

只使用历史 memory 原型，不使用当前 batch 刚写入的特征。默认
\`pcl_temperature=0.2\`。

**\### 9.4 Uncertain-aware NCL**

uncertain 样本在 memory 中检索 top-N 相似邻居，以相似度和邻居
reliability 加权其软概率，形成 \`soft_target_i\`：

\`\`\`text

L_NCL = KL(stopgrad(soft_target_i) \|\| p_i)

\`\`\`

默认邻居数 \`N=3\`、\`ncl_temperature=0.2\`。NCL 不把 uncertain
样本强行压向单一硬类别。

**\### 9.5 目标总损失**

\`\`\`text

L_total = L_SEM

-   lambda_mt  \* L_MT

-   lambda_pcl \* L_PCL

-   lambda_ncl \* L_NCL

-   lambda_adapter \* L_adapter_reg

-   lambda_warp    \* L_warp_reg

\`\`\`

PU4D 最终冻结调优配置（见
\`logs/PU4D_0711_STRICT_TUNING_20260821_054101/report.md\`）为：

\`\`\`yaml

Opt.lr_tar: 0.024

TTA0711.warp_lr_scale: 0.20

TTA0711.ema_beta: 0.995

TTA0711.warmup_batches: 10

TTA0711.aux_ramp_batches: 20

TTA0711.min_reliability: 0.20

TTA0711.lambda_mt: 0.02

TTA0711.lambda_pcl: 0.02

TTA0711.lambda_ncl: 0.01

TTA0711.memory_per_class: 32

TTA0711.pcl_temperature: 0.20

TTA0711.ncl_temperature: 0.20

\`\`\`

这里的 \`adapter_lr_scale\` 保持 1.0；adapter 学习率为
\`Opt.lr_tar\`，warp 学习率为 \`Opt.lr_tar \* warp_lr_scale\`。

**\## 10. 评估输出**

每个任务至少记录：

\- \`Before/SourceOnly\`：目标流开始前的准确率；

\- \`Strict Online\`：每个 batch 更新前预测的准确率加权平均；

\- \`Post-stream\`：目标流结束后，用最终 Student 对整个目标集重新评估；

\- pseudo-label purity；

\- certain ratio / uncertain ratio；

\- class coverage；

\- memory size 和每类容量；

\- PCL/NCL active batch 数；

\- loss finite、运行时间和峰值显存。

主结果应报告 12 个有向任务的逐任务值和算术平均。Post-stream
是诊断指标，不能替代 Strict Online，也不能用于调参选择。

当前 PU4D 恢复基线为 Strict Online 平均 62.7342%；冻结调优配置为
64.16%。

**\## 11. 推荐消融矩阵**

所有消融必须固定：4 个 Source
checkpoint、seed=2025、stream_seed=2025、12 个任务、batch
size=128、同一模型、同一数据 cache。每次只删除或替换一个模块。

**\### A. Source 阶段消融**

\| 编号 \| 配置 \| 保留内容 \| 目的 \|

\|---\|---\|---\|---\|

\| S0 \| Vanilla Source \| 仅 clean CE \| 测量无 Source 鲁棒化基线 \|

\| S1 \| clean + SSP \| 去掉 SDE \| 测量幅值/风格增强贡献 \|

\| S2 \| clean + SDE \| 去掉 SSP \| 测量频率轴增强贡献 \|

\| S3 \| clean + SSP + SDE，无一致性 \| 去掉 SKL/feature \|
测量多视图监督本身 \|

\| S4 \| clean + SSP + SDE + SKL \| 去掉 feature consistency \|
测量预测一致性 \|

\| S5 \| clean + SSP + SDE + feature \| 去掉 SKL \| 测量特征一致性 \|

\| S6 \| Full Source \| 当前完整 Source recipe \| 主方法 Source 端 \|

**\### B. Target 可靠性消融**

\| 编号 \| 配置 \| 修改 \|

\|---\|---\|---\|

\| R0 \| Full \| \`r=c\*a\*e\` \|

\| R1 \| confidence only \| \`r=c\` \|

\| R2 \| confidence + agreement \| \`r=c\*a\` \|

\| R3 \| confidence + evidence \| \`r=c\*e\` \|

\| R4 \| evidence only \| \`r=e\` \|

\| R5 \| fixed threshold \| 去掉类内均值动态阈值，使用固定阈值 \|

\| R6 \| no class balance \| 全局阈值，不按预测类别分组 \|

\| R7 \| no evidence intervention \| 不构造 destructive view，\`e=1\` \|

核心指标除 accuracy 外必须报告 certain pseudo-label purity，验证
reliability 是否真的提高了样本资格质量。

**\### C. Teacher / view 消融**

\| 编号 \| 配置 \|

\|---\|---\|

\| T0 \| Student 直接产生伪标签，无 EMA \|

\| T1 \| EMA clean only \|

\| T2 \| EMA + clean/style/warp \|

\| T3 \| EMA + 全部 6 views（Full） \|

\| T4 \| 去掉 view agreement，仅保留 confidence \|

\| T5 \| 改变 \`ema_beta\`：0.990 / 0.995 / 0.999 \|

**\### D. Memory / contrastive 消融**

\| 编号 \| 配置 \|

\|---\|---\|

\| M0 \| SEM only \|

\| M1 \| SEM + MT \|

\| M2 \| SEM + MT + PCL \|

\| M3 \| SEM + MT + NCL \|

\| M4 \| SEM + MT + PCL + NCL（Full） \|

\| M5 \| memory 不做 reliability weighting \|

\| M6 \| memory 允许 uncertain 写入 \|

\| M7 \| memory 不做 class balance \|

**\### E. 适配参数消融**

\| 编号 \| 可训练参数 \|

\|---\|---\|

\| A0 \| 无更新，SourceOnly \|

\| A1 \| 仅 \`band_scale\`/\`band_bias\` \|

\| A2 \| 仅 \`warp_ctrl\` \|

\| A3 \| Adapter + F-Warp（Full Strict） \|

\| A4 \| A3 + BN affine（放宽协议，单独报告） \|

\| A5 \| A3 + backbone 最后一层（放宽协议，单独报告） \|

**\### F. 物理证据参数敏感性**

只在主方法确认后进行，不与方法模块消融混淆：

\- harmonics：4 / 8 / 12；

\- \`mask_sigma_bins\`：0.5 / 1.0 / 2.0；

\- \`max_mask_ratio\`：0.10 / 0.18 / 0.25；

\- background width：3 / 7 / 11；

\- sideband：无边带 / 一阶边带 / 当前完整边带。

**\## 12. 统计和报告规范**

1\. 主表报告 12 个任务逐任务结果，不能只报告挑选任务。

2\. 同时报告平均值、标准差、最小任务准确率和按 Domain shift 类型的均值。

3\. SourceOnly、Strict Online、Post-stream 三个指标分开，不将
Post-stream 当作在线性能。

4\. 每项消融使用相同 Source checkpoint 和目标流顺序。

5\. 若消融改变 Source 训练，必须重新训练所有 4 个 Source 域，并记录
Source 训练 accuracy、目标域 SourceOnly 和 checkpoint hash。

6\. 若只改变 Target 模块，Source checkpoint 必须完全复用并校验 SHA256。

7\. 禁止用目标标签选择超参数；调参只能使用预先声明的 development
tasks，最终结果必须在冻结配置后评估全部 12 个任务。

8\. 任何改动导致 backbone/classifier 解冻、目标流重复访问、target-label
routing 或动态任务选择，都必须作为"非严格扩展"单独报告。

**\## 13. 当前代码入口与证据文件**

\- PU4D
Source：\`main_Src_SDE_STABLE.py\`、\`main_Src_stronger_SSP_lite_STABLE.py\`

\- PU4D Strict 0711：\`main_tta_0711_strict_online.py\`

\- Strict 随机流扩展：\`main_tta_0711_strict_randomstream.py\`

\- 模型：\`Model_Zoos/ResNet18_1D_SDE.py\`

\- 物理 evidence：\`Lib/physical_fault_evidence.py\`

\- EMA：\`Lib/adaptation_ema.py\`

\- 固定随机流：\`Lib/fixed_random_stream.py\`

\- PU4D 数据：\`Dataset/PU4D.py\`、\`Dataset/PU4D_CACHE/\`

\- 已冻结调参配置：\`Configs/Experiments/PU4D0711_strict_tuning.yaml\`

\-
已完成调优报告：\`logs/PU4D_0711_STRICT_TUNING_20260821_054101/report.md\`

\- 恢复验证报告：\`docs/PU4D_RECOVERY_REPORT_20260821.md\`
