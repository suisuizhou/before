# PU4D 0711 Domain Split — DtCC Rerun (2026-09-05)

本实验仅运行 DtCC，采用此前 0711 实验使用的 PU4D 四域划分，不采用原论文 PU 划分，也未使用 0711 模型。

## Domain

| Domain | 数据文件 | 工况 |
|---|---|---|
| D0 | `D1_1500_0.7_1000_fft.pt` | D1, 1500 rpm, load 0.7, 1000 Hz |
| D1 | `D2_900_0.7_1000_fft.pt` | D2, 900 rpm, load 0.7, 1000 Hz |
| D2 | `D3_1500_0.1_1000_fft.pt` | D3, 1500 rpm, load 0.1, 1000 Hz |
| D3 | `D4_1500_0.7_400_fft.pt` | D4, 1500 rpm, load 0.7, 400 Hz |

每个域 32 类，样本量约 160k；共完成全部 12 个有向 Domain-shift 任务。

## Results

| Task | Beginning / Source-only (%) | Strict Online (%) | Post-stream (%) |
|---|---:|---:|---:|
| 0→1 | 17.71 | 42.48 | 11.88 |
| 0→2 | 94.60 | 83.37 | 19.58 |
| 0→3 | 57.36 | 65.58 | 17.00 |
| 1→0 | 23.37 | 58.91 | 15.04 |
| 1→2 | 23.76 | 58.72 | 14.41 |
| 1→3 | 21.66 | 45.41 | 11.43 |
| 2→0 | 95.72 | 79.13 | 15.69 |
| 2→1 | 17.43 | 42.29 | 11.72 |
| 2→3 | 60.46 | 63.78 | 15.36 |
| 3→0 | 51.86 | 67.57 | 17.45 |
| 3→1 | 15.82 | 31.28 | 10.10 |
| 3→2 | 55.00 | 70.09 | 16.61 |
| **平均** | **44.56** | **59.05** | **14.69** |

## Configuration

- Model: `ResNet18_1D_SDE`, FFT input, mean-std normalization
- Source checkpoints: `TTA_Model_VANILLA/PU4D/source_{0..3}/seed_2025/`
- DtCC: `lr=0.01`, `weight_decay=0.001`, `optim_steps=2`, `filter_k=50`, `neighbor_k=5`, `alpha=2.0`, `ncl_temperature=1.0`
- Batch size 128, stream seed 2025, one target pass
- BN-only affine adaptation; source model is not retrained in this rerun

Per-task logs are in `logs/PU4D_DTCC_0711DOMAIN_RERUN_20260905/` (`dtcc_<source>to<target>.log`).
