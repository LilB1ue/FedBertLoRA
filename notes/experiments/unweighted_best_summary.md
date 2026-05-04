---
tags: [experiments, results, unweighted]
related: [all_methods_comparison.md, ../concepts/eval_weighting_convention.md]
---

# Unweighted best accuracy summary（4 tables × 目前已跑方法）

> 最後更新：2026-04-24
> Metric: **client-side eval_metrics.tsv** 的 unweighted per-client mean accuracy（primary metric per `.claude/rules/evaluation_metric.md`）
> Source: `plot_method_comparison.py` on logs
> Note: **α=0.3 尚未跑 FedALC-AP-LWC**（其他 run 組合完整）

## α=0.5 × SST-2

| Rank | Method | Best unweighted acc | @ Round |
|:---:|---|:---:|:---:|
| 🥇 | FedAvg | 94.27% | R13 |
| 🥈 | FFA | 94.06% | R13 |
| 🥉 | FedSA | 93.72% | R17 |
| 4 | FedALC-AP | 93.33% | R25 |
| 5 | FedALC-AP-LWC | 93.32% | R13 |

## α=0.5 × QNLI

| Rank | Method | Best unweighted acc | @ Round |
|:---:|---|:---:|:---:|
| 🥇 | FedALC-AP-LWC | 92.69% | R9 |
| 🥈 | FedALC-AP | 92.55% | R27 |
| 🥉 | FedAvg | 92.16% | R14 |
| 4 | FFA | 91.71% | R13 |
| 5 | FedSA | 89.94% | R17 |

## α=0.3 × SST-2

| Rank | Method | Best unweighted acc | @ Round |
|:---:|---|:---:|:---:|
| 🥇 | FedALC-AP | 97.10% | R22 |
| 🥈 | FFA | 94.90% | R19 |
| 🥉 | FedAvg | 94.68% | R21 |
| 4 | FedSA | 92.97% | R20 |
| — | FedALC-AP-LWC | *not run* | — |

## α=0.3 × QNLI

| Rank | Method | Best unweighted acc | @ Round |
|:---:|---|:---:|:---:|
| 🥇 | FedALC-AP | 91.03% | R24 |
| 🥈 | FFA | 90.61% | R11 |
| 🥉 | FedAvg | 90.21% | R16 |
| 4 | FedSA | 87.10% | R16 |
| — | FedALC-AP-LWC | *not run* | — |

## Cross-setting observation

| Setting | Winner | Win margin over #2 | Small-client noise 估計 | 是否 robust |
|---|---|:---:|:---:|:---:|
| SST-2 α=0.5 | FedAvg | +0.21pp | ~1.5pp | 🔴 在 noise 內 |
| SST-2 α=0.3 | FedALC-AP | +2.20pp | ~2pp | 🟡 邊界可信 |
| QNLI α=0.5 | FedALC-AP-LWC | +0.14pp | ~1.5pp | 🔴 在 noise 內 |
| QNLI α=0.3 | FedALC-AP | +0.42pp | ~3pp | 🔴 在 noise 內 |

**唯一明顯 signal**：SST-2 α=0.3 下 FedALC-AP 大贏 +2.20pp（超出 noise 範圍）。其餘三個設定的 win 都在 small-client noise 內，需要 multi-seed 才能確認。

## Data 來源

- `logs/20260402_132128_fedavg_a0.5/`、`logs/20260405_071932_fedsa_a0.5/`、`logs/20260420_120302_ffa_a0.5/`、`logs/20260406_203614_fedalc_a0.5/`、`logs/20260415_063849_fedalc-lwc_a0.5/`、`logs/20260422_065151_ffa_a0.5/`（α=0.5）
- `logs/20260412_235402_fedavg_a0.3/`、`logs/20260412_235402_fedsa_a0.3/`、`logs/20260416_064902_ffa_a0.3/`、`logs/20260408_114021_fedalc_a0.3/`（α=0.3）

## Related

- `all_methods_comparison.md` — 完整 weighted + unweighted + per-client distribution
- `../concepts/eval_weighting_convention.md` — 為何用 unweighted 當 primary
