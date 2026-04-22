# Plots 說明

## 共通設定

- Model: RoBERTa-large (355M) + LoRA r=8
- Clients: 30
- Local epochs: 1
- Optimizer: AdamW, cosine annealing, lr=0.001
- 所有比較圖的數據來源為 aiserver1（不跨機器比較）

## 目錄結構

```
plots/
├── r30_c30/          # 目前 paper 圖源（30 rounds, α=0.3 & α=0.5, client-side eval）
├── _archived/        # 過時版本（20 rounds / server-side / warm-up LWC / 跨機器），見 _archived/README.md
└── README.md
```

> 2026-04-20 archive：`r20_c30/` 全數 + `r30_c30/` 共 13 張（server-side 或 warm-up 版 LWC）搬進 `_archived/`。
> 2026-04-21 archive：`r30_c30/` 再 9 張（bar / heatmap / bubble 類）搬進 `_archived/`，paper 圖源收斂到**折線圖 + box plot**。
> Paper 引用只用 `r30_c30/` 剩下的圖。詳見 [`_archived/README.md`](_archived/README.md)。

## 指標說明

- **evaluate/accuracy**: 每個 client 收到聚合後的個人化參數，在 local eval split 上測，取 weighted average
  - FedAvg: 所有 client 用同一個 global model
  - FedSA: global_A + own_B + own_classifier
  - FedALC: global_A + cluster_B + own_others
- **server/loss**: server 用 global model (agg_A + avg_B + avg_others) 在 centralized valid set 上測
- **centralized 虛線**: centralized training 的 best epoch accuracy（upper bound）

---

## r30_c30/ — FedALC 實驗（30 rounds）

### Main comparison（client-side eval, 現行圖源）

**折線圖**（accuracy over rounds）
- `all_methods_unweighted_{sst2,qnli}_a{0.3,0.5}.png` — unweighted mean per round
- `all_methods_weighted_{sst2,qnli}_a{0.3,0.5}.png` — weighted mean per round

**Box plot**
- `all_methods_perclient_{sst2,qnli}_a{0.3,0.5}.png` — per-client distribution at best round

數字見 `notes/experiments/all_methods_comparison.md`。

### FedALC α=0.3 vs α=0.5（折線）

- [SST-2](r30_c30/fedalc_alpha03_vs_05_accuracy_sst2.png) — α=0.3 (0.9694) > α=0.5 (0.9547)
- [QNLI](r30_c30/fedalc_alpha03_vs_05_accuracy_qnli.png) — α=0.3 (0.9588) > α=0.5 (0.9385)

### Clustering 分析（折線）

- [Cluster count α=0.3 vs α=0.5](r30_c30/fedalc_cluster_count_alpha03_vs_05.png) — α=0.3 分 3-4 群，α=0.5 分 4-5 群；兩者後期 AP 不穩定
- [Silhouette α=0.3 vs α=0.5](r30_c30/fedalc_silhouette_score_all.png) — 0.05 → 0.99 持續上升，R22+ 驟降

### A/B 矩陣分析（box）

- [A vs B cosine box plot](r30_c30/ab_cosine_boxplot_sst2.png) — R1-R3。A≈0.95，B≈0.02-0.16

### FedALC-AP vs AP-LWC（折線，無 warm-up）

- `fedalc_vs_lwc_no_warmup_accuracy_{sst2,qnli}.png` — 差距 ≈ 0.25%
- `fedalc_vs_lwc_no_warmup_silhouette_sst2.png` — LWC 的 freeze 在 R4 觸發，避開 AP 後期震盪
- `fedalc_vs_lwc_no_warmup_cluster_count_sst2.png` / `fedalc_vs_lwc_no_warmup_clustering_qnli.png`
