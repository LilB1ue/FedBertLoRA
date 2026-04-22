# Archived plots（2026-04-20）

本目錄收錄已被更新版本取代、或因分析方式改變而不再作為 paper 圖源的 plots。
**不要直接在論文 / 報告引用這裡的圖**。需要時可從 log / wandb 重繪。

## `_archived/r20_c30/`（整個 r20_c30 子目錄搬過來）

- 8 張 FedAvg vs FedSA（20 rounds, α=0.5, SST-2 / QNLI / MNLI / QQP）
- **過時原因**：
  - 20 rounds 還沒收斂（尤其 MNLI / QQP 上 FedSA 完全未收斂 → 結論「FedSA 很差」是 artifact）
  - 用 server-side `evaluate/accuracy`（weighted avg of clients with aggregated params）
  - 已被 `r30_c30/all_methods_*_a{0.3,0.5}.png` 取代（30 rounds + client-side eval）

## `_archived/r30_c30/` 內容

| 檔 | 過時原因 |
|---|---|
| `fedavg_vs_fedsa_vs_fedalc_accuracy_{sst2,qnli}{,_alpha03}.png` (×4) | Server-side 三方比較，被 `all_methods_unweighted/weighted_*` 取代 |
| `all_runs_accuracy_{sst2,qnli}.png` (×2) | Server-side；含異常 ai2 50r run（跨機器混比） |
| `fedalc_vs_lwc_accuracy_sst2.png` | Warm-up 版 LWC：20 輪全停 Phase 0，曲線等同 FedSA → 結論誤導 |
| `fedalc_vs_lwc_silhouette_sst2.png` | 同上：silhouette 停在 0.01-0.05 的故事是 warm-up 版的 artifact |
| `fedalc_lwc_phase_timeline_sst2.png` | 同上：畫 20 輪 Phase 0 紅塊，無資訊量 |
| `wandb_fedavg_vs_fedsa_{sst2,qnli}.png` (×2) | Wandb 截圖；有正式 matplotlib 版本 |
| `fedalc_silhouette_score.png` | 單 α 版，被 `fedalc_silhouette_score_all.png` (α=0.3 vs 0.5) 取代 |
| `fedalc_cluster_count.png` | 單 α 版，被 `fedalc_cluster_count_alpha03_vs_05.png` 取代 |

## 2026-04-21 archive 批次（9 張，非 line/box 類）

> 使用者決定 paper 圖源先收斂到**折線圖 + box plot**兩種 — bar / heatmap / bubble 類先全部歸檔，需要時重繪。

| 檔 | 類型 | 原位置 |
|---|---|---|
| `all_methods_best_bar_{sst2,qnli}_a{0.3,0.5}.png` (×4) | bar | `r30_c30/` |
| `ab_cosine_heatmap_r3_sst2.png` | heatmap | `r30_c30/` |
| `cluster_vs_label_ratio.png` | scatter | `r30_c30/` |
| `data_distribution_alpha03.png` | bubble chart | `r30_c30/` |
| `fedalc_cluster_membership.png` | membership matrix | `r30_c30/` |
| `fedalc_lwc_selected_layers_sst2.png` | layer heatmap | `r30_c30/` |

## 留在 `r30_c30/` 的 plots（仍為 paper 圖源）

**折線圖**
- `all_methods_{unweighted,weighted}_{sst2,qnli}_a{0.3,0.5}.png` (×8) — main comparison（client-side eval, accuracy over rounds）
- `fedalc_alpha03_vs_05_accuracy_{sst2,qnli}.png` (×2) — α sensitivity
- `fedalc_cluster_count_alpha03_vs_05.png` / `fedalc_silhouette_score_all.png` — clustering dynamics
- `fedalc_vs_lwc_no_warmup_*` (5 張) — LWC vs FedALC-AP

**Box plot**
- `all_methods_perclient_{sst2,qnli}_a{0.3,0.5}.png` (×4) — per-client distribution at best round
- `ab_cosine_boxplot_sst2.png` — A vs B cosine similarity
