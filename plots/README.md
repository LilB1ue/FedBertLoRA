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
├── r20_c30/          # FedAvg vs FedSA, 20 rounds, α=0.5
├── r30_c30/          # FedALC 實驗, 30 rounds, α=0.5 & α=0.3
└── README.md
```

## 指標說明

- **evaluate/accuracy**: 每個 client 收到聚合後的個人化參數，在 local eval split 上測，取 weighted average
  - FedAvg: 所有 client 用同一個 global model
  - FedSA: global_A + own_B + own_classifier
  - FedALC: global_A + cluster_B + own_others
- **server/loss**: server 用 global model (agg_A + avg_B + avg_others) 在 centralized valid set 上測
- **centralized 虛線**: centralized training 的 best epoch accuracy（upper bound）

---

## r20_c30/ — FedAvg vs FedSA（20 rounds, α=0.5）

### Accuracy 曲線

- [SST-2](r20_c30/fedavg_vs_fedsa_accuracy_sst2.png) — FedSA 0.952 > FedAvg 0.946
- [QNLI](r20_c30/fedavg_vs_fedsa_accuracy_qnli.png) — FedSA 0.924 > FedAvg 0.919
- [MNLI](r20_c30/fedavg_vs_fedsa_accuracy_mnli.png) — FedSA 0.765 << FedAvg 0.882, FedSA 未收斂
- [QQP](r20_c30/fedavg_vs_fedsa_accuracy_qqp.png) — FedSA 0.820 << FedAvg 0.892, FedSA 未收斂

### Server Loss 曲線

- [SST-2](r20_c30/fedavg_vs_fedsa_server_loss_sst2.png)
- [QNLI](r20_c30/fedavg_vs_fedsa_server_loss_qnli.png)
- [MNLI](r20_c30/fedavg_vs_fedsa_server_loss_mnli.png)
- [QQP](r20_c30/fedavg_vs_fedsa_server_loss_qqp.png)

---

## r30_c30/ — FedALC 實驗（30 rounds）

### 三方比較：FedAvg vs FedSA vs FedALC（α=0.5）

- [SST-2](r30_c30/fedavg_vs_fedsa_vs_fedalc_accuracy_sst2.png) — FedALC 0.9547 > FedSA 0.9520 > FedAvg 0.9457
- [QNLI](r30_c30/fedavg_vs_fedsa_vs_fedalc_accuracy_qnli.png) — FedALC 0.9385 > FedSA 0.9243 > FedAvg 0.9190

### 三方比較：FedAvg vs FedSA vs FedALC（α=0.3）

- [SST-2](r30_c30/fedavg_vs_fedsa_vs_fedalc_accuracy_sst2_alpha03.png) — FedALC 0.9694 > FedSA 0.9546 > FedAvg 0.9487（+1.5% vs FedSA）
- [QNLI](r30_c30/fedavg_vs_fedsa_vs_fedalc_accuracy_qnli_alpha03.png) — FedALC 0.9588 > FedSA 0.9188 > FedAvg 0.9120（+4.0% vs FedSA）

### FedALC α=0.3 vs α=0.5（含 α=0.5 baselines）

- [SST-2](r30_c30/fedalc_alpha03_vs_05_accuracy_sst2.png) — α=0.3 (0.9694) > α=0.5 (0.9547)
- [QNLI](r30_c30/fedalc_alpha03_vs_05_accuracy_qnli.png) — α=0.3 (0.9588) > α=0.5 (0.9385)

### Clustering 分析

- [Cluster count α=0.3 vs α=0.5](r30_c30/fedalc_cluster_count_alpha03_vs_05.png) — α=0.3 分 3-4 群，α=0.5 分 4-5 群；兩者後期 AP 不穩定
- [Cluster count α=0.5 only](r30_c30/fedalc_cluster_count.png) — SST-2: 5群，QNLI: 4群
- [Silhouette score α=0.5](r30_c30/fedalc_silhouette_score.png) — 0.05 → 0.99 持續上升，R22+ 驟降
- [Cluster membership α=0.5](r30_c30/fedalc_cluster_membership.png) — R1-R21 完全不變，R22+ AP 崩掉

### A/B 矩陣分析（α=0.5, SST-2）

- [A vs B cosine box plot](r30_c30/ab_cosine_boxplot_sst2.png) — R1-R3。A≈0.95（高度相似），B≈0.02-0.16（差異大）
- [A vs B cosine heatmap R3](r30_c30/ab_cosine_heatmap_r3_sst2.png) — B received 顯示清晰 block-diagonal

### 資料分佈與 Cluster-Label 分析

- [α=0.3 資料分佈](r30_c30/data_distribution_alpha03.png) — SST-2 + QNLI，30 clients bubble chart
- [Cluster vs Label Ratio](r30_c30/cluster_vs_label_ratio.png) — R1 的 cluster assignment vs label ratio（α=0.3 & α=0.5）。Cluster 數 > label 數，反映 adaptation pattern 而非單純 label distribution

### 跨機器比較（參考用，不用於正式比較）

- [SST-2 all runs](r30_c30/all_runs_accuracy_sst2.png) — 含 ai2 FedSA 50r（結果異常）
- [QNLI all runs](r30_c30/all_runs_accuracy_qnli.png) — 含 ai2 FedSA 50r（結果異常）
- [SST-2 wandb FedSA 跨機器](r30_c30/wandb_fedavg_vs_fedsa_sst2.png) — aiserver1 vs aiserver2，差 2.6%
- [QNLI wandb FedSA 跨機器](r30_c30/wandb_fedavg_vs_fedsa_qnli.png) — aiserver1 vs aiserver2，差 6.3%
