# FedALC-LWC 實驗結果與討論

> 日期：2026-04-15
> 設定：SST-2, α=0.5, 30 clients, 20 rounds, layer-selection-k=10, warmup-sil-threshold=0.5

## 核心發現：Warm-up (FedSA) 模式下 silhouette 永遠觸發不了 Phase 1

### 問題描述

FedALC-LWC 設計了三個 phase：
- Phase 0 (warm-up)：跑 FedSA（B 留 local），等 silhouette > 0.5 才進入 Phase 1
- Phase 1：layer-selected clustering
- Phase 2：frozen

**結果：20 輪都停在 Phase 0，silhouette 從 0.05 掉到 0.01，從未超過 0.5。**

### 原因分析

FedALC 的 silhouette 能快速上升（R1: 0.05 → R5: 0.72）是因為 **cluster aggregation 的自我強化效應**：

```
同群 B 做 avg → 同群 client 的 B 趨同 → cosine similarity 升高 → silhouette 升高
```

FedSA 沒有 B 聚合，每個 client 的 B 獨立演化，cosine similarity 維持散亂 → silhouette 永遠低。

**之前分析的 silhouette 數據是從 FedALC 的 checkpoint 算的（已經過 cluster aggregation），不是 FedSA。** 兩者完全不同，不能用 FedALC 的 silhouette 趨勢來預測 FedSA warm-up 的行為。

## 討論用圖表

### 圖 1: Accuracy 比較

[FedALC vs FedALC-LWC Accuracy](../../plots/r30_c30/fedalc_vs_lwc_accuracy_sst2.png)

**觀察：** FedALC-LWC（紅色虛線）跟 FedSA（橘色）完全重疊，因為 20 輪都在跑 FedSA mode。FedALC（綠色）明顯領先。

| 方法 | Best Accuracy | Round |
|---|---|---|
| FedALC | **0.9547** | R26 |
| FedSA-LoRA | 0.9520 | R17 |
| FedALC-LWC | 0.9513 | R13 |
| FedAvg | 0.9457 | R13 |

### 圖 2: Silhouette Score 比較

[FedALC vs FedALC-LWC Silhouette](../../plots/r30_c30/fedalc_vs_lwc_silhouette_sst2.png)

**觀察：** FedALC 的 silhouette 從 0.05 快速爬到 0.99（cluster aggregation 自我強化）。FedALC-LWC 的 trial silhouette 停在 0.01-0.05（FedSA 沒有聚合效應）。灰色虛線是 threshold=0.5，FedALC-LWC 永遠到不了。

### 圖 3: Phase Timeline

[FedALC-LWC Phase Timeline](../../plots/r30_c30/fedalc_lwc_phase_timeline_sst2.png)

**觀察：** 20 輪全部都是 Phase 0（紅色），從未進入 Phase 1 或 Phase 2。

### 圖 4-5: FedALC 基礎結果（對照）

[SST-2 三方比較 α=0.5](../../plots/r30_c30/fedavg_vs_fedsa_vs_fedalc_accuracy_sst2.png)
[SST-2 三方比較 α=0.3](../../plots/r30_c30/fedavg_vs_fedsa_vs_fedalc_accuracy_sst2_alpha03.png)

### 圖 6: Clustering 分析

[Silhouette α=0.3 vs α=0.5](../../plots/r30_c30/fedalc_silhouette_score_all.png)
[Cluster count α=0.3 vs α=0.5](../../plots/r30_c30/fedalc_cluster_count_alpha03_vs_05.png)
[Cluster membership heatmap](../../plots/r30_c30/fedalc_cluster_membership.png)
[Cluster vs label ratio](../../plots/r30_c30/cluster_vs_label_ratio.png)

### 圖 7: A/B 矩陣驗證

[A vs B cosine similarity boxplot](../../plots/r30_c30/ab_cosine_boxplot_sst2.png)
[A vs B cosine heatmap R3](../../plots/r30_c30/ab_cosine_heatmap_r3_sst2.png)

## 下一步選項

### 選項 A：去掉 warm-up，R1 直接用 layer-selected clustering

- 最簡單，R1 就開始 clustering（跟 FedALC 一樣）
- 差別只在 clustering feature 用 top-K 而非 full B
- 但 R1 的 layer selection 品質差（B≈0，Metric B 無意義）

### 選項 B：固定輪數 warm-up（例如 R3）

- R1-R3 跑 FedSA，R4 開始 layer-selected clustering
- 不依賴 silhouette threshold
- 但「R3」是經驗值，不夠科學

### 選項 C：放棄 LWC，專注 FedALC

- Single task 下 FedALC 已經夠好
- Layer-wise 在 single task 改善有限（+0.01-0.07 silhouette）
- 把 LWC 留給 multi-task 場景

## 結論

FedSA warm-up + silhouette threshold 的設計在實際跑起來不可行。Silhouette 的上升依賴 cluster aggregation 的自我強化效應——沒有聚合就沒有信號。需要重新設計 warm-up trigger 機制，或改為不使用 warm-up。
