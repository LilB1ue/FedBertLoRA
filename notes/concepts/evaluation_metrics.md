# 評估指標與資料切分

> 最後更新：2026-04-16

## 資料切分結構

```
GLUE dataset:
├── train split
│   └── Dirichlet 分給 30 clients
│       └── 每 client 再 8:2 切成 train / local eval
├── validation split (完整，放 server 端當 global test set)
└── test split (label hidden，不使用)
```

**關鍵**：GLUE 的 test set 是 hidden label（需上傳 prediction server 評分），所以實務上用 validation split 當 test。

## 兩個核心指標

### `evaluate/accuracy`（personalized 指標）

**計算流程：**
```
每輪 R：
  1. Server 下發個人化參數給 30 個 client
     (FedAvg: global_A + global_B; FedALC: global_A + cluster_B + own_others)
  2. 每個 client 在自己的 local eval split（20% of own train）上測 accuracy
  3. Server 收集 30 個 client 的 (num_examples, accuracy)
  4. Weighted average across 30 clients
  
Best round = argmax(weighted avg) over all rounds
```

**意義**：每個 client 的 personalized model 在自己的 data distribution 上表現如何。

**適合比較**：personalized FL 方法的標準指標（PF2LoRA、HiLoRA、FedDPA 都這樣報）。

### `server/accuracy`（global 指標）

**計算流程：**
```
每輪：
  1. Server 用 global model (global_A + avg_B + avg_others) 在 GLUE validation split 上測
  2. 記錄 loss + accuracy
```

**意義**：一個「代表性 global model」在通用 data 上的能力。

## FedALC 的 server_eval 問題

### 問題描述

`server_eval.tsv` 用的 global model 是：
```
global_A = weighted_avg(所有 client 的 A)
global_B = weighted_avg(所有 client 的 B)  ← 跨 cluster 平均
global_others = weighted_avg(所有 client 的 others)
```

**對 FedAvg 來說有意義**：global B 就是實際下發給 client 用的 B。

**對 FedALC 來說沒意義**：
- FedALC 的每個 client 收到的是 **cluster_B**（自己所屬 cluster 的聚合 B），不是 global_B
- `global_B = avg(5 個 cluster 的 B)` 是硬把不同 cluster 的 B 混在一起
- 這個 model 不是任何一個 client 實際用的
- 但也不等於 FedAvg——因為 client 訓練時的 B 起點不同（FedAvg 從 global_B，FedALC 從 cluster_B），最終回傳的 B 也不同

### 跟 FedAvg 的差異

| | FedAvg | FedALC 的 server_eval model |
|---|---|---|
| A | global avg | global avg（相同） |
| B | global avg | global avg（相同） |
| Client 訓練起點 | global_B | cluster_B |
| 訓練動態 | 從 global avg 演化 | 從 cluster avg 演化 |
| 最終 B 分佈 | 低 variance | 每 cluster 內低 variance，cluster 間差異大 |

**結論**：FedALC 的 server_eval 不是 FedAvg，但也不是真實 inference 場景。

## 正確的 FedALC global 能力評估方式

### 方式 A：Per-client personalized on global test

**做法**：每個 client 用自己的 personalized model（`global_A + own_cluster_B + own_others`）在 GLUE validation split 上測，取 30 個 client 的 weighted average。

**優點**：真實反映 FedALC 的 global 能力——每個 client 用自己實際的 model，測通用資料。

**實作**：
- 改 `configure_evaluate` 送 global validation set 給 client
- 或 server 端從 checkpoint 載入每個 client 的 personalized model

### 方式 B：保留現在做法（avg B global model）

**做法**：不改，繼續用 avg B global model 在 validation set 測。

**優點**：實作簡單，跟 FedAvg 做法一致（可直接比較）。

**缺點**：對 FedALC 是誤導的——測的是虛擬 model，不是真實場景。

### 建議

- **主指標用 `evaluate/accuracy`**（personalized，跟 HiLoRA/FedLEASE 的標準一致）
- **global 能力用方式 A**（per-client personalized on global test），作為 supplementary analysis
- 現在的 `server_eval.tsv` 可以當作「虛擬 global model」的參考，但 paper 不報這個

## Best Round 的定義

**FedALC 的 best round**：

```
best_round = argmax_{r ∈ rounds} weighted_avg_{c ∈ clients}(evaluate_accuracy_{r,c})
```

即「30 個 client 的 weighted avg accuracy 最高的那一輪」。這是 personalized 指標的 best。

**注意**：不是「每個 client 各自的 best round」（那樣每個 client 可能在不同輪達到 best）。

## Centralized Baseline 的比較

Centralized training 也用 GLUE validation split 當 eval，跟 FL 的 `server/accuracy` 是同一個資料。**但不能直接跟 `evaluate/accuracy` 比**，因為：
- Centralized: 一個 model 在 centralized validation 上測
- `evaluate/accuracy`: 30 個 personalized model 各在自己的 local eval 上測

兩個測的東西不同。Paper 裡 centralized 通常當 upper bound 參考。
