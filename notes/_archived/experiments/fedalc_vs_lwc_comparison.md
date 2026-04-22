# FedALC vs FedALC-LWC 比較實驗

> 目的：在相同條件下比較 clustering feature 的差異（full B vs top-K layer-selected B）

## 實驗設定

| 參數 | FedALC | FedALC-LWC |
|---|---|---|
| Warm-up | 無 | **無**（已移除） |
| Clustering feature | 全部 144 層 B flatten | **Top-K by Metric B** |
| Aggregation | 全部 B per-cluster avg | 全部 B per-cluster avg（相同） |
| A 矩陣 | Global FedAvg | Global FedAvg（相同） |
| Others | Local | Local（相同） |
| Clustering 演算法 | AP | AP（相同） |
| Freeze | 無 | Silhouette > 0.9 或 3 輪穩定 |
| Layer reselect | — | 每輪 |
| Code path | `fedalc_strategy.py` | `fedalc_lwc_strategy.py` |

**唯一差別：AP clustering 用什麼 feature。**

## 跑什麼

| 實驗 | Task | α | Rounds | 方法 |
|---|---|---|---|---|
| 1 | SST-2 | 0.5 | 20 | FedALC-LWC |
| 2 | QNLI | 0.5 | 20 | FedALC-LWC |
| 3 | SST-2 | 0.3 | 20 | FedALC-LWC |
| 4 | QNLI | 0.3 | 20 | FedALC-LWC |

Baseline 已有（同機器、同 seed）：
- FedALC α=0.5: SST-2, QNLI (30 rounds)
- FedALC α=0.3: SST-2, QNLI (30 rounds)
- FedAvg α=0.5: SST-2, QNLI (20 rounds)
- FedSA α=0.5: SST-2, QNLI (20 rounds)

## 比較指標

### 1. Accuracy 收斂曲線
- X 軸: Round, Y 軸: evaluate/accuracy (weighted avg across clients)
- 比較 FedALC vs FedALC-LWC 的收斂速度和最終 accuracy
- **預期**：差異很小（single task 下 top-K vs full B 的 silhouette 只差 0.01-0.07）

### 2. Silhouette Score per Round
- FedALC 用 full B 算的 silhouette vs FedALC-LWC 用 selected layers 算的 silhouette
- **預期**：LWC 的 silhouette 可能略高（去掉了低判別力的層）

### 3. Cluster 數量和穩定性
- 兩者的 AP 分出幾群？是否一致？
- Cluster membership 是否相同？
- LWC 的 freeze 在第幾輪觸發？

### 4. Selected Layers 分析
- 每輪選出的 top-K layers 是哪些？
- 是否穩定（每輪都一樣）還是會變化？
- 跟之前 per-layer discriminability 分析的結果一致嗎（都是 ffn_inter）？

### 5. Cluster vs Label Ratio
- 兩種方法的 cluster assignment 跟 client label ratio 的 correlation 是否不同？

## 需要的圖

| 圖 | 說明 |
|---|---|
| Accuracy: FedALC vs LWC (α=0.5) | 兩條線，SST-2 和 QNLI 各一張 |
| Accuracy: FedALC vs LWC (α=0.3) | 同上 |
| Silhouette: FedALC vs LWC | 兩條線 per task |
| Cluster count: FedALC vs LWC | 觀察 AP 分群是否一致 |
| Selected layers heatmap | X=round, Y=layer index, 顏色=是否被選中 |
| Cluster membership: FedALC vs LWC | 兩者的 heatmap 並排 |

## 預期結果

**樂觀情境**：LWC 的 silhouette 更高 → 分群更好 → accuracy 收斂更快或更高

**中性情境**：兩者差不多（single task 下 full B 已經夠好）

**悲觀情境**：LWC 反而更差（R1 的 layer selection 太 noisy，選錯層導致初始分群不如 full B）

不管哪個結果都有分析價值——如果差不多，說明 layer selection 在 single task 不必要（留給 multi-task）；如果更好，說明去噪有效。
