# FedALC 方法比較定位

> 最後更新：2026-04-15

## 比較方法分類

### Tier 1：必須比較（直接相關）

| 方法 | 為什麼要比 | A 處理 | B 處理 | Clustering |
|---|---|---|---|---|
| **FedAvg** | 最基本 baseline | global avg | global avg | 無 |
| **FedSA-LoRA** | FedALC 的直接上游，A/B 分離的基礎 | global avg | local | 無 |
| **FFA-LoRA** | FedSA 的反向操作 | freeze | global avg | 無 |

**比較重點**：FedALC 的 B clustering 比 FedSA 的 B local 好多少？比 FedAvg 的 B global 好多少？

### Tier 2：強烈建議比較（clustering-based FL）

| 方法 | 為什麼要比 | A 處理 | B 處理 | Clustering |
|---|---|---|---|---|
| **FedLEASE** | 也用 B 做 clustering，NeurIPS 2025 | per-cluster | per-cluster | Agglomerative + silhouette |
| **HiLoRA** | 三層 hierarchy 含 cluster level，CVPR 2026 | 三層 hierarchy | 三層 hierarchy | Spectral + eigengap |

**比較重點**：
- vs FedLEASE：FedALC 用 A global（輕量）vs FedLEASE 用 A per-cluster + MoE（重量），accuracy 差多少？
- vs HiLoRA：FedALC 1× LoRA vs HiLoRA 3× LoRA + orthogonality，參數效率 vs 效果

### Tier 3：可選比較（相關但場景不同）

| 方法 | 為什麼可比 | 注意事項 |
|---|---|---|
| **FedADC** | 也用 AP clustering + A/B 分離 | 交替 freeze A/B，場景是 device-to-device |
| **PF2LoRA** | Personalization 導向，兩層 LoRA | 不同 personalization paradigm |
| **FedDPA** | 兩套 LoRA + instance-wise 加權 | 用 LLaMA-7B 非 RoBERTa |
| **FL-TAC** | Server-side adapter clustering | 每 client 多 adapter，場景不同 |
| **IFCA+LoRA** | Clustered FL 經典方法 + LoRA | Loss-based 選 cluster，非 B similarity |

## 按場景的比較計畫

### Single Task + Dirichlet Non-IID（目前的設定）

已有結果（SST-2, QNLI, α=0.3/0.5, 30 clients）：

| 方法 | 需要跑？ | 狀態 |
|---|---|---|
| FedAvg | ✅ 已有 | α=0.5 (20r), α=0.3 (30r) |
| FedSA-LoRA | ✅ 已有 | α=0.5 (20r), α=0.3 (30r) |
| FFA-LoRA | 需要跑 | 改 `aggregation-mode=ffa` |
| **FedALC** | ✅ 已有 | α=0.5 (30r), α=0.3 (30r) |
| FedLEASE (簡化) | 需要實作 | 見下方 |
| HiLoRA | 困難 | 需要三套 LoRA + orthogonality，改動大 |
| IFCA+LoRA | 可選 | 需要實作 loss-based cluster selection |

**最少需要的 baselines**：FedAvg + FedSA + FFA-LoRA

### Multi-Task FL（計畫中）

設定：SST-2 + QNLI + MNLI + QQP，每 task 分配若干 clients

| 方法 | 需要跑？ |
|---|---|
| FedAvg | 需要跑 |
| FedSA-LoRA | 需要跑 |
| **FedALC** | 需要跑 |
| **FedALC-LWC** | 需要跑（layer selection 在 multi-task 可能有價值） |
| FedLEASE (簡化) | 需要跑（最重要的 multi-task baseline） |
| IFCA+LoRA | 可選 |

## FedLEASE 簡化版 vs 完整版

| | 簡化版（可做） | 完整版（困難） |
|---|---|---|
| Clustering | Agglomerative + silhouette ✅ | 同左 |
| A 聚合 | Per-cluster FedAvg | 同左 |
| B 聚合 | Per-cluster FedAvg | 同左 |
| MoE router | **無**（只用 assigned cluster 的 expert） | 有（2M-1 擴展 router） |
| Client 訓練 | 只訓練 assigned expert | assigned expert + router |
| 改動量 | 新 strategy + server_app | 新 strategy + model 架構 + client_app |

簡化版已能比較核心差異：「A per-cluster vs A global」。MoE 部分貢獻 +1.85%（FedLEASE ablation），可以在 paper 裡 acknowledge。

## 實作優先順序

1. **FFA-LoRA**（已在 fedsa_strategy.py 裡支援，`aggregation-mode=ffa`，直接跑）
2. **FedLEASE 簡化版**（新 strategy，Agglomerative + per-cluster A+B）
3. **Multi-task dataset 支援**（改 dataset.py 支援多 task 混合 clients）
4. **IFCA+LoRA**（可選，loss-based cluster selection）

## 跟每個方法比較時要回答的問題

| 比較對象 | 要回答什麼 |
|---|---|
| vs FedAvg | B clustering 比 B global avg 好多少？ |
| vs FedSA | B clustering 比 B local 好多少？（variance reduction） |
| vs FFA-LoRA | A global + B cluster vs A freeze + B global，哪個 A/B 分工更好？ |
| vs FedLEASE | A global（輕量）vs A per-cluster（重量），accuracy 差多少？MoE 值不值得？ |
| vs HiLoRA | 1× LoRA vs 3× LoRA，參數效率 trade-off |
| vs FedADC | Cosine on B vs MADC on full update，哪個 similarity metric 更好？ |
