# FedALC-LoRA 研究規劃

> 最後更新：2026-04-08

## 方法概述

**FedALC-LoRA**（Adaptive Layer-selective Clustering for LoRA）：
- A 矩陣：global FedAvg（同 FedSA-LoRA）
- B 矩陣：AP clustering → 群內 FedAvg（介於 FedSA-LoRA 的「完全不聚合」和 FedAvg 的「全部聚合」之間）
- Layer selection：自適應挑選高判別力的層作為 clustering feature（Phase 3 引入）

---

## Phase 1：GLUE 上驗證基礎機制

**目標：** 證明「clustering B > local B（FedSA-LoRA）」

### 實作
- AP clustering on full B → 群內 FedAvg B → global FedAvg A
- 不需 warm-up，每輪都做 clustering
- AP 自動決定 cluster 數量（不需指定 K）

### 實驗設定
- 任務：SST-2 / QNLI / MNLI / QQP
- 模型：RoBERTa-large + LoRA r=8
- Clients：30，Dirichlet α=0.5
- Rounds：20 + 50

### 比較方法

| 方法 | A | B |
|---|---|---|
| FedAvg | global avg | global avg |
| FedSA-LoRA | global avg | local（不聚合） |
| **FedALC-LoRA v1** | global avg | **cluster avg (AP)** |

### 報告指標
- Accuracy：mean ± std across clients
- AP 自動決定的 cluster 數量
- Silhouette score（分群品質）
- 收斂曲線（accuracy + server loss vs round）

### 回答的問題
B 矩陣在相似 client 間合作，到底比完全留本地好不好？

### Phase 1 結果（2026-04-08）

**SST-2 + QNLI，30 rounds，30 clients，α=0.5：**

| Task | FedAvg | FedSA | FedALC | Centralized |
|---|---|---|---|---|
| SST-2 | 0.9457 @R13 | 0.9520 @R17 | **0.9547 @R26** | 0.9599 |
| QNLI | 0.9190 @R13 | 0.9243 @R17 | **0.9385 @R29** | 0.9484 |

**Clustering 觀察：**
- SST-2: 5 clusters，QNLI: 4 clusters，R1-R21 cluster membership 完全不變
- Silhouette score 持續上升（0.05 → 0.99），但 R22+ AP 不穩定（cluster 數暴增到 19-21）
- 原因：同群 B 因 FedAvg 趨同 → cosine → 1.0 → similarity matrix degenerate → AP 崩掉
- 需要 fallback 機制（沿用上輪分群 when silhouette 驟降或 cluster 數突變）

**A/B 矩陣驗證：**
- A cosine similarity ≈ 0.95（跨 client 高度相似，確認 A = general knowledge）
- B cosine similarity ≈ 0.02-0.16（跨 client 差異大，確認 B = client-specific）
- B received heatmap 顯示清晰 block-diagonal 結構（clustering 有效）

**已知問題：**
- AP 後期不穩定（需 fallback 機制）
- Cosine annealing LR 跟 total_round 綁定，不同 round 數的實驗 LR schedule 不同（見 memory/issue_lr_schedule.md）

---

## Phase 2：Non-IID 程度分析（GLUE 延伸）

**目標：** 證明 clustering 的收益跟 heterogeneity 程度有關

### 實驗
- α = 0.3（較強 non-IID）/ 0.5（中等）/ 1.0（輕微）
- 注意：α=0.1 在 30 clients + binary task 下 DirichletPartitioner 分割失敗，改用 α=0.3
- 同樣跑 FedAvg / FedSA-LoRA / FedALC-LoRA v1
- 任務：SST-2 + QNLI

### 預期 Finding
- α=1.0（近 IID）：三者差不多
- α=0.5：clustering B 勝出
- α=0.1：可能 local B 反而好（太 heterogeneous，合作是噪音）

### 回答的問題
B 的最佳聚合粒度取決於 heterogeneity 程度，adaptive clustering 能自動找到合適粒度。

---

## Phase 3：Layer Selection（Multi-task FL）

**目標：** 在 multi-task FL 場景下，證明 adaptive layer selection 比 full B 或 per-layer equal-weight（FedLEASE 風格）更好

### Motivation（更新於 2026-04-08）

Phase 1 觀察：單任務 + α=0.5 下 full B clustering 已經足夠好（silhouette 0.99），layer selection 沒有改善空間。原因：所有層反映同一個信號（label ratio），clustering 不需要區分層。

Layer selection 的價值在 **multi-task / cross-domain** 場景：
- 不同任務對不同層敏感，等權混合會讓信號互相抵消
- FedLEASE 用 per-layer averaged cosine（等權），我們改為 adaptive weighting → 更精準
- 通訊效率：如果少量層就能達到同樣分群效果，只需上傳 selected layers 的 B

### 候選 Testbed

| Testbed | 模型 | 任務差異來源 | 優先度 |
|---|---|---|---|
| **Multi-task GLUE** | RoBERTa-large | SST-2 + QNLI + MNLI 混合 client | 高（不換架構） |
| Instruction tuning | LLaMA / Mistral | 不同 instruction 類型 | 中 |
| ViT cross-domain | ViT-base/large | 不同影像 domain | 中 |

### 實作

**Clustering metric 比較（四種）：**

| Metric | 公式 | 來源 |
|---|---|---|
| Full flatten cosine | cos(flatten(B_all), flatten(B_all)) | FedALC Phase 1 |
| Per-layer equal-weight | $\frac{1}{L}\sum_l \cos(\mathbf{B}_l^n, \mathbf{B}_l^m)$ | FedLEASE |
| Per-layer adaptive weight | $\sum_l w_l \cdot \cos(\mathbf{B}_l^n, \mathbf{B}_l^m),\; w_l \propto \text{score}_B(l)$ | **FedALC Phase 3** |
| Per-layer top-K selection | cos(concat(B_{selected}), ...) | **FedALC Phase 3** |

**Discriminability score（三個 metric，見 FedALC-LoRA.md §5）：**
- Metric A：cosine dissimilarity
- Metric B（推薦）：dissimilarity × ||B||_F
- Metric C（備選）：dissimilarity × Fisher

### 報告
- Heatmap：layer × task 的 discriminability score
- 不同任務選到不同層的視覺化證據
- Layer selection vs full B vs per-layer equal-weight 的 silhouette score + accuracy
- Ablation：K = 1, 2, 4, 8, all
- Ablation：Metric A vs B vs C 的分群品質比較

### 回答的問題
哪些層的 B 對 client 分群有判別力？這個 pattern 是否隨任務/domain 變化？

---

## Paper 結構

| Section | 內容 | 資料來源 |
|---|---|---|
| Introduction | B 矩陣聚合粒度問題 + layer 判別力假說 | — |
| Related Work | FedSA-LoRA, FedADC (MADC+AP), FFA-LoRA, clustering FL | notes/papers/ |
| Method | AP clustering B + adaptive layer selection | — |
| Exp 1: GLUE | clustering B vs local B vs global avg | Phase 1 |
| Exp 2: non-IID | α 對 clustering 收益的影響 | Phase 2 |
| Exp 3: cross-domain | layer selection 在異質任務上的效果 | Phase 3 |
| Analysis | per-layer discriminability heatmap + 可解釋性 | Phase 3 |

## Contributions

1. 提出 B 矩陣的 clustering aggregation，介於 FedAvg（全聚合）和 FedSA-LoRA（不聚合）之間
2. 發現 discriminative layers 隨任務/domain 不同，提出 adaptive layer selection
3. Empirical analysis：B 矩陣在哪些層、什麼 heterogeneity 程度下值得合作

---

## 設計決策記錄

| 決策 | 選擇 | 理由 |
|---|---|---|
| Clustering 演算法 | Affinity Propagation | 自動決定 K；FedADC 已驗證可行；N=30 下 O(N²) 無負擔 |
| Similarity metric | Cosine similarity on B（Phase 1）；Layer-selected B（Phase 3） | 先用最簡單版本驗證，再引入 layer selection |
| Client 數量 | 30 | 現有 baseline（FedAvg、FedSA-LoRA）都在 30 clients 上跑，避免重跑 |
| Warm-up | 不使用 | 減少超參數，先驗證無 warm-up 的效果 |
| Delta B vs Raw B | Delta（B_current - B_init） | 標準 LoRA 下 B_init=0 兩者等價，但相容 PiSSA 等非零初始化變種 |
| GLUE 上的 layer selection | 不報告 cross-task 差異 | GLUE 任務間差異太小，layer pattern 可能相似；留給異質 testbed |
