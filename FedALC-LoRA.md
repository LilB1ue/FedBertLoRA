# FedALC-LoRA: Adaptive Layer-selective Clustering for Federated Low-Rank Adaptation

> 初步概念草案，待討論修訂

## 一句話摘要

在聯邦 LoRA 微調中，不對整個 B 矩陣做高維 clustering，而是**自適應地挑選最具判別力的 layer 作為 clustering feature**，以低維且精準的 client 分群驅動差異化聚合。

---

## 1. Problem Statement

### 現有方法的侷限

| 方法 | 做法 | 問題 |
|---|---|---|
| FedSA-LoRA | A 聚合，B 全留本地 | B 完全不交流，miss 了相似 client 間的合作機會 |
| FedADC | Similarity/dissimilarity clustering on gradients | Gradient 是間接 proxy，且全維度 clustering 受維度災難影響 |
| flowertune-clustering | Flatten 所有 B → KMeans | 高維向量（RoBERTa-large: ~50K dims）分群效果差，噪音多 |

### 核心矛盾

B 矩陣包含豐富的 client-specific 資訊，**理論上是最好的 clustering signal**，但直接使用面臨：
- **維度災難**：全部 flatten 後維度太高，距離度量失效
- **信噪比低**：不是每一層的 B 都有判別力，淺層 B 差異小（general features），混入後稀釋信號
- **計算開銷**：高維 clustering 的 computation + communication cost 不適合 FL 場景

---

## 2. Key Insight

> **不同 layer 的 B 矩陣對 client 資料分佈的敏感度不同。**

- 淺層 (layer 0-5)：B 矩陣跨 client 高度相似 → 學到的是 general linguistic patterns，對 client 分群無用
- 深層 (layer 18-23)：B 矩陣跨 client 差異顯著 → 捕捉 task/domain-specific adaptation，是分群的關鍵信號
- 同一 layer 內不同 module (query vs value) 的判別力也可能不同

**因此，只需要從 B 矩陣中挑選少數幾個最具判別力的 layer，就能在極低維度下實現精準分群。**

---

## 3. Method Overview

### 3.1 架構

```
┌─────────────────────────────────────────────────────┐
│                    Server                            │
│                                                     │
│  ┌─────────────┐    ┌──────────────┐                │
│  │  Layer       │───>│ Layer        │                │
│  │  Score       │    │ Selector     │                │
│  │  Analyzer    │    │ (top-K)      │                │
│  └─────────────┘    └──────┬───────┘                │
│         ▲                   │                        │
│         │                   ▼                        │
│  Per-layer B from    Selected layers                 │
│  all clients         as clustering feature           │
│                             │                        │
│                      ┌──────▼───────┐                │
│                      │  Client      │                │
│                      │  Clustering  │                │
│                      └──────┬───────┘                │
│                             │                        │
│                      ┌──────▼───────┐                │
│                      │  Per-cluster │                │
│                      │ B Aggregation│                │
│                      └──────┬───────┘                │
│                             │                        │
│                      Global A (FedAvg for all)       │
│                      Cluster B (FedAvg within group) │
│                      → Personalized params per client│
└─────────────────────────────────────────────────────┘
```

### 3.2 三階段流程

#### Phase 1: Warm-up（Round 1 ~ T_w）
- 跑標準 FedSA-LoRA（A 聚合，B 留本地）
- 目的：讓 B 矩陣穩定下來，避免早期隨機性干擾 layer selection

#### Phase 2: Layer Selection（Round T_w + 1）
- Server 收集所有 client 的 per-layer B 矩陣
- 對每個 layer 計算 **discriminability score**：

```
score(layer_l) = mean_pairwise_distance(B_l across clients)
              或 variance of B_l across clients
              或 1 - mean_cosine_similarity(B_l across clients)
```

- 選出 top-K 個 score 最高的 layer 作為 clustering feature
- K 可以是固定值或按 threshold 自動決定

#### Phase 3: Adaptive Layer-selective Clustering（Round T_w + 1 ~ T）
- 每輪（或每 N 輪）：
  1. 取各 client 的 **selected layers 的 B 矩陣**，concat 成低維向量
  2. 對低維向量做 clustering（KMeans / SphericalKMeans）
  3. **A 矩陣**：全域 FedAvg（同 FedSA-LoRA）
  4. **B 矩陣**：同一 cluster 內的 client 做 FedAvg，cluster 間獨立
- Layer selection 可週期性 re-evaluate（每 M 輪重新算 score），自適應調整

### 3.3 聚合策略

```
對於每個 client c 在 cluster g:
  A_c ← global_avg(A_all_clients)          # A 全域聚合
  B_c ← cluster_avg(B_clients_in_g)        # B 群內聚合
```

---

## 4. Discriminability Score 設計（待探討）

幾個候選 metric：

### 4.1 Cross-client Cosine Dissimilarity（基本版）
```python
def layer_score(B_layer_all_clients):
    """B_layer_all_clients: list of flattened B for one layer from all clients."""
    sims = pairwise_cosine_similarity(B_layer_all_clients)
    return 1 - sims.mean()  # 越不相似 → 越有判別力
```

### 4.2 Silhouette-based（若有先驗 label）
- 如果知道 client 的大致 data distribution（例如 Dirichlet 的 label proportion），可以用 silhouette score 衡量某個 layer 的 B 是否能區分不同分佈的 client

### 4.3 SVD-based Effective Rank
- 對某 layer 的 cross-client B 矩陣做 SVD
- Effective rank 高 → client 間變異在多個方向上都有 → 更有判別力
- Effective rank 低 → client 間差異集中在少數方向 → 不需要那麼多維度

### 4.4 Round-over-Round Stability
- 計算某 layer 的 score 在連續幾輪間的變化
- 穩定的 layer 更適合做 clustering feature（避免用不穩定的信號分群）

---

## 5. 與現有方法的關係

```
FedAvg (full)
  └── FedSA-LoRA (A 聚合, B 留本地)
        └── FedALC-LoRA (A 聚合, B 按 layer-selective clustering 群內聚合)
              │
              ├── 退化情況 1: K=0 (不選任何 layer) → 等同 FedSA-LoRA
              ├── 退化情況 2: K=all layers, 1 cluster → 等同 FedAvg (full)
              └── 退化情況 3: K=all layers, N clusters → 等同 flowertune-clustering
```

FedALC-LoRA 是 FedSA-LoRA 的自然推廣：FedSA-LoRA 假設所有 B 都應該留本地，FedALC-LoRA 認為**相似 client 的 B 可以合作，但 clustering 信號應該來自最有判別力的 layer，而非全部**。

---

## 6. 預期優勢

1. **降維**：從 ~50K 維降到 ~2K-8K 維（選 2-4 個 layer），分群品質大幅提升
2. **自適應**：不同任務/non-IID 程度下自動挑選不同 layer，不需要人工 tune
3. **B 矩陣合作**：比 FedSA-LoRA 多了相似 client 間的 B 知識共享
4. **通訊開銷可控**：只需額外傳 selected layer 的 B 給 server 做 clustering（或直接用每輪回傳的完整 B）
5. **理論退化性好**：極端情況下退化為已知方法，保底不會比 FedSA-LoRA 差

---

## 7. 潛在風險 & 待解決問題

- **Warm-up 長度 T_w**：太短 → B 還沒穩定，layer selection 被噪音主導；太長 → 浪費前期 round
- **Layer selection 頻率**：固定一次 vs 週期性 re-evaluate？隨訓練進行，判別力分佈可能會變
- **Clustering 數量 K_cluster**：幾群？固定？自適應？（可參考 FedADC 的 affinity propagation）
- **額外通訊成本**：layer selection 需要 server 拿到各 client 的 per-layer B（但 FedSA-LoRA 模式下 B 本來不上傳 → 需要在 selection round 額外上傳）
- **隱私考量**：上傳 B 矩陣是否洩漏更多資訊？（跟 FedAvg 上傳全部參數相比應該不會更差）

---

## 8. 初步實驗計劃

### Baseline
- FedAvg (full A+B aggregation)
- FFA-LoRA (freeze A, aggregate B)
- FedSA-LoRA (aggregate A, B local)
- flowertune-clustering (cluster on full B)

### Ablation
| 實驗 | 目的 |
|---|---|
| K = 1, 2, 4, 8, all | Layer 數量對分群品質的影響 |
| 不同 score metric | Cosine dissim vs variance vs SVD effective rank |
| T_w = 0, 5, 10, 20 | Warm-up 長度的影響 |
| Re-select 頻率 | 固定 vs 每 10 輪 vs 每 20 輪 |
| Dirichlet α = 0.1, 0.5, 1.0 | Non-IID 程度對 layer selection 的影響 |

### 分析
- 視覺化：哪些 layer 被選中？跨任務是否一致？
- Per-layer B similarity heatmap（client × client，per layer）
- Clustering quality: Silhouette score, NMI（若有 ground truth partition）
- 收斂曲線 + 最終 accuracy vs baselines

---

## 8. 設計決策記錄

### 8.1 Delta B vs Raw B

標準 LoRA 初始化 B = zero matrix，因此 `B_current - B_init = B_current`，兩者等價。

**決定：使用 delta（B_current - B_init）作為 clustering feature。**

理由：若未來使用非零初始化的 LoRA 變種（如 PiSSA），delta 才是正確的 learned signal。對標準 LoRA 無損。

受影響的變種：
- PiSSA：SVD 分解初始化，A 和 B 都非零 → 必須用 delta
- rsLoRA：scaling 不同但 B=0 → 無差
- VeRA：共享 frozen random A/B → 不太適用
- LoRA+：init 同標準 LoRA（B=0）→ 無差

### 8.2 Clustering 方法選擇

#### 不指定 K 的方法一覽

| 方法 | 原理 | 優缺點 |
|---|---|---|
| **Affinity Propagation (AP)** | 資料點互相投票選 exemplar | FedADC 驗證過；不需 K，但對 preference 敏感，O(n²) |
| **DBSCAN** | 密度可達性 | 不需 K，但需調 eps + min_samples，高維效果差 |
| **HDBSCAN** | DBSCAN 層次版本 | 幾乎不需調參，但 cluster 數量不穩定 |
| **Mean Shift** | 沿密度梯度爬升找 modes | 不需 K，但 bandwidth 要設，高維慢 |
| **OPTICS** | 類似 DBSCAN 產出 reachability plot | 視覺化好，不需 eps |
| **X-Means** | BIC/AIC 自動 split KMeans cluster | 給 K 上下界，自動搜索 |
| **Silhouette scan + KMeans** | 跑 K=2..K_max，選最高分 | FedLEASE 使用；簡單穩定 |
| **Gap Statistic + KMeans** | 跟 null reference 比較 | 理論基礎好，計算量較大 |
| **Spectral + Eigengap** | Laplacian eigenvalue gap 自動選 K | 理論優雅，但小 N 下不穩定 |
| **Gaussian Mixture + BIC** | 跑多個 K，選 BIC 最低 | 軟分群，給機率 |
| **Agglomerative + 動態切割** | Dendrogram + distance threshold | 可視覺化 hierarchy |
| **Louvain / Leiden** | Graph community detection | 不需 K，需 similarity graph |
| **Self-Tuning Spectral** | 自動學 local scaling + eigengap | 全自動，處理不同密度 |

#### Eigengap 說明

Spectral Clustering 本身需要指定 K，但可透過 **eigengap heuristic** 自動推斷：

計算 graph Laplacian 的特徵值 λ₁ ≤ λ₂ ≤ ... ，找最大的 gap（λ_{k+1} - λ_k），該 k 即為最佳 cluster 數。直覺：K 個明確 cluster → K 個接近 0 的特徵值 → 之後突然跳大。

**問題**：N=40 clients 時只有 40 個特徵值，gap 的統計意義有限，不穩定。

#### 決定：Silhouette scan + KMeans (K=2..K_max)

理由：
1. **N=40 夠小**：AP/DBSCAN/Louvain 在小 N 上行為不穩定，KMeans 最可靠
2. **Layer selection 已降維**：feature 品質好，KMeans 的 cosine/歐氏距離就夠用
3. **跨 round 穩定**：Silhouette 是 deterministic metric
4. **可解釋性好**：方便寫論文報告分群品質
5. **計算量極低**：K_max ≤ 10，跑幾次 KMeans 幾乎不花時間

AP 作為備選可做 ablation 比較。

### 8.3 Client 數量分析

#### 各任務 × Client 數量的 per-client 資料量

| 任務 | 訓練集 | 20 clients | 40 clients | 50 clients | 60 clients |
|---|---|---|---|---|---|
| SST-2 | 67K | ~3,400 | ~1,680 | ~1,350 | ~1,120 |
| QNLI | 105K | ~5,200 | ~2,620 | ~2,100 | ~1,750 |
| MNLI | 393K | ~19,600 | ~9,820 | ~7,850 | ~6,550 |
| QQP | 364K | ~18,200 | ~9,100 | ~7,270 | ~6,070 |
| RTE | 2.5K | ~125 | ~62 ❌ | ~50 ❌ | ~42 ❌ |

#### 決定：預設 40 clients

- Clustering 需要每群有足夠 client（K=4 時每群 ~10 人），20 太少
- 40 clients 時 SST-2 每人 ~1,680 筆，仍然充裕
- 比 FedSA-LoRA 的 3 clients 更接近真實場景
- Ablation 可跑 20 / 60 做對照
- **RTE 排除或降到 10 clients**（資料量不足）

### 8.4 Personalized FL 評估策略

**評估方式**：每個 client 在自己的 local test split（80/20 的 20%）上評估。這是 personalized FL 的標準 protocol（PF2LoRA、FedDPA 皆如此）。

**Label 數少（2-3）的說服力問題**：

二分類任務下 Dirichlet non-IID 的主要變異是 label ratio，reviewer 可能質疑 per-client 提升只是 overfitting 到 local label distribution。

**對策**：
1. 報告 per-client accuracy 的 **mean ± std**，不只 mean
2. 同時報告 **global test set accuracy**，說明 personalization 不犧牲全域表現
3. 加入 **MNLI（3 classes）** 作為 label 較多的 case
4. 分析 **worst-client performance** — personalization 的價值在拉高尾部 client
5. 分析 per-client accuracy **按 label ratio 分組**，區分是 personalization 還是 distribution match

**核心比較對象**：FedALC-LoRA 應與其他 personalized FL 方法做 apple-to-apple 比較，而非僅與 global model 比。「personalized > global」太容易達成，真正有意義的是「FedALC-LoRA > 其他 personalization baselines」。

### 8.5 Personalized FL Baselines

以下為應比較的 personalized federated learning 方法，按相關性排序：

#### 直接可比（LoRA-based Personalized FL）

| 方法 | 論文全稱 | 發表 | Personalization 機制 | 備註 |
|---|---|---|---|---|
| **FedSA-LoRA** | FedSA-LoRA: Selective Aggregation for Low-Rank Adaptation in Federated Learning | ICLR 2025 | B 矩陣留本地（implicit personalization） | 最直接 baseline：FedALC-LoRA 是其推廣 |
| **PF2LoRA** | Personalized Federated Fine-Tuning via Two-Level LoRA | arXiv 2025 | 兩層 LoRA：global {A,B} + local {C_k,D_k} | 明確的 personalization 方法，有 GLUE 實驗 |
| **FedDPA** | Dual-Personalizing Adapter for Federated Foundation Models | NeurIPS 2024 | 兩套完整 LoRA (global+local) + instance-wise 動態加權 | 強 baseline，但原論文用 LLaMA-7B 非 RoBERTa |
| **HiLoRA** | Hierarchical LoRA for Personalized Federated Learning | arXiv 2025 | 三層 hierarchy (global→cluster→client) | 最接近 FedALC-LoRA 的 clustering personalization |

#### 輔助比較（Non-personalized baselines）

| 方法 | 論文全稱 | 發表 | 聚合方式 | 備註 |
|---|---|---|---|---|
| **FedAvg (full)** | — | — | A+B 全部 FedAvg | 最基本 baseline |
| **FFA-LoRA** | FFA-LoRA: Federated Freeze-A LoRA | ICLR 2024 | Freeze A，只聚合 B | FedSA-LoRA 的反向操作 |
| **FedADC** | Federated Fine-Tuning on Heterogeneous Data with Alternating Device-to-Device Collaboration | Computer Networks 2026 | 交替 similarity/dissimilarity clustering + A/B 分離 | Clustering 方法（AP），但非 personalization 導向 |
| **FedLEASE** | Adaptive LoRA Experts Allocation and Selection | NeurIPS 2025 | LoRA-B similarity clustering + silhouette 選 K | Clustering 相關，但跨任務設定 |

#### 比較要點

- FedALC-LoRA vs **FedSA-LoRA**：B 群內共享 vs B 完全本地 → 證明 clustering 帶來合作收益
- FedALC-LoRA vs **PF2LoRA**：layer-selective clustering vs 兩層 LoRA → 不同 personalization paradigm
- FedALC-LoRA vs **FedDPA**：clustering-based vs instance-wise 加權 → 結構性 vs 動態 personalization
- FedALC-LoRA vs **HiLoRA**：layer-selective clustering vs hierarchical clustering → 最直接的方法論競爭
- FedALC-LoRA vs **FedAvg/FFA-LoRA**：personalized vs global → 證明 personalization 有效（必要但不充分）
