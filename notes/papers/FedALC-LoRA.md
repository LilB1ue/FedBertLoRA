# FedALC-LoRA: Adaptive Layer-selective Clustering for Federated Low-Rank Adaptation

> 最後更新：2026-04-06

## 一句話摘要

在聯邦 LoRA 微調中，用 AP clustering 讓相似 client 的 B 矩陣群內聚合（而非完全留本地），並透過**自適應 layer selection** 挑選最具判別力的層作為 clustering feature，提升分群品質。

---

## 1. Problem Statement

### 現有方法的侷限

| 方法 | 做法 | 問題 |
|---|---|---|
| FedSA-LoRA | A 聚合，B 全留本地 | B 完全不交流，miss 了相似 client 間的合作機會 |
| FedADC | MADC + AP clustering，交替 sim/dissim | MADC 解決 structural awareness，但未從根本解決高維 cosine concentration |
| flowertune-clustering | Flatten 所有 B → KMeans | KMeans 在高維座標空間直接受 concentration 影響 |

### 核心觀察

B 矩陣包含 client-specific 資訊，是 clustering 的最佳 signal。但：
- **不是每一層的 B 都有判別力**：淺層 B 差異小（general features），混入後稀釋信號
- **不同任務/domain 的判別力分佈不同**：某些任務靠淺層區分 client，某些靠深層
- 直接用全部 B 做 clustering，信噪比低

---

## 2. Key Insight

> **不同 layer 的 B 矩陣對 client 資料分佈的敏感度不同，且此 pattern 隨任務/domain 變化。**

自適應挑選高判別力的層作為 clustering feature，比使用全部層（flowertune-clustering）或被動加權（FedADC 的 MADC）更精準。

---

## 2.5 設計合理性論證

### 為什麼 A 矩陣做 global aggregation？

FedSA-LoRA (ICLR 2025) 已有理論和實驗支持：在 LoRA 中，$\mathbf{A}$ 矩陣學習的是 **general knowledge**（跨 client 共通的 input projection / feature extraction direction）。所有 client 共享 $\mathbf{A}$ 能加速收斂，因為：

$$\Delta W = \mathbf{B} \mathbf{A}$$

$\mathbf{A}$ 負責從 input space 投影到 low-rank subspace，這個 subspace 的方向對所有 client 是共通的。Global aggregation 等於讓所有 client 協作找到更好的共用 subspace。

**直接 cite FedSA-LoRA 即可，不需自己論證。**

### 為什麼 B 矩陣做 clustering aggregation 而非完全留本地？

FedSA-LoRA 的假設：B 矩陣是 **client-specific**，所以完全不聚合。但 client-specific $\neq$ client-unique：

1. Non-IID (Dirichlet $\alpha$=0.5) 下，部分 client 的 label distribution 相近
2. Label distribution 相近 → 他們的 B 適應方向也相近（因為 B 學的是從 low-rank subspace 到 output space 的 client-specific mapping）
3. 對這些 client，完全不共享 B 浪費了合作機會——每個 client 只用自己的少量 data 訓練 B，**variance 高**
4. 全部共享（FedAvg）又會把不相關 client 的 B 混進來，引入 **bias**

**Bias-Variance 角度**：

$$\mathbb{E}[\text{error}] = \underbrace{\text{bias}^2}_{\text{聚合不相關 client 的代價}} + \underbrace{\text{variance}}_{\text{data 不夠的代價}}$$

| 方法 | Bias | Variance |
|---|---|---|
| FedAvg（全聚合 B） | 高（混了不同分佈的 client） | 低（所有 client 的 data 一起用） |
| FedSA-LoRA（不聚合 B） | 低（只用自己的 data） | 高（單 client data 少） |
| **FedALC-LoRA（cluster 聚合 B）** | **低**（同 cluster 內分佈相近） | **中偏低**（cluster 內多人共享） |

Clustering 的效果是：在幾乎不增加 bias 的前提下顯著降低 variance。

**理論支持**：Clustered FL 的收斂分析（IFCA, Ghosh et al., ICML 2020）已證明：如果 client 自然形成 $K$ 個 cluster，per-cluster aggregation 的 error bound 為 $O(\sigma^2 / n_k T) + O(\epsilon_k^2)$，其中 $n_k$ 是 cluster 內 client 數、$\epsilon_k$ 是 cluster 內 distribution 差異。我們的 AP clustering 把相近 client 分在同一群 → $\epsilon_k$ 小、$n_k$ 大 → bound tight。

### 為什麼 classifier 留本地？

Classifier（最後的 linear head）直接把 hidden representation 映射到 label：$\hat{y} = \text{softmax}(\mathbf{W}_{\text{cls}} \mathbf{h})$。

Non-IID 下每個 client 的 label distribution $P_n(y)$ 不同 → optimal decision boundary 因人而異。聚合 classifier 等於平均不同的 decision boundary，反而更差。

**文獻支持**：FedPer (Arivazhagan et al., 2019)、LG-FedAvg (Liang et al., 2020) 都有大量實驗證明：聚合 feature extractor、不聚合 classifier head 是 personalized FL 的有效策略。

### 為什麼需要 Layer Selection？（Phase 3）

不是所有層的 B 都包含同等品質的 clustering 信號：

1. **淺層**做 general feature extraction → B 跨 client 差異小 → 混入 similarity 計算會**稀釋信號**
2. **深層**做 task-specific adaptation → B 跨 client 差異大 → 是真正有價值的 **clustering signal**
3. 但「哪些層有判別力」跟 task/domain 有關 → 需要 **adaptive** selection，而非固定規則

**類比 Fisher's Linear Discriminant**（注意：跟 Fisher Information Matrix 是不同概念）：

選高 discriminability 的層 ≈ 最大化 clustering feature 的 between-cluster variance / within-cluster variance：

$$J(l) = \frac{\sigma_{\text{between}}^2(l)}{\sigma_{\text{within}}^2(l)}$$

Cosine dissimilarity 是 $J(l)$ 的 proxy：dissimilarity 高 → between-cluster spread 大。

---

## 3. Method

### 3.1 基礎版（Phase 1：無 layer selection）

每輪 FL round：
1. 各 client 完成 local training，回傳 A + B
2. Server 收集所有 client 的 B 矩陣，flatten 後算 pairwise cosine similarity
3. Affinity Propagation 自動分群（不需指定 K）
4. **A 矩陣**：全域 FedAvg
5. **B 矩陣**：同一 cluster 內 FedAvg，cluster 間獨立
6. 為每個 client 組裝 global A + cluster B，下發

### 3.2 進階版（Phase 3：加入 layer selection）

在 3.1 基礎上，step 2 改為：
1. 對每個 layer 計算 discriminability score（跨 client 的 B cosine dissimilarity）
2. 選出 top-K 個 score 最高的 layer
3. 只用 selected layers 的 B 做 AP clustering

### 3.3 聚合策略

```
對於每個 client c 在 cluster g:
  A_c ← global_avg(A_all_clients)          # A 全域聚合
  B_c ← cluster_avg(B_clients_in_g)        # B 群內聚合
```

---

## 4. 與現有方法的關係

```
FedAvg (full A+B aggregation)
  └── FedSA-LoRA (A 聚合, B 留本地)
        └── FedALC-LoRA (A 聚合, B 按 clustering 群內聚合)
              │
              ├── 退化情況 1: 每個 client 自成一群 → 等同 FedSA-LoRA
              ├── 退化情況 2: 所有 client 同一群 → 等同 FedAvg
              └── Layer selection 退化: K=all layers → 等同 full B clustering
```

### 與 FedADC 的差異

| | FedADC | FedALC-LoRA |
|---|---|---|
| Similarity metric | MADC（二階：比較 similarity profile） | Cosine on selected layers（一階，但降噪） |
| 處理高維方式 | MADC 繞過 structural awareness 問題 | Layer selection 從根本降維，避開 concentration |
| Clustering 對象 | Full model update (A+B) | 只對 B 矩陣 |
| 聚合方式 | 交替 sim/dissim stage，A 和 B 分階段 | A 永遠全域聚合，B 永遠群內聚合 |
| 訓練方式 | 交替 freeze A/B | A 和 B 同時訓練 |

---

## 5. Discriminability Score 設計（Phase 3 Layer Selection 用）

Layer selection 的目標：從所有 LoRA 層中挑出最適合做 clustering feature 的 top-K 層。

### 5.1 Metric A：Cross-client Cosine Dissimilarity

$$\text{score}_A(l) = 1 - \frac{1}{\binom{N}{2}} \sum_{n < m} \cos(\mathbf{B}_l^n,\; \mathbf{B}_l^m)$$

其中 $\mathbf{B}_l^n$ 是 client $n$ 在第 $l$ 層的 B 矩陣（flatten 成向量）。

**意義**：量測「這層的 B 能不能區分 client」。
- $\text{score}_A$ 高 → client 之間在這層走了不同方向 → 適合做 clustering feature
- $\text{score}_A$ 低 → 所有 client 的 B 方向一致 → 無判別力

**盲點**：某層 B 的 norm 接近 0（幾乎沒適應），值太小導致 cosine 方向隨機 → dissimilarity 虛高，實際是 noise。

### 5.2 Metric B：Dissimilarity × Norm（推薦）

$$\text{score}_B(l) = \left(1 - \frac{1}{\binom{N}{2}} \sum_{n < m} \cos(\mathbf{B}_l^n,\; \mathbf{B}_l^m)\right) \times \frac{1}{N} \sum_{n=1}^{N} \|\mathbf{B}_l^n\|_F$$

在 Metric A 基礎上乘以該層 B 的平均 Frobenius norm。

**意義**：結合「判別力」和「適應幅度」。
- $\|\mathbf{B}_l^n\|_F$ 大 → 這層 LoRA 學到很多東西（B 從 0 離開得遠）
- $\|\mathbf{B}_l^n\|_F$ 小 → 這層幾乎沒動，即使 cosine dissimilarity 高也是 noise
- 兩者相乘 → 只有「學了很多且方向不同」的層才會得高分

**優勢**：不需要 gradient，server 端每輪已經拿到所有 client 的 B 矩陣，零額外成本。

### 5.3 Metric C：Fisher-weighted Dissimilarity（備選）

$$\text{score}_C(l) = \left(1 - \frac{1}{\binom{N}{2}} \sum_{n < m} \cos(\mathbf{B}_l^n,\; \mathbf{B}_l^m)\right) \times \text{FIM}(l)$$

$$\text{FIM}(l) = \frac{1}{|D_{\text{proxy}}|} \sum_{d \in D_{\text{proxy}}} \|\nabla_{\theta_l} \ell(\theta, d)\|_2^2$$

用 Fisher Information Matrix score 取代 norm 作為 importance weight。

**意義**：Fisher 量測「微調這層參數對 loss 的影響有多大」，比 norm 更直接反映 task-level 重要性。
- FIM 高 → 這層對 task performance 很敏感 → 重要
- FIM 低 → 改了也不影響 loss → 不重要

**實作方式**（參考 Fed-HeLLo, IEEE TNNLS 2025）：
- Server 端用 global model 在 proxy dataset（eval set 取樣）上跑 forward + backward
- 對每層算 gradient 的 L2 norm² → 即為 FIM score
- 不需要 client 額外上傳任何東西
- 現有 `server_app.py` 的 `evaluate_fn` 已有 global model + eval dataset，只需移除 `torch.no_grad()` 改為算 gradient norm

**與 Metric B 的差異**：
- $\|\mathbf{B}\|_F$（Metric B）量測的是「適應幅度」— 這層 B 離初始值多遠
- $\text{FIM}$（Metric C）量測的是「敏感度」— 動這層的參數，loss 變多少
- 理論上 FIM 更精準，但需要額外的 backward pass；Metric B 完全免費

### 5.4 三個 Metric 的比較

| | Metric A | Metric B | Metric C |
|---|---|---|---|
| 公式 | dissimilarity | dissimilarity × norm | dissimilarity × Fisher |
| 量測 | 判別力 | 判別力 + 適應幅度 | 判別力 + task 重要性 |
| 計算成本 | 零（已有 B） | 零（已有 B） | 每 $T$ 輪一次 backward pass |
| Noise robustness | 低（norm≈0 時虛高） | 高 | 高 |
| 適合 | 快速 baseline | **推薦預設** | Ablation / 精細分析 |

### 5.5 實作位置

Phase 3 在 `fedalc_strategy.py` 的 `aggregate_fit` 中實作：

```python
def _compute_layer_scores(self, client_b_list, method="dissim_norm"):
    """Compute per-layer discriminability scores.

    Args:
        client_b_list: List[List[np.ndarray]], each inner list is one client's B matrices.
        method: "dissim" (Metric A), "dissim_norm" (Metric B), "dissim_fisher" (Metric C).

    Returns:
        scores: List[float], one score per layer.
    """
    n_layers = len(client_b_list[0])
    scores = []
    for l in range(n_layers):
        # Collect layer l's B from all clients, flatten
        vecs = np.stack([c[l].flatten() for c in client_b_list])  # (N, d_l)

        # Cosine dissimilarity
        sim_matrix = cosine_similarity(vecs)
        dissim = 1.0 - sim_matrix.mean()

        if method == "dissim":
            scores.append(dissim)
        elif method == "dissim_norm":
            avg_norm = np.mean([np.linalg.norm(c[l]) for c in client_b_list])
            scores.append(dissim * avg_norm)
        # dissim_fisher: requires FIM scores computed separately
    return scores
```

---

## 6. 設計決策記錄

### 6.1 Clustering 演算法：Affinity Propagation

- 自動決定 K，不需人為指定
- FedADC 已在 FL 場景驗證 AP 可行
- N=30 下 O(N²) 完全無負擔
- 備選：Silhouette scan + KMeans（ablation 比較用）

### 6.2 Client 數量：30

- 現有 baseline（FedAvg、FedSA-LoRA）都在 30 clients 上跑
- 改數量需重跑所有 baseline，Phase 1 先避免
- Ablation 可跑 20 / 50 做對照

### 6.3 不使用 Warm-up

- 減少超參數
- 先驗證無 warm-up 是否足夠
- 如果效果不好再考慮加入

### 6.4 Delta B vs Raw B

標準 LoRA 初始化 B = zero matrix，`B_current - B_init = B_current`，兩者等價。
使用 delta 以相容 PiSSA 等非零初始化變種。

### 6.5 Similarity Metric：Cosine（非 MADC）

Phase 1 先用最簡單的 cosine similarity on full B。理由：
- 先驗證基礎機制（clustering B 是否有效）
- MADC 可作為 ablation baseline
- Phase 3 的 layer selection 是我們對高維問題的解法，不需依賴 MADC

---

## 7. 預期優勢

1. **B 矩陣合作**：比 FedSA-LoRA 多了相似 client 間的 B 知識共享
2. **自動分群**：AP 不需指定 K，自適應 non-IID 程度
3. **Layer selection 降噪**：只用高判別力的層，比全維度 clustering 更精準
4. **可解釋性**：哪些層被選中 → 分析 client heterogeneity 在模型中的位置
5. **退化性好**：極端情況退化為 FedSA-LoRA 或 FedAvg

---

## 8. 潛在風險

- **Clustering B 是否真的比 local B 好？** → Phase 1 直接驗證
- **GLUE 任務間差異太小**，layer selection 的 cross-task pattern 可能不明顯 → Phase 3 換 testbed
- **AP 的 preference 參數**會影響群數，需要調整策略
- **每輪都做 clustering 的開銷**：N=30 下可忽略，但 client 多時需考慮週期性 clustering

---

## 9. Baselines

### 直接比較

| 方法 | A | B | Clustering |
|---|---|---|---|
| FedAvg | global avg | global avg | 無 |
| FedSA-LoRA | global avg | local | 無 |
| FFA-LoRA | freeze | global avg | 無 |
| **FedALC-LoRA** | global avg | **cluster avg** | **AP** |

### Personalization baselines（Phase 3）

| 方法 | 機制 |
|---|---|
| PF2LoRA | 兩層 LoRA (global + local) |
| FedDPA | 兩套 LoRA + instance-wise 加權 |
| HiLoRA | 三層 hierarchy (global → cluster → client) |
