# HiLoRA: Hierarchical Low-Rank Adaptation for Personalized Federated Learning

- **發表**: CVPR 2026（arXiv 2026/03）
- **連結**: [arXiv](https://arxiv.org/abs/2603.02785)
- **作者**: Zihao Peng et al., Beijing Normal University
- **PDF**: `papers/HiLoRA- Hierarchical Low-Rank Adaptation for Personalized Federated Learning.pdf`

---

## 1. 核心方法

### 1.1 三層 LoRA 架構

每個 client i 的 weight update 是三套獨立 LoRA 的疊加：

```
ΔW_i = B_r A_r + B_{c,j(i)} A_{c,j(i)} + B_{ℓ,i} A_{ℓ,i}
```

| 層級 | 參數 | 共享範圍 | 學什麼 |
|---|---|---|---|
| **Root** | (B_r, A_r) | 所有 N 個 client 共用 | Global pattern |
| **Cluster** | (B_{c,j}, A_{c,j}) | 同 cluster 內共用 | Subgroup commonality |
| **Leaf** | (B_{ℓ,i}, A_{ℓ,i}) | 每個 client 獨有 | Client-specific residual |

**注意**：A 和 B 在每一層都有，都會被訓練和聚合。沒有 A/B 角色分離。

### 1.2 Cross-tier Orthogonality

三層的 B 矩陣的 column space 必須互相正交：

```
R(B_r) ⊥ R(B_{c,j(i)}) ⊥ R(B_{ℓ,i})
```

用 regularization 實現（不是硬約束）：
- Cluster stage: γ_c × ||B_r^T B_{c,j}||_F^2
- Leaf stage: γ_c × ||B_r^T B_{ℓ,i}||_F^2 + γ_ℓ × ||B_{c,j}^T B_{ℓ,i}||_F^2

**目的**：防止三層 LoRA 學到重複的東西。每層只學「上一層沒學到的 residual」。

### 1.3 Product Space Aggregation

不直接平均 B 和 A，而是：
1. 先算 product：ΔW = Σ π_i B_i A_i
2. 再做 truncated SVD：ΔW = U Σ V^T
3. 設 B := U, A := Σ V^T

**理由**：直接平均 B 和 A 會產生 cross terms，不等於平均 BA。

---

## 2. LoRA-Subspace Adaptive Clustering

### Step 1: Basis Extraction + Stabilization

每輪收集 client 的 B 矩陣，normalize 後做 EMA：

```
B̄_i^(t) = λ B̄_i^(t-1) + (1-λ) B̂_i^(t)
```

λ 是 decay 參數（0 < λ < 1），用來穩定跨 round 的 basis。

### Step 2: Subspace Representation

對 B̄_i^(t) 做 SVD，取 top-r left singular vectors U_i ∈ R^{p×r}。

**目的**：用 SVD 降維 + 去噪，得到 adaptation 的主方向。

### Step 3: Distance Metric — Principal Angles

```
d_{ij} = 1 - (1/r) ||U_i^T U_j||_F^2
```

這是基於 principal angles 的 subspace distance：
- d = 0 → 兩個 client 的 adaptation subspace 完全相同
- d = 1 → 完全正交

如果有多個 LoRA layer，per-layer 算 d 再取平均。

### Step 4: Spectral Clustering

1. Distance matrix → Gaussian kernel affinity matrix：S_{ij} = exp(-d_{ij}^2 / (2σ^2))
2. σ = median of off-diagonal distances
3. Spectral Clustering，sweep K ∈ [K_min, K_max]，用 eigengap 自動選 K*

**與 FedALC-LoRA 的差異**：
- HiLoRA 用 SVD + principal angles + spectral clustering
- FedALC-LoRA 用 cosine similarity + AP（更簡單）

---

## 3. 訓練流程：Cascaded Tier-wise Optimization

**分三個 stage 依序訓練**（不是同時）：

### Stage 1: Root（Global）
- 所有 client 一起訓練 (B_r, A_r)
- Server 在 product space 聚合 + SVD
- 跑到收斂（relative step-size criterion：ρ_t ≤ τ_rel）
- 收斂後 **freeze** B_r, A_r

### Stage 2: Clustering + Cluster LoRA
- 用 Stage 1 訓練好的 B_r 做 clustering（§2 的方法）
- 每個 cluster 內訓練 (B_{c,j}, A_{c,j})，加 orthogonality penalty vs B_r
- Cluster 內 product space 聚合 + SVD
- 收斂後 **freeze** B_{c,j}, A_{c,j}

### Stage 3: Leaf（Client-specific）
- 每個 client 獨立訓練 (B_{ℓ,i}, A_{ℓ,i})，加 orthogonality penalty vs B_r 和 B_{c,j}
- **不聚合**（純 local training）
- 收斂後 freeze

**總 round 預算**：T_root + T_cluster + T_leaf = 50（跟 baseline 一樣都是 50 rounds）

---

## 4. Theoretical Guarantee

HiLoRA 提供了 tier-wise generalization bound（Theorem 1）：

```
L_D_i(h^(r,c,ℓ)) - inf L_D_i(h) ≤ GE + DS + EO
```

三個 component：
- **GE (Generalization)**：由 Rademacher complexity 控制，orthogonality 縮小 function class → 更 tight
- **DS (Distribution Shift)**：clustering 讓 cluster 內分佈更接近 → disc(D_i, C_{j(i)}) 更小
- **EO (Empirical Optimization)**：每層的 empirical loss 非遞增（orthogonality 確保每層只學 residual）

---

## 5. 實驗設定

### Model
- ViT-Base pretrained on ImageNet-21K
- LoRA 插在 attention 的 query 和 value projections
- **純 CV，沒有 NLP 實驗**

### Datasets

| Dataset | Clients | Non-IID 方式 | Classes |
|---|---|---|---|
| **CIFAR-100** | 100 | GL-Dir(0.3), SC-Dir(3), Patho(10) | 100 |
| **DomainNet** | 90 | 6 domains × Dirichlet α=0.6 | top 10 classes |

CIFAR-100 的三種 non-IID：
- **GL-Dir(α=0.3)**：所有 class 上做 Dirichlet，strong skew
- **SC-Dir(α=3)**：superclass 層級做 Dirichlet，每人集中 1-2 個 superclass
- **Patho(10)**：每人只有 10 個 class（pathological non-IID）

### Training

| 參數 | 值 |
|---|---|
| Total rounds | 50（T_root + T_cluster + T_leaf = 50） |
| Local epochs | 1（CIFAR-100）/ 2（DomainNet） |
| LoRA target | query, value |
| LoRA rank | 未在主文明確提到（需查 appendix） |
| Unseen clients | Hold out 20% 做 generalization 測試 |
| Unseen 評估 | 5 local epochs fine-tuning 後測 |

### Baselines（9 個）

| 方法 | 類型 |
|---|---|
| Local-LoRA | 不聚合 |
| FedIT | 直接聚合 B 和 A |
| FlexLoRA | Product space BA 聚合 |
| FedSA-LoRA | Shared A, local B |
| FDLoRA | 交替更新 shared/client adapter |
| FedDPA-F | Sequential training（兩套 LoRA） |
| FedDPA-T | Alternating training（兩套 LoRA） |
| PF2LoRA | Global + personal branch, auto rank |
| FedALT | Adaptive gating mix individual + rest-of-world adapter |

---

## 6. 主要結果

### CIFAR-100 Personalization（Mean Accuracy）

| Method | GL-Dir(0.3) | SC-Dir(3) | Patho(10) |
|---|---|---|---|
| FedSA-LoRA | 0.786 | 0.734 | 0.860 |
| FedDPA-T | 0.803 | 0.895 | 0.928 |
| PF2LoRA | 0.793 | 0.873 | 0.916 |
| FedALT | 0.809 | 0.912 | 0.929 |
| **HiLoRA** | **0.846** | **0.934** | **0.941** |

### Unseen Client Generalization

| Method | GL-Dir(0.3) | SC-Dir(0.3) | Patho(10) |
|---|---|---|---|
| FedSA-LoRA | 0.813 | 0.765 | 0.846 |
| **HiLoRA** | **0.841** | **0.859** | **0.940** |

HiLoRA 在 personalization 和 generalization 上都顯著領先。

---

## 7. 與 FedALC-LoRA 的比較

| | HiLoRA | FedALC-LoRA |
|---|---|---|
| **架構** | 三套獨立 LoRA 疊加 | 一套 LoRA，A/B 分離處理 |
| **參數量** | 3× LoRA 參數 | 1× LoRA 參數 |
| **A/B 角色** | 不區分，每層都有 A 和 B | A = general (global), B = specific (cluster) |
| **Clustering 依據** | SVD + principal angles on B | Cosine similarity on B |
| **Clustering 方法** | Spectral clustering + eigengap | Affinity Propagation |
| **Orthogonality** | 需要正交約束 + regularization | 不需要（只有一套 LoRA） |
| **訓練流程** | 分三階段依序（root → cluster → leaf） | 每輪同時訓練 A+B |
| **Classifier** | 沒有明確分離 | 明確分離，留本地 |
| **Layer selection** | 無 | 有（Phase 3） |
| **模態** | 純 CV（ViT-Base） | NLP（RoBERTa），計畫擴展到 ViT |
| **理論** | 有 tier-wise generalization bound | Bias-variance motivation + cite IFCA |

### 核心設計哲學差異

**HiLoRA**：用更多參數 + 正交約束來強制分離 global/cluster/local knowledge。三套 LoRA 各管各的，靠 orthogonality 避免重複。**重方法**，效果好但計算量大。

**FedALC-LoRA**：基於 FedSA-LoRA 的 A/B 功能差異 insight，不加參數，只改聚合策略。A 天然就是 general → global avg；B 天然就是 specific → cluster avg。**輕方法**，利用 LoRA 本身的結構性質。

### Related Work 定位建議

> HiLoRA organizes three independent LoRA modules (root, cluster, leaf) with orthogonality constraints to separate hierarchical knowledge, tripling the parameter count. In contrast, FedALC-LoRA exploits the functional asymmetry between A and B matrices established by FedSA-LoRA: A captures general knowledge via global aggregation, while B encodes client-specific adaptations via cluster-level aggregation. This achieves cluster-level personalization without additional parameters or orthogonality constraints.
