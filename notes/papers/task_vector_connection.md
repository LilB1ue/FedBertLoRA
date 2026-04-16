# LoRA, FedALC 與 Task Vector 研究的連結

> 最後更新：2026-04-16

## Section 1: Task Vector 基本概念

**Task Arithmetic**（Ilharco et al., ICLR 2023）提出 task vector：

$$\tau_{\text{task}} = \theta_{\text{ft}} - \theta_{\text{pre}}$$

其中 $\theta_{\text{ft}}$ 是 fine-tuned 模型參數，$\theta_{\text{pre}}$ 是 pre-trained 參數。τ 代表「這個 task 學到的適應」。

**核心性質**：task vector 可做 arithmetic
- Addition: $\tau_A + \tau_B$ ≈ 同時具備 A 和 B 能力
- Subtraction: $-\tau_A$ = 移除 A 能力
- Scaling: $\lambda \tau_A$ = 調整 A 強度

## Section 2: LoRA 的 A/B 是 Task Vector 的 Low-Rank Factorization

LoRA 的 fine-tuning delta：

$$\Delta W = BA, \quad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times k},\; r \ll \min(d, k)$$

這就是一個 **rank-r 的 task vector**。相較於 full fine-tuning 的 full-rank $\tau$，LoRA 在 low-rank subspace 內表達 task adaptation。

**A 和 B 的角色**：
- $A \in \mathbb{R}^{r \times k}$：把 input 投影到 r 維 subspace 的方向（subspace basis）
- $B \in \mathbb{R}^{d \times r}$：把 r 維 subspace 投回 output 的線性組合（task-specific mapping）
- $\Delta W = BA$：完整的 rank-r task vector

**FedSA-LoRA 的觀察**（A global, B local）在這個框架下變得自然：
- $A$ = **shared subspace basis**（所有 task/client 共用的 general feature directions）
- $B$ = **task-specific linear combination**（在該 basis 下的個人化權重）

**ΔB 的意義**：
$$\Delta B = B_{\text{current}} - B_{\text{init}}$$

對標準 LoRA（$B_{\text{init}} = 0$），$\Delta B = B_{\text{current}}$。對 LoRA 變體（PiSSA 等 $B_{\text{init}} \neq 0$），$\Delta B$ 捕捉純粹學到的 task signal，排除 initialization。

## Section 3: FedALC = Federated Task Vector Clustering with Shared Basis

在 task vector 框架下重新詮釋 FedALC：

| FedALC 操作 | Task vector 解讀 |
|---|---|
| Client $i$ 的 $\Delta W_i = B_i A_i$ | Client $i$ 的 task vector（rank-r） |
| A global FedAvg | 所有 client 共享 subspace basis |
| B per-cluster FedAvg | 同群 client 合併 task vectors |
| Others 留 local | Classifier head 保留個人化 decision boundary |

**FedALC 在做的事**：
$$\{\Delta W_i\}_{i=1}^N \xrightarrow{\text{cluster by }B_i} \{\mathcal{C}_g\} \xrightarrow{\text{avg within cluster}} \{\bar{B}_{\mathcal{C}_g} \bar{A}\}$$

本質上是 **federated task vector clustering**：
- 把分散在 N 個 client 的 task vectors 按相似度分群
- 群內合併（降低 variance），群間獨立（避免 bias）
- A 保持全域聚合（維持 shared basis）

## Section 4: FedALC-AP-Multi 的 Task Vector 視角

**Cumulative ΔB as task signal**：
$$\overline{\Delta B}_i = \frac{1}{W} \sum_{r=1}^{W} (B_i^{(r)} - B_i^{(0)})$$

這是 **client $i$ 在 warm-up 期間累積的 task vector 分量**。
- 單輪 $B^{(r)}$：一個 noisy 的 task vector snapshot
- 累積 $\overline{\Delta B}$：多輪平均的穩定 task vector

**Hopkins statistic on task vector space**：
測試「task vectors 是否已經形成 clusters」：
- Warm-up 早期：所有 client 的 ΔB 還小且隨機 → Hopkins ≈ 0.5（無結構）
- B 累積 signal 後：不同 client 的 ΔB 分化到不同方向 → Hopkins → 1.0（有結構）
- Trigger clustering 當 H > 0.75 → 確保 task vectors 已經 well-formed 才 clustering

**關鍵實作細節 — Hopkins 不能直接吃 full ΔB**：
LoRA 全層 ΔB 的展平維度 D 常達 50K 量級（例：roberta-large r=8 的所有 B matrix）。在這個維度下：
1. **Curse of dimensionality**：pairwise 距離趨近等值，讓 Hopkins 公式中的 `u_sum / (u_sum + w_sum)` 退化到 0.5，完全失去區分能力
2. **數值溢位**：`u_dist ** d` 在 d ≈ 50000 時會爆炸到 inf / nan

因此 FedALC-AP-Multi 先用 **Metric B（layer-wise dissim × norm）挑 top-K 層**，把 ΔB 投影到低維子空間（通常降到 D ≈ 10K 以下），再在這個子空間做 Hopkins / AP clustering。這個降維**不是可選外掛**，而是 Hopkins 在 task vector 空間運作的必要條件。

## Section 5: 關鍵相關文獻

### Task Arithmetic & Model Merging

1. **Task Arithmetic** (Ilharco et al., ICLR 2023)
   - [arXiv](https://arxiv.org/abs/2212.04089)
   - 首次系統化 task vector arithmetic
   - **跟 FedALC 的關係**：提供 ΔW 作為 task vector 的 theoretical grounding

2. **TIES-Merging** (Yadav et al., NeurIPS 2024)
   - [arXiv](https://arxiv.org/abs/2306.01708)
   - Trim + Elect Sign + Disjoint Merge
   - **跟 FedALC 的關係**：啟發 cluster 內 merge 時的 sign conflict 處理（future work）

3. **DARE** (Yu et al., ICML 2024)
   - [arXiv](https://arxiv.org/abs/2311.03099)
   - Drop And REscale before merging
   - **跟 FedALC 的關係**：啟發 merge 前的 regularization

4. **LoRA Hub** (Huang et al., EMNLP 2024)
   - [arXiv](https://arxiv.org/abs/2307.13269)
   - Dynamic composition of trained LoRAs
   - **對比**：LoRA Hub 是 centralized multi-task 動態組合，FedALC 是 FL 下靜態 clustering

5. **AdaMerging** (Yang et al., ICLR 2024)
   - [arXiv](https://arxiv.org/abs/2310.02575)
   - Learned merging coefficients via unsupervised test-time adaptation
   - **對比**：AdaMerging 學 coefficient，FedALC 用 FedAvg weights

### Federated LoRA Clustering

6. **FedLEASE** (Zhao et al., NeurIPS 2025)
   - [arXiv](https://arxiv.org/abs/2509.15087)
   - Per-cluster LoRA experts + MoE router
   - **直接 baseline**：clustering 概念相似但加 MoE

7. **HiLoRA** (Peng et al., CVPR 2026)
   - [arXiv](https://arxiv.org/abs/2603.02785)
   - Three-tier LoRA hierarchy with orthogonality
   - **直接 baseline**：同樣對 B 做 clustering 但加 hierarchy

8. **FL-TAC** (Ping et al., ICLR 2024)
   - [arXiv](https://arxiv.org/abs/2404.15384)
   - Multi-task adapter clustering
   - **場景不同**：每 client 多 task adapter，FedALC 是 single task per client

### 經典 Clustered FL

9. **IFCA** (Ghosh et al., ICML 2020)
   - Iterative Federated Clustering Algorithm
   - Loss-based cluster selection
   - **啟發**：經典 clustered FL framework

10. **CFL** (Sattler et al., NeurIPS 2020)
    - Clustered Federated Learning via gradient cosine similarity
    - **啟發**：用參數方向做 clustering 的思路

## Section 6: FedALC 的獨特定位

**Unique combination 在 FL + LoRA landscape 中**：

| 方法 | LoRA? | Clustering? | Per-cluster A? | MoE? | Hierarchy? | 場景 |
|---|---|---|---|---|---|---|
| Task Arithmetic | No | No | - | - | - | Centralized |
| FedAvg | Yes | No | - | - | - | FL |
| FedSA-LoRA | Yes | No | - | - | - | FL |
| IFCA | No | Yes | - | - | - | FL |
| FL-TAC | Yes | Yes | Per-cluster | - | - | Multi-task FL |
| HiLoRA | Yes | Yes | Per-tier | - | Yes | FL |
| FedLEASE | Yes | Yes | Per-cluster | Yes | - | Multi-task FL |
| **FedALC** | Yes | Yes | **Global** | No | No | FL (single/multi-task) |
| **FedALC-AP-Multi** | Yes | Yes | **Global + warm-up** | No | No | FL (single/multi-task) |

**FedALC 的 niche**：
- **唯一保留 A global** 的 clustering 方法（輕量、參數效率高）
- 利用 FedSA-LoRA 的 A/B 功能分離 insight
- 以最少的 architectural change 達到 cluster-level personalization

**FedALC-AP-Multi 的進一步貢獻**：
- 首次在 FL+LoRA 中使用 **Hopkins statistic 作為 clustering trigger**（cluster tendency test）
- 首次將 **cumulative ΔB 作為 clustering feature**（task vector 的穩定 proxy）
- 首次結合 **FedSA warm-up with adaptive termination**（不同於 FedLEASE 的固定 E）

## Section 7: Paper 寫作建議

### Theoretical motivation paragraph

> LoRA's low-rank decomposition $\Delta W = BA$ can be interpreted as a rank-$r$ task vector (Ilharco et al., 2023), where $A$ provides the subspace basis and $B$ provides the task-specific linear combination. Under this view, FedSA-LoRA's insight—"$A$ captures general knowledge, $B$ captures client-specific adaptation"—maps naturally to $A$ as the shared basis and $B$ as the task-specific direction. Our work extends this interpretation by clustering clients based on their task vectors (B matrices) while maintaining a shared basis (globally aggregated A), yielding a federated analog of task vector merging.

### Contribution framing

> We make three contributions that collectively position FedALC as a lightweight federated task vector clustering framework:
> (1) **Federated task vector clustering**: we cluster clients by B-matrix similarity, a proxy for task vector alignment in LoRA's low-rank space;
> (2) **Adaptive clustering trigger**: Hopkins statistic on cumulative ΔB detects when task vectors have clustered, avoiding the fixed warm-up length of prior work (FedLEASE);
> (3) **Cumulative task signal**: averaging ΔB over the warm-up phase reduces single-round noise, providing a more stable clustering feature than post-init B.
