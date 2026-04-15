# FedALC-LWC 設計與 Ablation 規劃

> 最後更新：2026-04-15

## 方法流程

### Phase 0: Warm-up（R1 ~ R_w）

跑 FedSA-LoRA（A global, B local），讓 B 累積 signal。

**何時結束 warm-up（科學判斷，非經驗法則）：**

每輪對全部 B 做 AP clustering（不聚合），計算 silhouette score：

$$s(i) = \frac{b(i) - a(i)}{\max(a(i),\; b(i))}$$

其中 $a(i)$ = client $i$ 跟同群其他 client 的平均距離，$b(i)$ = client $i$ 跟最近其他群的平均距離。全域 silhouette：

$$S = \frac{1}{N} \sum_{i=1}^{N} s(i)$$

**Trigger 條件**：$S > 0.5$（Kaufman & Rousseeuw 1990 定義的 "reasonable structure" threshold）

| $S$ 範圍 | 解讀 | 引用 |
|---|---|---|
| 0.71 - 1.0 | Strong structure | Kaufman & Rousseeuw, 1990 |
| 0.51 - 0.70 | Reasonable structure | ← trigger threshold |
| 0.26 - 0.50 | Weak structure | |
| < 0.25 | No structure | |

從實驗數據看，大約 R2-R4 就到 0.5（α 越小越快）。

### Phase 1: Layer Selection + Clustering

1. 計算所有 $L$ 層的 Metric B score：

   $$\text{score}_B(l) = \underbrace{\left(1 - \frac{1}{\binom{N}{2}} \sum_{n < m} \cos(\mathbf{B}_l^n,\; \mathbf{B}_l^m)\right)}_{\text{cross-client dissimilarity}} \times \underbrace{\frac{1}{N} \sum_{n=1}^{N} \|\mathbf{B}_l^n\|_F}_{\text{mean adaptation magnitude}}$$

   其中 $\mathbf{B}_l^n$ 是 client $n$ 在第 $l$ 層的 B 矩陣（flatten 成向量），$N$ 是 client 總數。

   - **Dissimilarity 項**：量測「這層的 B 能不能區分 client」。值高 → client 間方向不同 → 有判別力
   - **Norm 項**：量測「這層的 LoRA 學了多少」。值高 → B 離初始值（0）遠 → 有 meaningful signal。過濾掉 norm≈0 但 cosine 方向隨機的噪音層

2. 選出 top-$K$ 層：$\mathcal{S} = \text{argtop}_K\; \text{score}_B(l)$
3. 用 selected layers 的 B 做 AP clustering：

   $$\mathbf{f}_n = \text{concat}\left(\text{flatten}(\mathbf{B}_l^n) \;\middle|\; l \in \mathcal{S}\right) \quad \forall n \in [N]$$

   $$\mathbf{S}_{nm} = \cos(\mathbf{f}_n, \mathbf{f}_m) \quad \rightarrow \quad \text{AP}(\mathbf{S}) \rightarrow \text{cluster assignment}$$

4. **聚合用全部 B**（不只 selected layers）→ 同群 client 的所有 B 做 weighted average：

   $$\mathbf{B}_l^{(c)} \leftarrow \frac{\sum_{n \in \mathcal{C}_g} w_n \cdot \mathbf{B}_l^n}{\sum_{n \in \mathcal{C}_g} w_n} \quad \forall l \in [L],\; \forall c \in \mathcal{C}_g$$

5. A 矩陣 global FedAvg，others 留 local：

   $$\mathbf{A} \leftarrow \frac{\sum_{n=1}^{N} w_n \cdot \mathbf{A}^n}{\sum_{n=1}^{N} w_n}$$

### Phase 2: Stable Clustering

- 每輪監控 silhouette score
- 如果 silhouette > 0.9 或連續 M 輪 cluster assignment 不變 → freeze clustering
- 之後沿用固定的 cluster assignment，不重新 AP

## 與 FedALC（Phase 1）的差異

| | FedALC | FedALC-LWC |
|---|---|---|
| Warm-up | 無（R1 就 clustering） | 有（silhouette > 0.5 才開始） |
| Clustering 用的 B | 全部 144 層 flatten | Top-K by Metric B |
| Aggregation 用的 B | 全部 | 全部（跟 FedALC 一樣） |
| Clustering freeze | 無（每輪都做 AP） | 有（silhouette > 0.9 後 freeze） |
| Layer reselect | — | 可選：one-shot（預設）或每 N 輪（`layer-reselect-every`） |
| 預期優勢 | 簡單 | 更好的初始分群 + 更快收斂 + 不會 AP 震盪 |

## Weaknesses

### 1. Warm-up 期間沒有 clustering 收益
R1-R_w 跑的是 FedSA（B local），clustering 帶來的 variance reduction 在 warm-up 期間完全沒有。如果 warm-up 要 3-5 輪，等於浪費了前幾輪的合作機會。

### 2. Layer selection 頻率
預設在 warm-up 結束時做一次 layer selection（one-shot）。可透過 `layer-reselect-every` 參數設定每 N 輪重新選層（0=one-shot）。從實驗看 top-K 在 R3 和 R10 差異不大（都是 ffn_inter 層），one-shot 可能夠用，但 multi-task 場景下可能需要 periodic reselect。

### 3. Single task 下改善有限
實驗顯示 Top-5 Metric B 比 Full B 的 silhouette 只高 0.01-0.07。差距不大，因為 binary task 的信號簡單，全部 B 已經能分得很好。

### 4. Metric B 依賴 norm
Metric B = dissimilarity × norm。如果某層的 norm 很大但判別力低（所有 client 的 B 方向一致但幅度大），Metric B 會給虛高的分數。目前沒觀察到這個問題，但理論上可能發生。

### 5. K 的選擇
Top-K 的 K 是超參數。K 太小（5）→ 在 R1 不穩定（QNLI α=0.5 R1 的 Top-5 silhouette 只有 0.015）。K 太大 → 接近 Full B，失去 layer selection 的意義。

## 應用到 FedLEASE 的 Multi-task 場景

FedLEASE 設定：4 個 GLUE tasks × 4 clients = 16 clients，每 client 只做一個 task。

**預期差異：**

1. **Layer selection 會更有效**：不同 task 的 client 在不同層有差異。SST-2 的 client 可能在 layer 20 差異大，QNLI 在 layer 15。等權平均（FedLEASE）會互相抵消，adaptive Metric B selection 能挑出真正有差異的層。

2. **Top-K 的組成可能會變**：single task 下 top-K 全是 ffn_inter；multi-task 下 attention 層可能也有判別力（不同 task 的 attention pattern 不同）。

3. **Cluster 數會跟 task 數相關**：FedLEASE 觀察到 K=4 恰好是 4 個 task。FedALC-LWC 的 AP 應該也會自動分出 ~4 群。

4. **Global layer selection 的限制**：我們的 Metric B 是全域的（all 16 clients），但不同 task 可能需要不同的 top-K 層。FedALC-LWC 無法處理這個——可能是 future work。

## Config 參數（pyproject.toml）

| 參數 | 預設值 | 說明 |
|---|---|---|
| `aggregation-mode` | `"fedsa"` | 設為 `"fedalc-lwc"` 啟用 |
| `warmup-sil-threshold` | `0.5` | Phase 0→1 的 silhouette 門檻 |
| `freeze-sil-threshold` | `0.9` | Phase 1→2 的 silhouette 門檻 |
| `layer-selection-k` | `10` | Top-K 層數 |
| `layer-reselect-every` | `0` | 0=one-shot，N=每 N 輪重新選層 |

## 實作檔案

| 檔案 | 說明 |
|---|---|
| `bert/fedalc_lwc_strategy.py` | FedALCLWCStrategy 主邏輯（獨立於 fedalc_strategy.py） |
| `bert/server_app.py` | `"fedalc-lwc"` strategy selection + config 讀取 |
| `pyproject.toml` | LWC 專用參數 |
| `run_fedalc_lwc.sh` | 批次跑腳本（支援 alpha 參數） |

## 設計 vs 實作對照

| 設計 | 實作 | 一致？ |
|---|---|---|
| Phase 0: FedSA mode | aggregate_fit phase 0: A global, B per-client local | ✅ |
| Phase 0: trial AP 算 silhouette | `_trial_ap_silhouette()` on full B | ✅ |
| Phase 0→1: silhouette > 0.5 | `if sil >= warmup_sil_threshold` | ✅ |
| Phase 1: Metric B = dissim × norm | `_compute_layer_scores()` | ✅ |
| Phase 1: top-K selection | argsort + slice | ✅ |
| Phase 1: selected layers AP, 全部 B aggregation | `_cluster_with_selected_layers()` | ✅ |
| Phase 1→2: sil > 0.9 or 3 輪穩定 | `sil >= freeze_sil_threshold or rounds_stable >= 3` | ✅ |
| Phase 2: frozen cluster groups | `frozen_cluster_groups: Dict[int, List[str]]` | ✅ |
| configure_fit: phase 0 送 own B, phase 1/2 送 cluster B | 統一用 `client_b_matrices.get(cid)`，phase 0 存 own B，phase 1/2 存 cluster B | ✅（行為正確） |
| Layer reselect | `layer_reselect_every` + `_rounds_since_reselect` | ✅（設計外的額外功能） |

## Ablation 實驗計畫

### A1: Clustering Feature 比較

| Ablation | 說明 |
|---|---|
| Full B (144 layers) | FedALC baseline |
| ffn_inter only (24) | 只用 FFN intermediate 層 |
| Top-10 Metric B | Adaptive selection, K=10 |
| Top-5 Metric B | Adaptive selection, K=5 |
| Per-layer equal-weight | FedLEASE 風格 |

**指標**：silhouette score per round, best accuracy, convergence speed

### A2: K 的影響

K = 5, 10, 24 (all ffn_inter), 48, 144 (all)

### A3: Warm-up 輪數

| Ablation | 說明 |
|---|---|
| No warm-up (R1 clustering) | 現有 FedALC |
| Warm-up until silhouette > 0.5 | Adaptive（約 R2-R4） |
| Fixed warm-up R3 | 固定 3 輪 |
| Fixed warm-up R5 | 固定 5 輪 |

### A4: Clustering Freeze

| Ablation | 說明 |
|---|---|
| 每輪 AP | 現有 FedALC（會震盪） |
| Silhouette > 0.9 後 freeze | Adaptive freeze |
| 連續 3 輪不變後 freeze | Stability-based |

### A5: Clustering 演算法

| Ablation | 說明 |
|---|---|
| AP（現有） | 自動 K，但會震盪 |
| Agglomerative + silhouette | 穩定，FedLEASE 用的 |

### A6: α 的影響（已完成）

α = 0.3, 0.5

**Finding**: α 越小 FedALC 優勢越大（+1.5%/+4.0% at α=0.3 vs +0.3%/+1.4% at α=0.5）

## 優先順序

1. **A1（clustering feature 比較）**— 最重要，直接驗證 layer-wise 有沒有用
2. **A3（warm-up）**— 驗證初始分群品質的影響
3. **A4（clustering freeze）**— 解決 AP 震盪
4. **A2（K 的影響）**— K 的 sensitivity analysis
5. **A5（clustering 演算法）**— AP vs Agglomerative
6. **A6**— 已完成
