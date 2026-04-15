# Federated LoRA Clustering 方法比較

> 比較 related work 中各方法使用的 clustering 技術

## 總覽

| 方法 | Clustering 演算法 | 選 K 方式 | Similarity metric | Clustering 對象 | 需要 exemplar？ |
|---|---|---|---|---|---|
| **FedALC-LoRA (ours)** | Affinity Propagation | AP 自動 | Cosine similarity on full B | B 矩陣 only | 不需要（只用 labels） |
| **FedADC** | Affinity Propagation | AP 自動 | MADC（二階 cosine profile） | Full model update (A+B) | 是（exemplar 做 intra-cluster agg） |
| **FedLEASE** | Agglomerative Hierarchical | Silhouette scan K=2..K_max | Per-layer averaged cosine distance on B | B 矩陣 only | 不需要 |
| **HiLoRA** | Spectral Clustering | Eigengap heuristic | SVD + principal angles on B | B 矩陣 only | 不需要 |
| **IFCA** | — (固定 K) | 預先指定 | Loss-based（client 選 loss 最小的 cluster） | Full model | 不需要 |
| **FedGroup** | KMeans | 預先指定 | EDC（Euclidean-norm-weighted Decomposed Cosine） | Full model update | 不需要 |
| **CORNFLQS** | KMeans | Silhouette scan | Cosine similarity | Full model update | 不需要 |

## 各方法細節

### FedALC-LoRA（ours）

```
1. 每個 client 的 B 矩陣 flatten → 向量
2. cosine_similarity → N×N matrix
3. AffinityPropagation(affinity='precomputed', damping=0.5)
4. labels = ap.fit_predict(sim_matrix)
5. 同 label 的 B 做 weighted average
```

- **優點**：不需指定 K，AP 自動決定
- **缺點**：AP 的 exemplar 機制我們用不到（只用 labels）；後期 similarity matrix degenerate 時會震盪
- **改進方向**：可換成 Agglomerative + silhouette（更穩定、不產生多餘的 exemplar）

### FedADC

```
1. 算 pairwise cosine similarity → N×N matrix
2. MADC(n,m) = mean(|cos(n,l) - cos(m,l)|) for all l≠n,m → N×N matrix
3. Preference matrix:
   - Similarity clustering: preference = -madc（小 = 像 → 同群）
   - Dissimilarity clustering: preference = +madc（大 = 不像 → 同群）
4. AP clustering → exemplars = cluster heads
5. Exemplar 負責做 intra-cluster aggregation
```

- **AP 在 FedADC 裡有意義**：exemplar 是真實的 device，負責收集群內 model 做聚合
- **MADC vs cosine**：MADC 是二階 similarity（比較「跟其他人的關係模式」），解決 cosine 的 structural awareness 問題

### FedLEASE

```
1. Per-layer cosine distance: d(i,j) = (1/|L|) Σ_l (1 - cos(B_i^l, B_j^l))
2. Agglomerative Hierarchical Clustering → 建 dendrogram
3. 對 K=2,3,...,K_max 切 dendrogram → 算 silhouette score
4. K* = argmax silhouette
5. 每個 cluster 初始化一個 LoRA expert（A+B averaged）
```

- **Per-layer averaged**：逐層算 cosine 再等權平均，隱式的 layer-wise 處理
- **Agglomerative**：建一次 dendrogram 可嘗試所有 K，穩定
- **跟我們的差異**：FedLEASE 建多個 expert（MoE），我們只改 B 的聚合方式

### HiLoRA

```
1. 每個 client 的 B 做 SVD → top-r left singular vectors U_i
2. Subspace distance: d(i,j) = 1 - (1/r) ||U_i^T U_j||_F^2（principal angles）
3. Distance → Gaussian kernel affinity matrix: S_ij = exp(-d_ij^2 / (2σ^2))
4. Spectral Clustering，sweep K ∈ [K_min, K_max]
5. Eigengap of normalized Laplacian → 自動選 K*
```

- **SVD + principal angles**：比直接 cosine 更 robust（reparameterization invariant）
- **Spectral Clustering**：最後一步用 KMeans 在 eigenvector 空間分群
- **Eigengap**：理論優雅但 N=30 時只有 30 個 eigenvalue，統計意義有限

## Clustering 演算法比較

### 不需指定 K 的方法

| 演算法 | 原理 | 穩定性 | 計算量 | 適合場景 |
|---|---|---|---|---|
| **Affinity Propagation** | Message passing 選 exemplar | 後期可能震盪 | O(N² × iter) | 需要真實 exemplar 時（FedADC） |
| **Agglomerative + silhouette** | 建 dendrogram + 掃 K | 穩定 | O(N²) + O(K_max) | 只需要 labels 時 ✅ |
| **Spectral + eigengap** | Laplacian eigenvalue gap | 穩定但 N 小時不準 | O(N³) | N 大時 |
| **HDBSCAN** | 密度層次聚類 | 穩定 | O(N log N) | 任意形狀 cluster |

### 需要指定 K 的方法

| 演算法 | 原理 | 穩定性 | 計算量 |
|---|---|---|---|
| **KMeans** | Centroid-based | 穩定 | O(NKd) |
| **KMeans + silhouette scan** | 掃 K=2..K_max 選最佳 | 穩定 | O(NKd × K_max) |

## 對 FedALC-LoRA 的建議

目前用 AP，但我們**不需要 exemplar**——只需要 labels。AP 的震盪問題是額外的負擔。

**最適合的替代方案**：Agglomerative + silhouette（跟 FedLEASE 一樣）
- 穩定，不會震盪
- 不產生多餘的 exemplar
- 建一次 dendrogram 可嘗試所有 K
- N=30 計算量極小
- 可以做 ablation：AP vs Agglomerative 比較

**或保持 AP + fallback**：如果 cluster 結構穩定（silhouette > threshold），停止重新 clustering。
