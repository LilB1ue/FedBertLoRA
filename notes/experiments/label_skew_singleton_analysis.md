---
tags: [experiments, fedalc, clustering, singleton]
related: [ap_vs_lwc_clustering_deepdive.md, ../concepts/fedalc_methods_comparison.md, ../concepts/evaluation_metrics.md]
---

# Singleton cluster 成因分析：QNLI α=0.5 的 pid_16

> 最後更新：2026-04-24
> TL;DR：FedALC-AP on QNLI α=0.5 從 R1 起穩定出現 size=1 cluster。根因不是 dataset 大小、不是 clustering algorithm bug，而是 **Dirichlet 切分造成 pid_16 的 label marginal 跟其他高 accuracy client 完全相反**（99.5% label_1 vs 99%+ label_0）。這是**結構性 outlier**，不是 clustering artifact。

## 現象回顧

FedALC-AP on QNLI α=0.5（`logs/20260406_203614_fedalc_a0.5/qnli_fedalc_a0.5/`）30 輪裡：
- **30/30 輪都有 size=1 的 cluster**
- Singleton client 全程是同一個（`client_16` = partition_id 16）
- Silhouette 從 R1 的 0.134 升到 R15 的 0.957，singleton 不消失

此現象在 FedALC-AP-LWC（top-K layer selection）下同樣存在 — **跨 clustering method 的 structural phenomenon**。

## 假設檢驗

曾有 4 條 hypothesis 解釋這個 singleton：

| Hypothesis | 驗證結果 |
|---|---|
| H1：client_16 是小 dataset client，更新不夠 → singleton | ❌ **反駁**。client_16 `n_train=8736`，是全 30 client 第 3 大 |
| H2：Cosine in high-D 誤判 | ❌ **反駁**。LWC 的 top-K 降維（~10K dim）下 pid_16 仍然 singleton |
| H3：AP damping=0.5 太鬆 | ❓ 未測，但 AP-LWC 用相同 damping 也見同 pid singleton → 非主因 |
| **H4：Label 分布極偏**（inverse marginal） | ✅ **命中**。見下表 |

## 驗證 H4：per-client label 分布

由 `tools/dump_partition_stats.py --task qnli --alpha 0.5 --seed 42` 產出，資料同 FL 訓練實際使用（Dirichlet + stratified 80/20 split）。

### 高 accuracy client 的 label marginal

| partition_id | n_train | label_0 | label_1 | **label_1 比例** | cluster 行為 |
|:---:|---:|---:|---:|:---:|---|
| **16** | 8736 | 42 | 8694 | **99.52%** | 🔴 **SINGLETON** (30/30) |
| 1  | 3225 | 3220 | 5 | 0.16% | 🟢 cluster 內 |
| 9  | 1876 | 1874 | 2 | 0.11% | 🟢 cluster 內 |
| 20 | 2797 | 2752 | 45 | 1.61% | 🟢 cluster 內 |

### 對照小 dataset client（也有偏 label，但不極端）

| partition_id | n_train | label_0 | label_1 | label_0 比例 | cluster 行為 |
|:---:|---:|---:|---:|:---:|---|
| 4  | 56 | 40 | 16 | 71.4% | 🟢 cluster 內（不極端 → 不 singleton） |
| 26 | 58 | 49 | 9  | 84.5% | 🟢 cluster 內 |
| 12 | 171 | 60 | 111 | 35.1% | 🟢 cluster 內 |

## 機制：為何 label 相反 ⇒ B 方向相反 ⇒ cosine ≈ 負 ⇒ singleton

LoRA 的 output 由 $y = Wx + BAx$ 決定。對 binary classification：

- Client $i$ 的 data 99.5% 是 label_1 → local gradient 持續推 B 往「預測 label_1」的方向
- Client $j$ 的 data 99.5% 是 label_0 → local gradient 持續推 B 往「預測 label_0」的方向
- 兩個方向在決策邊界上**幾乎反號**：$B_i \approx -B_j$

因此：

$$\cos(B_i, B_j) \approx -1 \quad \Rightarrow \quad \text{distance}(B_i, B_j) = 1 - \cos \approx 2$$

AP 的 exemplar 選擇邏輯：若 pid_16 跟所有其他 client 的 similarity 都很低（負值或接近 0），AP 找不到比它「更接近 pid_16」的 exemplar → **pid_16 自選為 exemplar**，沒有其他 client 會加入它的 cluster → **singleton**。

同樣邏輯套到 Agglomerative clustering：pid_16 會留到合併樹最後一步才被併進某個大 cluster，如果 K 夠大就成為 size=1 cluster。

## 為什麼 pid_1 / pid_9 / pid_20 不 singleton

這三個 client 的 label marginal 相似（99%+ label_0），**B 方向互相一致** → cosine 高 → 被 AP 歸為同 cluster。

而 pid_16 是**唯一** label_1 佔優的大 client，沒有「同伴」可以合群。

## 對 method 設計的 implications

### Singleton 是 feature 不是 bug

按上面的機制，singleton = 「這個 client 的 task direction 跟其他人都不一樣」。在 personalized FL 的定位下：

- **保留 singleton**（per-client dedicated B）= 正確的 personalization
- **強制合併**（min_cluster_size=2, merge to nearest）= 把 pid_16 塞進一個 label_0 dominant cluster → cluster 平均的 B 會**正負抵消** → pid_16 的 accuracy 會**掉**

### 預期的 accuracy 代價（若強制合併）

pid_16 目前 accuracy 99.5%（`eval_metrics.tsv`）。若強制 merge 到 label_0 cluster（含 pid_1/9/20），預期：
- pid_16 的 local B ← average(pid_16_B, pid_1_B, pid_9_B, pid_20_B)
- 平均後 B 方向變得模糊（99.5% + 0.1% + 0.1% + 1.6% ≈ 25% label_1）
- pid_16 的 accuracy 預計從 99.5% 掉到 ~50–60%（因為它的 test set 也是 99%+ label_1，被拖到決策邊界中間）

待**實驗驗證**（FedALC-Agglo-LWC with `min_cluster_size=2` post-process）。

### Paper 立場

> "Singleton clusters in FL with Dirichlet-based label skew are not clustering artifacts but expected behavior when one client's label marginal is inverted relative to the population. Forcing such an outlier into the nearest cluster will cancel its task-specific direction and degrade its personalized accuracy. **Singleton personalization** — letting the outlier keep its own B — is therefore the correct design choice for this type of heterogeneity."

## 跟 Dirichlet α 的關係

α=0.5 已經讓 pid_16 變 inverse marginal (99.5% label_1)。α=0.3 下分布應更極端，可能產生**多個** inverse marginal client → 多個 singleton。待 `tools/dump_partition_stats.py --task qnli --alpha 0.3` 驗證。

SST-2 α=0.5 目前在 R25+ 才後期爆炸 singleton（與 AP late-round 不穩有關），R1-R24 無 singleton — 可能 SST-2 的 Dirichlet 切分下**沒有**inverse marginal client。待同一 tool 驗證。

## 資料檔案

- `logs/partition_stats/qnli_c30_a0.5_s42.json` — 完整 JSON
- `logs/partition_stats/qnli_c30_a0.5_s42.tsv` — flattened table（pandas / spreadsheet 可直接開）
- Source tool: `tools/dump_partition_stats.py`

## 下一步（open）

1. **跑完剩 3 個組合**（sst2 α=0.5, sst2 α=0.3, qnli α=0.3）→ 確認 SST-2 沒 inverse marginal、確認 α=0.3 下 inverse 是否更嚴重
2. **實測 min_cluster_size=2 force merge 的 accuracy 代價**（在 FedALC-Agglo-LWC 用 post-process 跑，比對 pid_16 accuracy）
3. **其他 α / task 下的 singleton audit**：用 `dump_partition_stats.py` 系統性查出哪些 (task, alpha, pid) 組合會產生 inverse marginal

## Related

- `ap_vs_lwc_clustering_deepdive.md` — AP vs LWC clustering 結構比較（含 QNLI singleton 持續性觀察）
- `../concepts/fedalc_methods_comparison.md` — FedALC family clustering feature / aggregation 差異
- `../concepts/evaluation_metrics.md` — client-side eval vs server-side eval 的選擇理由
