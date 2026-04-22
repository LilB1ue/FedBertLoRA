---
tags: [concepts, fedalc]
related: [fedalc_naming_convention.md, ../papers/FedALC-LoRA.md, ../papers/task_vector_connection.md]
---

# FedALC-AP / AP-LWC / AP-Multi 比較：相同點與差異點

> 最後更新：2026-04-21
> 目的：一眼看出三個 strategy 哪裡完全一致、哪裡差在 clustering 的細節。Code-grounded（引用到行）。
>
> 跟 `fedalc_naming_convention.md` 的分工：那份寫命名/定位/ablation 設計；這份聚焦實作行為的同與異。

## TL;DR

三者**核心結構一致**（A global / B per-cluster / Others local），差別全部集中在「**誰進去 clustering、什麼時候 clustering、clustering 前做什麼降噪**」。

---

## 一、相同點（三者完全一致）

### 1. 參數分離規則（name-based，同一份邏輯複製三次）

| key pattern | 歸類 |
|---|---|
| `"lora_A" in key` | A |
| `"lora_B" in key` | B |
| 其餘（classifier / score head） | Others |

實作：`_separate_a_b_others()` 在三個 file 文字完全相同
（`fedalc_ap_strategy.py:81-102` / `fedalc_ap_lwc_strategy.py:110-122` / `fedalc_ap_multi_strategy.py:267-276`）。

### 2. 三種角色的聚合原則

| 角色 | 處理方式 | 備註 |
|---|---|---|
| **A** | Global FedAvg（weighted by `num_examples`） | 三者 `aggregate_fit()` 第一步無條件做 |
| **B** | Per-cluster weighted average（cluster 內 FedAvg） | **差異只在「cluster 怎麼產生」** |
| **Others** | Per-client local（存 `self.client_others[cid]`，不跨 client 聚合） | 三者都另外算一個 `global_others = weighted_avg(...)` 供 server-side `evaluate_fn` fallback，但不回傳給 client |

### 3. Clustering algorithm

三者都用 **Affinity Propagation (AP)** on cosine similarity，參數相同：

```python
AffinityPropagation(
    affinity="precomputed",
    damping=0.5,        # ap_damping
    max_iter=100,       # ap_max_iter
    random_state=42,
)
similarity = cosine_similarity(feature_matrix)
```

AP 自動決定 K（damping 影響 K；本專案固定 0.5）。三者都跟著 `random_state=42`，所以**相同 feature matrix 會產生完全相同的 cluster assignment**。

### 4. Clustering log 輸出

三者都 append 到 `{log_dir}/clustering.jsonl`，每 round 一筆：
`{round, n_clusters, silhouette_score, clusters: {cluster_id: [pid,...]}}`。

### 5. 聚合後的 per-client 分派

`configure_fit()` 在下一 round 發參數時三者都做相同事：**global A + 自己 cluster 的 B + 自己的 others**，送回對應 client。

---

## 二、差異點

### 表 A — 架構維度

| 維度 | FedALC-AP | FedALC-AP-LWC | FedALC-AP-Multi |
|---|---|---|---|
| **Phase 設計** | 無 phase（每輪行為一致） | Phase 1（選層 + clustering）→ Phase 2（freeze） | Phase 0（warm-up）→ 1（clustering）→ 2（freeze） |
| **Clustering 起始 round** | R1 | R1 | Hopkins trigger 後（或 `warmup_max_rounds=10` fallback 後） |
| **Clustering 的 feature 向量** | Flatten(**full B**) | Flatten(**top-K layer 的 current B**) | Flatten(**top-K layer 的 cumulative ΔB**) |
| **Layer selection** | 無 | Metric B = $\text{dissim}\times\text{norm}$ on current B | Metric B on **cumulative ΔB**（default）或 current B（via `layer_score_feature`） |
| **Layer reselect 頻率** | — | `layer_reselect_every=0`（一次性） | `=1`（每 round），trigger 時 freeze 一份 snapshot 給下游 |
| **Warm-up trigger** | — | — | Hopkins $H>0.75$ on top-K ΔB，或 round ≥ 10 fallback |
| **Freeze trigger** | 無 freeze | silhouette > 0.9 或 cluster 連續 N 輪不變 | silhouette > 0.9 或 連續 `freeze_stable_rounds=3` 輪不變 |
| **ΔB 追蹤** | 無 | 無 | Per-client running cumulative $B_t - B_{t=0}$（`client_delta_b_cumulative`） |
| **Warm-up 期間 B 行為** | — | — | **Per-client local**（FedSA-style），`client_b_matrices[cid] = b_self` |

### 表 B — 最關鍵一列：「B 用什麼做 clustering」

| 方法 | Feature matrix 的列向量 | 典型維度 D |
|---|---|---|
| AP | `concat([B_l.flatten() for l in all_layers])` | ~20K–50K |
| AP-LWC | `concat([B_l.flatten() for l in top_K_layers])` | `K × d`（default K=10 → ~10K） |
| AP-Multi | `concat([ΔB_l_cumulative.flatten() for l in frozen_top_K])` | 同上 K × d |

> **為何 Multi 用 cumulative ΔB 而非 single-round B？**
> Single B 含 initial noise + 單輪 gradient 抖動；cumulative $\bar{\Delta B} = \frac{1}{t}\sum_\tau (B_\tau - B_0)$ 收斂到該 client 的「task vector」方向，clustering 信號更穩定。理論連結見 `../papers/task_vector_connection.md`。
>
> **為何需要 layer selection？**
> Hopkins 在 D > 50 後因 curse of dimensionality 失效（所有 pairwise distance 趨近等長 → $H\approx0.5$），且 `u_dist**d` 在 D≈50K 會數值溢位。Top-K 層降到 ~10K 以下 Hopkins 才能真的 trigger。

### 表 C — Config 參數 matrix

| 參數 | AP | AP-LWC | AP-Multi | default |
|---|:---:|:---:|:---:|---|
| `ap_damping` / `ap_max_iter` | ✓ | ✓ | ✓ | 0.5 / 100 |
| `warmup_sil_threshold` | | ✓ | | 0.5 |
| `hopkins_threshold` | | | ✓ | 0.75 |
| `warmup_max_rounds` | | | ✓ | 10 |
| `freeze_sil_threshold` | | ✓ | ✓ | 0.9 |
| `freeze_stable_rounds` | | | ✓ | 3 |
| `layer_selection_k` | | ✓ | ✓ | 10 |
| `layer_reselect_every` | | ✓ | ✓ | 0 (LWC) / 1 (Multi) |
| `layer_score_feature` | | | ✓ | `cumulative_delta_b` |

### 表 D — `aggregate_fit()` 流程差異（pseudocode）

**三者共同前置**

```python
for (client, fit_res) in results:
    a, b, others = _separate_a_b_others(fit_res.params)
    append to client_a_list / client_b_list / client_other_list

global_a = weighted_avg(client_a_list, weights)          # always global
client_others[cid] = others                              # always local
global_others = weighted_avg(client_other_list, weights) # only for eval_fn
```

**FedALC-AP**

```python
cluster_b, metrics = _cluster_b_matrices(client_b_list)
#   feature = concat([B.flatten() for B in full B layers])
#   AP on cosine sim → labels
#   per-cluster weighted_avg(B) → client_b_matrices[cid] = cluster_agg_b
```

**FedALC-AP-LWC**

```python
if phase == 1:
    if need_reselect:                           # one-shot 或每 N round
        selected = top_K(metric_B(client_b_list))
    feature = concat([B[l].flatten() for l in selected])
    clustered_b = AP(cosine(feature)) → per-cluster avg
    if silhouette > 0.9 or stable N rounds:
        phase = 2
        frozen_cluster_groups = current_groups
elif phase == 2:
    apply frozen_cluster_groups → per-cluster avg(B)
```

**FedALC-AP-Multi**

```python
update client_delta_b_cumulative[cid]          # running mean of (B_t - B_0)

if phase == 0:  # warm-up
    client_b_matrices[cid] = b_self            # B stays local (FedSA mode)
    if layer_reselect_due:
        selected = top_K(metric(cumulative_ΔB or current_B))
    H = hopkins(ΔB_cumulative[top_K=selected])
    if H > 0.75 or round >= warmup_max_rounds:
        phase = 1
        frozen_layer_indices = selected        # snapshot for downstream

elif phase == 1:  # clustering
    feature = concat([ΔB_cumul[l].flatten() for l in frozen_layer_indices])
    clustered_b = AP(cosine(feature)) → per-cluster avg
    if silhouette > 0.9 or stable 3 rounds:
        phase = 2; frozen_cluster_groups = current_groups

elif phase == 2:  # frozen
    apply frozen_cluster_groups → per-cluster avg(B)
```

---

## 三、為什麼 AP = baseline / LWC = ablation / Multi = 主方法

| 方法 | 弱點（想改進的地方） | 改進動作 |
|---|---|---|
| **AP** | full B (D~50K) cosine 不穩、R1 的 B 含 init noise、無 freeze（後期抖） | — 這是 baseline |
| **AP-LWC** | R1 就 cluster（沒 warm-up）、用 current B（含 noise） | 加 layer selection 降維 + freeze 穩定後期 |
| **AP-Multi** | — | 再加 cumulative ΔB（信號）+ Hopkins adaptive warm-up（避開「沒 signal 就硬 cluster」） |

LWC 比 AP 好的部分 = `layer selection + freeze` 的貢獻；Multi 比 LWC 好的部分 = `cumulative ΔB + Hopkins adaptive trigger` 的貢獻。這個拆分正是 paper ablation table 的軸。

---

## Reference

| 檔案 | 行數 | 內容 |
|---|---|---|
| `bert/fedalc_ap_strategy.py` | 46-80 / 141-239 / 241-320 | `__init__`, `_cluster_b_matrices`, `aggregate_fit` |
| `bert/fedalc_ap_lwc_strategy.py` | 51-106 / 160-206 / 210-322 / 324-481 | `__init__`, layer scoring, layer-selected AP, `aggregate_fit` (Phase 1/2) |
| `bert/fedalc_ap_multi_strategy.py` | 179-263 / 479-623 / 625-883 | `__init__`, `_cluster_on_delta_b`, `aggregate_fit` (Phase 0/1/2) |

相關 notes：
- `fedalc_naming_convention.md` — 命名 + ablation 設計藍圖
- `../papers/FedALC-LoRA.md` — 方法動機與對標
- `../papers/task_vector_connection.md` — cumulative ΔB 的理論基礎
