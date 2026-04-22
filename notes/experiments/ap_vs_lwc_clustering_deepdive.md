---
tags: [experiment, clustering, ap, lwc]
related:
  - notes/experiments/fedalc_vs_lwc_no_warmup_results.md
  - notes/experiments/all_methods_comparison.md
  - notes/plans/fedalc_lwc_design.md
---

# AP vs AP-LWC clustering 從 log 的直接比較

> 最後更新：2026-04-20
> Log 來源：
> - AP basic：`logs/20260406_203614_fedalc_a0.5/{sst2,qnli}_fedalc_a0.5/clustering.jsonl` (30 rounds)
> - AP-LWC (no warm-up)：`logs/20260415_063849_fedalc-lwc_a0.5/{sst2,qnli}_fedalc-lwc_a0.5/clustering.jsonl` (20 rounds)

## 本檔目的

回答「layer-wise 讓 cluster 變好但 accuracy 沒升多少」是不是事實。直接讀 `clustering.jsonl`，不依賴舊 notes 的數字。

## 1. Clustering 行為（α=0.5, SST-2 + QNLI）

### SST-2 — 幾乎相同

| R | AP_K | AP sizes | AP sil | LWC_K | LWC sizes | LWC sil | LWC phase |
|---|---|---|---|---|---|---|---|
| 1 | 5 | [3, 3, 5, 6, 13] | 0.053 | 5 | [3, 4, 5, 5, 13] | 0.057 | 1 |
| 2 | 5 | [3, 3, 5, 6, 13] | 0.424 | 5 | [3, 4, 5, 5, 13] | 0.448 | 1 |
| 3 | 5 | [3, 3, 5, 6, 13] | 0.575 | 5 | [3, 4, 5, 5, 13] | 0.584 | 1 |
| 4 | 5 | [3, 3, 5, 6, 13] | 0.657 | 5 | [3, 4, 5, 5, 13] | **0.691** ← freeze 觸發前最後一輪 |
| 5+ | ⋯ | ⋯ | ⋯ | 5 | [3, 4, 5, 5, 13] | (frozen) | 2 |

**觀察**：K 都 5 群；sizes 幾乎一樣（`[3,3,5,6,13]` vs `[3,4,5,5,13]`，只差 1 個 client 在兩小群間挪位），silhouette LWC 略高 ~0.03。

### QNLI — 差很多

| R | AP_K | AP sizes | AP sil | LWC_K | LWC sizes | LWC sil | LWC phase |
|---|---|---|---|---|---|---|---|
| 1 | **4** | [1, 5, 6, **18**] | 0.134 | **7** | [1, 2, 3, 4, 5, 6, **9**] | 0.044 | 1 |
| 2 | 4 | [1, 5, 6, 18] | 0.338 | 7 | [1, 2, 3, 4, 5, 6, 9] | 0.414 | 1 |
| 3 | 4 | [1, 5, 6, 18] | 0.464 | 7 | [1, 2, 3, 4, 5, 6, 9] | 0.583 | 1 |
| 4 | 4 | [1, 5, 6, 18] | 0.581 | 7 | [1, 2, 3, 4, 5, 6, 9] | **0.675** ← freeze |
| 5+ | ⋯ | ⋯ | ⋯ | 7 | [1, 2, 3, 4, 5, 6, 9] | (frozen) | 2 |

**觀察**：
- AP 分 4 群，其中 **一個 18 人 mega-cluster**（60% 的 client 擠一群）
- LWC 分 **7 群**，粒度細很多（最大才 9 人）
- LWC silhouette 從 R2 就甩開 AP，R4 已經 0.68（AP 要等 R7 才到）

**解讀**：Layer-selected top-10 feature（~3.5K 維，ffn_inter 層主導）比 full 144 層 flatten（~50K 維）更容易讓 AP 發現細粒度結構。QNLI 上這個效應特別強 — 可能因為 QNLI 是判斷 entailment，需要比 SST-2（情緒極性）更複雜的 feature，不同 client 的 adaptation 在不同層上的分佈更有差異，full B 混在一起反而被雜訊蓋掉。

## 2. AP basic 後期震盪（LWC 沒有）

AP basic R20-R30 node explosion：

**SST-2**：R1-R21 穩定 5 群 → R22 掉成 4 群 → R25 暴增到 **19 群**（16 個孤立 single-client cluster + 3 大群）。sil 從 0.998 崩到 0.467。

**QNLI**：R1-R21 穩定 4 群 → R22 暴增到 **21 群** → R23 回到 4 群 → R24-30 震盪於 {3, 6} 群，sil 在 0.36-0.87 之間來回。

**LWC**：R4 silhouette > 0.675（freeze threshold 0.8 沒真正達到，但 cluster 連續 3 輪不變觸發 stability-based freeze），之後 16 輪完全凍結，無震盪。

→ **LWC 的 freeze 是 single-task 下 clearest 的貢獻**（不是 layer selection 本身）。

## 3. Accuracy 對照（client-side eval, `all_methods_comparison.md`）

| Metric | α=0.5 SST-2 | α=0.5 QNLI |
|---|---|---|
| FedALC-AP unweighted best | 93.33 @R25 | 92.55 @R27 |
| FedALC-AP-LWC unweighted best | 93.32 @R13 | **92.69 @R9** |
| Δ | −0.01% | +0.14% |
| Δ@time-to-best | −12 rounds | **−18 rounds** |

- Best accuracy 幾乎一樣（< 0.2%）
- **LWC 提早很多達到頂峰**（QNLI 早 18 輪），對 communication-constrained 場景有實質意義
- 但如果指標只看 "final best unweighted"，layer-wise 帶來的細粒度 cluster 沒體現

## 4. 為什麼 cluster 變細但 accuracy 不變

### 可能原因 A：aggregation 抵消 clustering 的好處
兩個 strategy **aggregation 都是用全 144 層 B**（LWC 只用 top-K 做 clustering feature，不影響聚合）。cluster 分細了 → 同群 client 數變少 → 每個 client 的 B 受到的「合作平均」訊號變弱 → variance reduction 不如預期。QNLI 上 LWC 的 9 人最大群 vs AP 的 18 人群，variance reduction 基本上腰斬。

### 可能原因 B：Single-task 下 cluster 粒度飽和
Binary label 任務的 irreducible noise 就那樣，20 人分成 1 群 vs 5 人分成 4 群，personalized accuracy 的 upper bound 沒差太多。Multi-task 下這個原因會消失（不同 task 的 loss landscape 真的不同，細粒度 cluster 能抓到）。

### 可能原因 C：R1 LWC silhouette 較低的代價
QNLI R1 LWC sil=0.044 < AP sil=0.134。R1 分錯的代價在 cluster aggregation 的自我強化下會累積（初始分群固定後很難跳群，從 clustering.jsonl 看 LWC freeze 前後 7 群完全沒動）。

### 可能原因 D：Client-side eval 本身的平坦化
`eval_metrics.tsv` 是每個 client 用自己的 personalized model 測 local split。不同 clustering → 不同聚合 B → 但 local split 的難度大致固定（同樣的 label ratio），所以 accuracy 差異只反映 B 的「邊際品質」而不是 cluster structure。

## 5. 含義

1. **"Layer-wise cluster 變更好" 在 QNLI α=0.5 有 log-level 證據**（K 從 4 變 7，silhouette +0.09）
2. **但 accuracy 面沒顯現**，因為 (a) 聚合仍用全 B、(b) single-task variance 已接近飽和、(c) client-side eval 只看個人 split
3. **LWC 在 single-task 的真實價值是 clustering 穩定性**（R4 freeze vs AP 後期震盪），不是 accuracy
4. **留給 multi-task**：不同 task 會把 cluster structure 推開，層級粒度差異真的會反映在 accuracy

## 6. 待驗證

- α=0.3 下 AP vs LWC 的 cluster structure 差異會更大嗎？（LWC α=0.3 沒跑，這是 comparison table 的缺格）
- `fit_metrics.tsv` 有 `partition_id` 但 `clustering.jsonl` 用 cid（每 run UUID 不同）→ 現在沒辦法跨 run 直接比對哪些 partition 分到同群。要做需要在 `client_app.py` log `cid ↔ partition_id` 對應（1 行改動）
- 觀察 LWC 的 7 個 QNLI cluster 對應哪些 partition 的 label distribution → 細粒度 cluster 是抓到什麼 signal？
