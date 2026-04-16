# 實驗 Findings 總結與下一步建議

> 最後更新：2026-04-16
> 設定：aiserver1, 30 clients, RoBERTa-large + LoRA r=8

## 核心數據總覽

### α=0.5（30 rounds，同機器同環境）

| Task | FedAvg | FedSA-LoRA | FedALC | FedALC-LWC |
|---|---|---|---|---|
| SST-2 | 0.9457 @R13 | 0.9520 @R17 | **0.9547 @R26** | 0.9522 @R20 |
| QNLI | 0.9190 @R13 | 0.9243 @R17 | **0.9385 @R29** | 0.9312 @R20 |

### α=0.3（30 rounds）

| Task | FedAvg | FedSA-LoRA | FedALC | FedALC-LWC |
|---|---|---|---|---|
| SST-2 | 0.9487 @R21 | 0.9546 @R17 | **0.9694 @R26** | 待跑 |
| QNLI | 0.9120 @R21 | 0.9188 @R23 | **0.9588 @R8** | 待跑 |

**一致排序**：FedALC > FedALC-LWC ≈ FedSA > FedAvg

## Key Findings

### Finding 1: FedALC 在 non-IID 越強時優勢越大

| Task | α | FedALC - FedSA |
|---|---|---|
| SST-2 | 0.5 | +0.3% |
| SST-2 | 0.3 | **+1.5%** |
| QNLI | 0.5 | +1.4% |
| QNLI | 0.3 | **+4.0%** |

**Bias-Variance 解釋**：
- α=0.5：client 間差異中等，FedSA 的「B 留 local」variance 還能接受
- α=0.3：client 間差異大，FedSA 的 B 缺乏合作，variance 失控
- FedALC 的 cluster aggregation 在 α=0.3 下提供了合適粒度的 variance reduction

### Finding 2: Clustering 反映的是 adaptation pattern，不是單純 label distribution

- Binary task (2 labels)，AP 分出 4-5 cluster（α=0.5）或 3-4 cluster（α=0.3）
- Cluster 數 > label 數 → B 矩陣抓到更細緻的 adaptation 差異（不同 client 的 positive examples 可能學到不同 feature）
- Silhouette 高（0.99+）但這是 cluster aggregation 的自我強化效應，不一定代表分群「正確」

### Finding 3: Cluster assignment 穩定性

穩定期內完全沒有 client 跳群：

| 實驗 | 穩定輪數 | 期間換群 |
|---|---|---|
| SST-2 α=0.5 | R1-R21 | 0 |
| QNLI α=0.5 | R1-R21 | 0 |
| SST-2 α=0.3 | R1-R17 | 0 |
| QNLI α=0.3 | R1-R25 | 0 |

**原因**：cluster aggregation 的自我強化——R1 分群後同群 B 被平均，下一輪從同點出發，自然還是同群。

**副作用**：R1 的分群品質（silhouette 0.05）直接決定後續 30 輪的分群結果。但從 accuracy 看即使 R1「亂分」效果還是好，因為 variance reduction 的收益超過 bias 的損失。

### Finding 4: AP 後期不穩定

FedALC 在 R22+ 開始出現 AP 震盪：
- 同群 cosine 從 0.99 → 0.9999（自我強化到極端）
- Similarity matrix 變成 bimodal（要麼接近 1，要麼接近 0）
- AP 的 responsibility/availability message passing 在這種 degenerate 情況下不穩定
- Cluster 數暴增到 19 群 → 退化為 FedSA

**不是跨群 B 趨同**（實測 R20-R22 跨群 cosine 穩定在 0-0.3）——是 **AP 演算法本身在 extreme similarity 下的弱點**。

### Finding 5: FedALC-LWC 相對於 FedSA / FedAvg 的優勢

**跟 baseline（FedSA, FedAvg）比較**，LWC 是有明確優勢的：

| | SST-2 Best | QNLI Best | LWC vs FedAvg | LWC vs FedSA |
|---|---|---|---|---|
| FedAvg | 0.9457 | 0.9190 | — | — |
| FedSA-LoRA | 0.9520 | 0.9243 | — | — |
| **FedALC-LWC** | **0.9522** | **0.9312** | **+0.65% / +1.22%** | **+0.02% / +0.69%** |
| FedALC | 0.9547 | 0.9385 | | |

**收斂速度比較**（SST-2 α=0.5）：

| Round | FedAvg | FedSA | FedALC | FedALC-LWC |
|---|---|---|---|---|
| R1 | 0.8225 | 0.8413 | 0.8455 | 0.8244 |
| R3 | 0.9328 | 0.9143 | **0.9415** | 0.9412 |
| R5 | 0.9396 | 0.9441 | 0.9490 | **0.9498** |
| R10 | 0.9442 | 0.9489 | **0.9527** | 0.9498 |
| R15 | 0.9444 | 0.9516 | 0.9503 | **0.9505** |
| Best | 0.9457 | 0.9520 | **0.9547** | 0.9522 |

**收斂速度比較**（QNLI α=0.5）：

| Round | FedAvg | FedSA | FedALC | FedALC-LWC |
|---|---|---|---|---|
| R1 | 0.4999 | **0.7490** | 0.7220 | 0.6853 |
| R3 | **0.9014** | 0.8280 | 0.7531 | 0.8476 |
| R5 | 0.9142 | 0.8960 | **0.9273** | 0.9098 |
| R10 | 0.9167 | 0.9182 | **0.9316** | 0.9248 |
| R15 | 0.9172 | 0.9236 | **0.9305** | 0.9286 |
| Best | 0.9190 | 0.9243 | **0.9385** | 0.9312 |

**LWC 相對於 FedSA / FedAvg 的明確優勢：**

1. **Best accuracy 同時超越 FedSA 和 FedAvg**
   - SST-2: +0.02% vs FedSA, +0.65% vs FedAvg
   - QNLI: +0.69% vs FedSA, +1.22% vs FedAvg

2. **R5-R15 的中期收斂比 FedSA 快**
   - SST-2 R5: LWC 0.9498 > FedSA 0.9441
   - QNLI R10: LWC 0.9248 > FedSA 0.9182

3. **Clustering 30 輪全穩定**（freeze 機制），沒有 FedALC 後期的 AP 震盪問題

4. **可解釋性**：layer selection 告訴你 ffn_inter 層判別力最高

**LWC 相對於 FedALC 的劣勢：**

- Best accuracy 稍低（-0.25% SST-2, -0.73% QNLI）
- R1 收斂較慢（layer selection 在 B≈0 時選的層不穩定）

**LWC 真正價值可能在 multi-task**：single task 下 layer-wise 貢獻有限；multi-task 下不同 task 對不同層敏感，LWC 的 adaptive selection 才能彰顯。

### Finding 6: Layer selection 的結果

Metric B (dissim × norm) 在 single task 下穩定選出 **ffn_inter 層**（FFN intermediate dense）：

- R1 時 Top-10 幾乎全是 ffn_inter
- R3+ 後選擇穩定
- Attention 層（query/key/value/attn_dense）判別力低

**為什麼是 ffn_inter**：
- Attention 學「怎麼關注 input」→ 跨 client 差異小（英文語法結構一致）
- FFN intermediate 學「怎麼轉換 feature representation」→ 直接關聯 task decision → 跨 client 差異大

### Finding 7: A/B 矩陣角色驗證

實測 R1-R3 的 cosine similarity：
- **A 跨 client**：mean=0.95, std=0.04 → 高度一致（general knowledge）
- **B 跨 client**：mean=0.02-0.16, std=0.07-0.26 → 差異大（client-specific）

驗證 FedSA-LoRA 的假設，支持 FedALC 的「A global, B cluster」設計。

### Finding 8: Evaluation Metric 的限制

目前 `server_eval.tsv` 對 FedALC 是誤導的：
- 用 `global_A + avg(client_B)` 評估
- Avg B 是跨 cluster 平均，不是任何 client 實際用的 model
- 這不是 FedAvg（訓練動態不同），但也不是真實 FedALC inference 場景

**主指標應用 `evaluate/accuracy`**（personalized）——跟 HiLoRA/FedLEASE 一致。

## 下一步建議（按優先度）

### 優先度 1：補齊 Tier 1 baseline

**FFA-LoRA**（跑中）
- Code 已支援，bug 已修（client 端 freeze A）
- 腳本：`run_ffa_all.sh`，跑 SST-2/QNLI × α=0.3/0.5
- 預期：不如 FedSA（A freeze 犧牲 flexibility）

### 優先度 2：修正 global evaluation

目前 FedALC 的 server_eval 是誤導的。改成：
- **方式**：每個 client 用自己的 personalized model 在 GLUE validation split 上測 → 30 個 client 的 weighted avg
- **實作**：改 `configure_evaluate` 送 global validation set，或 server 端從 checkpoint 載入 personalized model
- **意義**：真實反映 personalization 對 global 能力的影響

### 優先度 3：FedLEASE 簡化版

最重要的 Tier 2 baseline（NeurIPS 2025），沒有 source code 需自己實作。

**簡化版**（無 MoE）：
- Agglomerative clustering + silhouette 選 K
- Per-cluster A+B 聚合（跟 FedALC 差在 A 也 per-cluster）
- 不做 MoE router（client 只用自己 cluster 的 expert）

**比較點**：FedALC 的 A global vs FedLEASE 的 A per-cluster，在 single task 下哪個更好？

### 優先度 4：Multi-task 實驗

FedALC-LWC 的真正價值可能在 multi-task。設定：
- Tasks：SST-2 + QNLI + MNLI + QQP
- 每 task 分配若干 clients（例如 4 tasks × 8 clients = 32 clients）
- 比較：FedAvg / FedSA / FedALC / FedALC-LWC / FedLEASE

**預期發現**：
- Cluster 數會自然等於 task 數（FedLEASE 觀察）
- Layer selection 會更有效（不同 task 對不同層敏感）
- FedALC-LWC 可能終於勝出 FedALC

### 優先度 5：分析型實驗

**A. Clustering 演算法 ablation**（AP vs Agglomerative）
- 解決 AP 震盪問題
- 跟 FedLEASE 對齊

**B. Clustering feature ablation**
- Full B vs ffn_inter only vs Top-K Metric B vs Per-layer equal-weight (FedLEASE 風格)
- 已有初步分析（silhouette 差異 0.01-0.07）

**C. α 敏感度分析**
- 已有 α=0.3, 0.5
- 可補 α=0.1（需解決 DirichletPartitioner 限制）和 α=1.0（接近 IID）

### 優先度 6：Paper writing

Phase 1+2 的結果已足夠支撐一篇 paper：
- Contribution 1: B 的 clustering aggregation > B local（FedSA）
- Contribution 2: α 越小收益越大（bias-variance 論述）
- Contribution 3: A/B 角色分離的實證驗證

Layer selection (LWC) 可以當 future work 或 ablation，不需要扛起整個 paper 的 novelty。

## Paper Story 建議

### 如果故事是 "FedALC"

**Contribution**：B 矩陣的 cluster aggregation，介於 FedAvg（全聚合）和 FedSA（不聚合）之間的中間粒度。

**Baselines**：FedAvg, FedSA, FFA-LoRA, FedLEASE（簡化版）

**Experiments**：
1. Main results：SST-2/QNLI/MNLI/QQP × α=0.3/0.5
2. α 敏感度
3. Clustering analysis（cluster membership stability, silhouette, vs label ratio）
4. A/B 矩陣角色驗證

### 如果故事是 "FedALC-LWC"

Multi-task 必要。Single task 沒有說服力（accuracy 反而略低於 FedALC）。
