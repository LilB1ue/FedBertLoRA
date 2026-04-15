# FedALC-LoRA 下一步行動計劃

> 最後更新：2026-04-15

## 現狀

- Phase 1 完成：FedALC > FedSA > FedAvg（SST-2, QNLI, α=0.5, 30 rounds）
- Phase 2（α=0.3）完成：FedALC 優勢更大（+1.5%/+4.0% vs FedSA）
- Clustering 穩定但後期 AP 震盪（R22+ cluster 數暴增）
- A/B 角色驗證完成（A cosine ≈ 0.95, B cosine ≈ 0.02-0.16）
- Layer-wise 在單任務 α=0.5 下沒有改善空間（silhouette 已 0.99）
- Cluster 數跟 label 數不直接相關（binary task 分 4-5 群），反映的是 adaptation pattern 而非 label distribution

---

## Step 1：修復 AP 震盪問題

**優先度：高（影響所有後續實驗）**

在 `fedalc_strategy.py` 加入 fallback 機制：
- 如果這輪 cluster 數相比上輪變化超過 2 倍，沿用上輪分群
- 或如果 silhouette 驟降超過 0.2，沿用上輪分群
- Log 哪些輪觸發了 fallback

## Step 2：跑 α=0.1 和 α=1.0 實驗

**優先度：高（Phase 2 核心）**

### 實驗矩陣

| α | FedAvg | FedSA | FedALC |
|---|---|---|---|
| 0.3 | 要跑 | 要跑 | 跑中 |
| 0.5 | ✅ 有 (20r) | ✅ 有 (20r) | ✅ 有 (30r) |
| 1.0 | 要跑 | 要跑 | 要跑 |

- 任務：SST-2 + QNLI
- Rounds：30
- **α=0.1 無法使用**：DirichletPartitioner 在 30 clients + binary task 下分割失敗（部分 client 分到 0 筆），改用 α=0.3（跟 HiLoRA 的 GL-Dir(0.3) 設定一致）
- **注意**：FedAvg 和 FedSA 的 α=0.5 只有 20 rounds，可能需要重跑 30 rounds 才能公平比較

### 腳本
- `run_fedalc_alpha03.sh`：FedALC α=0.3，SST-2 + QNLI，30 rounds

### α=0.3 結果（2026-04-15）

| Task | FedAvg | FedSA | FedALC | FedALC vs FedSA |
|---|---|---|---|---|
| SST-2 | 0.9487 @R21 | 0.9546 @R17 | **0.9694 @R26** | **+1.5%** |
| QNLI | 0.9120 @R21 | 0.9188 @R23 | **0.9588 @R8** | **+4.0%** |

**結論：α 越小，FedALC 的優勢越大。** α=0.3 的提升遠大於 α=0.5（+0.3%/+1.4%）。

完整比較：

| Task | α | FedAvg | FedSA | FedALC | FedALC vs FedSA |
|---|---|---|---|---|---|
| SST-2 | 0.5 | 0.9457 | 0.9520 | 0.9547 | +0.3% |
| SST-2 | 0.3 | 0.9487 | 0.9546 | **0.9694** | **+1.5%** |
| QNLI | 0.5 | 0.9190 | 0.9243 | 0.9385 | +1.4% |
| QNLI | 0.3 | 0.9120 | 0.9188 | **0.9588** | **+4.0%** |

### α=0.3 重點觀察項目

**Accuracy**：
- FedALC vs FedSA 的差距是正還是負？
- 收斂速度是否比 α=0.5 慢？

**Clustering 穩定性**：
- α=0.5 時 R1-R21 完全不變。α=0.3 是否也穩定？
- Client 差異更大 → B 差異更大 → 可能有 client 跳群現象
- 如果不穩定 → fallback 機制的 motivation

**Silhouette score**：
- α=0.5 達到 0.99。α=0.3 如果也很高 → layer-wise 單任務沒改善空間
- α=0.3 如果低（< 0.5）→ **layer-wise 有改善空間，可接 Step 4A**
- 這是決定 layer-wise 走單任務還是 multi-task 的關鍵數據

**Cluster 數量**：
- α=0.3 比 α=0.5 更 non-IID → AP 可能分出不同數量的 cluster
- 跟 α=0.5 的 5 群（SST-2）/ 4 群（QNLI）比較

## Step 3：分析 α=0.1 的 clustering 行為

**優先度：取決於 Step 2 結果**

- 觀察 clustering 穩定性（cluster 數是否固定、是否有 client 跳群）
- 觀察 silhouette score 趨勢
- Per-layer discriminability score 分析（全部 96 個 B module）
- 比較 full B clustering 的 silhouette vs per-layer top-K 的 silhouette

### 這一步決定 layer-wise 的方向

```
α=0.1 full B clustering 的 silhouette 低？
  ├── 是 → Step 4A：單任務 layer-wise
  └── 否 → Step 4B：multi-task layer-wise
```

## 開放問題（2026-04-15 討論）

### Q1: 需要跑到 50 rounds 嗎？

不需要。SST-2 best @R26，QNLI best @R8，兩者在 R10-15 後都已收斂。30 rounds 足夠。
如果要寫 paper，只需確認曲線已 plateau 就好。50 rounds 只會浪費計算和磁碟空間。

### Q2: Layer-wise 在 single task 下有可能更細緻分群嗎？

目前觀察：AP 在 binary task 上分出 4-5 群，cluster 數 > label 數。代表 B 矩陣抓到的不只是 label distribution，而是更細緻的 adaptation pattern（模型「怎麼學」的差異）。

Layer-wise 在 single task 下的可能性：
- 如果不同層的 B 反映不同維度的 adaptation pattern → layer-wise 選擇可以讓某些維度的信號更突出 → 更細緻的分群
- 但目前 full B clustering 的 silhouette 已經很高（0.98+），改善空間有限
- **需要先驗證**：算 per-layer discriminability score，看不同層的分數是否有顯著差異。如果所有層分數差不多 → layer-wise 沒用。如果某些層特別高 → layer-wise 有潛力

**建議**：先跑 per-layer discriminability 分析（不需要改 clustering，只需要從現有 checkpoint 算），再決定要不要實作 layer-wise clustering。

### Q3: Centralized 比較先不放？

對。Centralized 是不同的訓練方式（全量資料、no communication），不是 FL 方法的 baseline。FL 方法之間的比較（FedAvg vs FedSA vs FedALC）更重要。Centralized 可以在 paper 的 table 裡報，但 plot 裡不需要。

### Q4: Clustering 穩定性分析與 AP 頻率

**穩定期內完全沒有 client 換 cluster**（從 clustering.jsonl 驗證）：

| 實驗 | 穩定期 | 穩定輪數 | 期間 client 換群 |
|---|---|---|---|
| SST-2 α=0.5 | R1-R21 | 21 輪 | 0 |
| QNLI α=0.5 | R1-R21 | 21 輪 | 0 |
| SST-2 α=0.3 | R1-R17 | 17 輪 | 0 |
| QNLI α=0.3 | R1-R25 | 25 輪 | 0 |

驗證方式：每輪的 cluster 成員轉成 `set of frozensets`（不管 AP 的 label/exemplar，只看「誰跟誰同群」），跟上一輪比較。

**震盪開始後**才有大量 client 移動（一次 11-29 個），是 AP degenerate 造成的，不是有意義的重新分群。

**AP 頻率建議**：
- ❌ 只在 R1 做一次 → R1 的 B≈0（silhouette 0.05-0.14），分群品質差，雖然後來自我強化但不保證所有設定都行
- ✅ **前幾輪每輪做 AP → silhouette > 0.8 後 freeze clustering** → 之後只監控 silhouette 不重新分群
- 如果 silhouette 突然掉 → 觸發重新 clustering
- 這樣既避免 R1 亂分的風險，也避免後期 AP 震盪

---

## 接下來的 Steps（更新）

### Step 4: Per-layer Discriminability 分析

**不需要改 code，只需分析現有 checkpoint**

- 從 client_checkpoints 載入每個 client 的 B 矩陣
- 逐層計算 discriminability score（Metric B: dissimilarity × norm）
- 比較不同層的分數差異
- 分別做 α=0.3 和 α=0.5

**如果不同層差異大 → Step 5A（layer-wise clustering）**
**如果差異小 → Step 5B（multi-task 或直接寫 paper）**

### Step 5A：Layer-wise Clustering（如果 Step 4 有差異）

- 實作 per-layer weighted/selected clustering
- 比較 full B vs per-layer equal-weight（FedLEASE） vs adaptive weight/selection
- 報告 silhouette + accuracy

### Step 5B：Multi-task 或 Paper

**如果 single task layer-wise 沒有改善空間：**
- Multi-task GLUE：SST-2 + QNLI + MNLI 混合 client
- 或直接用現有結果寫 paper（Phase 1 + Phase 2 已有足夠 contribution）

### Step 6：整理結果 + 寫 Paper

- 所有實驗結果整理成表格和圖
- 跟 FedSA-LoRA / FedAvg / FedADC / HiLoRA / FedLEASE 做比較定位
