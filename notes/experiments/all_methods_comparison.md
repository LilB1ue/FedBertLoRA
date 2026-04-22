# Method Comparison — Client-side eval

> 最後更新：2026-04-20
> Setup：RoBERTa-large, r=8, 30 clients, **per-client test split (Flower evaluate protocol)**
> Source：`logs/<batch_dir>/<task>_<mode>_a<alpha>/eval_metrics.tsv`
> 分 α=0.5 / α=0.3 兩組，不要跨 α 混比

## 為什麼用 client-side eval（而非 server-side）

專案目標是訓練 **personalized models**（每個 client 有自己的 B / others）。
Server-side `server_eval.tsv` 取 parameter average 做 eval，對 personalized method 無意義（例如 FedSA 的 B 本地保留 → average B 是 noise，會呈現假性「崩潰」）。

**Client-side eval（`eval_metrics.tsv`）**：每個 client 在自己 local test split 用最新 model 跑 eval
- 對 FedAvg / FFA：global model on client test
- 對 FedSA：global A + local B on client test
- 對 FedALC-AP-*：global A + cluster B + local others on client test

這就是每個 method **部署後 end-user 真正體驗的 accuracy**。

## 兩種聚合方式都報

| Metric | 算法 | 意義 |
|---|---|---|
| **Unweighted mean** | Σ accᵢ / N | 每個 client 平等，符合 personalized fairness |
| **Weighted mean** | Σ (accᵢ · nᵢ) / Σ nᵢ | 等同把所有 client test pool 起來算（== global pooled accuracy）；大 client 主導 |

兩者差距大 = 大 client 跟小 client 表現差很多（fairness 問題）。

## α=0.5（中等 non-IID）

### Best per-client accuracy（client-side）

**SST-2**

| Method | Unweighted Best | Round | Weighted Best | Round |
|---|---|---|---|---|
| **FedAvg** | **94.27** | 13 | 94.57 | 13 |
| FedSA | 93.72 | 17 | 95.20 | 17 |
| FedALC-AP | 93.33 | 25 | **95.47** | 26 |
| FedALC-AP-LWC | 93.32 | 13 | 95.22 | 20 |
| FFA | 94.06 | 13 | 94.30 | 17 |

> FFA α=0.5 SST-2 已重跑（`logs/20260420_120302_ffa_a0.5/sst2_ffa_a0.5/`），client-side eval 補齊。

**QNLI**

| Method | Unweighted Best | Round | Weighted Best | Round |
|---|---|---|---|---|
| **FedALC-AP-LWC** | **92.69** | 9 | 93.12 | 20 |
| FedALC-AP | 92.55 | 27 | **93.85** | 29 |
| FedAvg | 92.16 | 14 | 91.90 | 13 |
| FedSA | 89.94 | 17 | 92.43 | 17 |
| FFA | **gap — 僅 server_eval reconstructed** | — | — | — |

> FFA α=0.5 QNLI client-side 仍缺：`logs/20260416_064902_ffa_a0.5/qnli_ffa_a0.5/` 只有 `RECONSTRUCTED_FROM_WANDB.txt` + `server_eval.tsv`，沒 `eval_metrics.tsv`。需要重跑一次才能補齊 α=0.5 QNLI 比較。

### 圖表

- `plots/r30_c30/all_methods_unweighted_{sst2,qnli}_a0.5.png`
- `plots/r30_c30/all_methods_weighted_{sst2,qnli}_a0.5.png`
- `plots/r30_c30/all_methods_best_bar_{sst2,qnli}_a0.5.png`
- `plots/r30_c30/all_methods_perclient_{sst2,qnli}_a0.5.png`（per-client box）

## α=0.3（強度 non-IID）

### Best per-client accuracy（client-side）

**SST-2**

| Method | Unweighted Best | Round | Weighted Best | Round |
|---|---|---|---|---|
| **FedALC-AP** | **97.10** | 22 | **96.94** | 26 |
| FFA | 94.90 | 19 | 94.57 | 19 |
| FedAvg | 94.68 | 21 | 94.87 | 21 |
| FedSA | 92.97 | 20 | 95.46 | 17 |

**QNLI**

| Method | Unweighted Best | Round | Weighted Best | Round |
|---|---|---|---|---|
| **FedALC-AP** | **91.03** | 24 | **95.88** | 8 |
| FFA | 90.61 | 11 | 90.57 | 16 |
| FedAvg | 90.21 | 16 | 91.20 | 21 |
| FedSA | 87.10 | 16 | 91.88 | 23 |

### 圖表

- `plots/r30_c30/all_methods_unweighted_{sst2,qnli}_a0.3.png`
- `plots/r30_c30/all_methods_weighted_{sst2,qnli}_a0.3.png`
- `plots/r30_c30/all_methods_best_bar_{sst2,qnli}_a0.3.png`
- `plots/r30_c30/all_methods_perclient_{sst2,qnli}_a0.3.png`

## Key findings（client-side view）

### 🚨 Finding 1 — 「FedSA 崩潰」是 server-side artifact，client-side 正常

- **Server-side view（之前的假象）**：FedSA α=0.3 QNLI = 50.54%
- **Client-side view（正確）**：
  - α=0.5 QNLI: unweighted 89.94%, weighted 92.43%
  - α=0.3 QNLI: unweighted 87.10%, weighted 91.88%
- 差距合理（非崩潰），只是 FedSA 在 high non-IID QNLI 下 unweighted 較低
- **原因**：FedSA 保留 local B，server-side eval 取 B 平均是無意義的 parameter averaging；client-side eval 用 client 自己的 B 正常 work
- **Paper implication**：server-side eval 對 personalized 類 methods（包括 FedSA）不公平，**所有 main comparison 應該用 client-side**

### 🌟 Finding 2 — FedALC-AP 在 α=0.3 client-side 大勝

- SST-2 unweighted: **97.10%**（比 FFA 的 94.90% 高 2.20%）
- QNLI unweighted: **91.03%**（比 FFA 的 90.61% 高 0.42%）
- QNLI weighted: **95.88%**（比 FedSA 的 91.88% 高 4.00%）
- **α=0.3 下 personalized B 的優勢明顯浮現**
- Server-side view 完全看不出這個 — 之前以為 FedALC-AP basic 在 α=0.3 QNLI 只有 90.23%，實際上是 server-side 低估

### 🎯 Finding 3 — FedALC-AP-LWC 在 α=0.5 QNLI 提早達到最佳（R9）

- Unweighted 92.69% best at round 9（比其他 methods 都早）
- FedALC-AP 要到 R27 才達 92.55%
- 說明 LWC 的 silhouette warm-up + layer selection 確實加速收斂
- 但 final accuracy 差不多（FedALC-AP 最終 weighted 93.85% > LWC 93.12%）

### 📊 Finding 4 — Weighted vs Unweighted 差距揭露 fairness 問題

| α=0.3 QNLI | Unweighted | Weighted | 差距 |
|---|---|---|---|
| FedALC-AP | 91.03 | 95.88 | **+4.85** |
| FedSA | 87.10 | 91.88 | +4.78 |
| FedAvg | 90.21 | 91.20 | +0.99 |
| FFA | 90.61 | 90.57 | -0.04 |

- **FedALC-AP 跟 FedSA 的 weighted 比 unweighted 高 4.8%** → 大 client 拿高分、小 client 拿低分（personalization 有 bias toward data-rich clients）
- **FFA 兩者差 0.04%** → fairness 最好（B 全域，無 personalization bias）
- **FedAvg 差 0.99%** → 相對公平
- **Paper 賣點 trade-off**：FedALC-AP 總 accuracy 高但不太公平；FFA 公平但總 accuracy 低

### 🔧 Finding 5 — FFA α=0.5 client-side 部分遺失（SST-2 已補、QNLI 仍缺）

- FFA α=0.3 overwrite 了 α=0.5 的整個 log dir（fit_metrics + eval_metrics + checkpoints）
- wandb `output.log` 只有 `[Server]` 印出（server-side eval），沒 per-client 印出
- **SST-2**：已於 2026-04-20 重跑（`logs/20260420_120302_ffa_a0.5/sst2_ffa_a0.5/`），client-side 補齊（uw 94.06 / w 94.30）
- **QNLI**：仍需重跑一次，才能完成 α=0.5 完整比較

### 🔬 Finding 6 — α=0.3 unweighted 的絕對值比 α=0.5 還高（SST-2）

- SST-2 α=0.5 FedAvg unweighted 94.27
- SST-2 α=0.3 FedAvg unweighted 94.68 (+0.41%)
- SST-2 α=0.3 FedALC-AP unweighted 97.10（比 α=0.5 的 93.33% 高 3.77%）
- **反直覺但可解釋**：α=0.3 下每個 client 的資料更集中（例如只看 positive sentiment），learning 目標更簡單 → local model 訓很好、per-client accuracy 看起來高
- 這實際上是 **client 各自 overfit 到 local task distribution**，不等於 generalizability 好

## 對 FedALC-AP-Multi 實驗的 updated priority

1. **必跑**：FedALC-AP-Multi α=0.3 QNLI — 看能否從 FedALC-AP basic 的 91.03%(uw) / 95.88%(w) 再往上推
2. **必跑**：FedALC-AP-Multi α=0.5 QNLI — 跟 basic / LWC 對比 component 貢獻
3. **必須**：**重跑 FFA α=0.5**（SST-2 + QNLI）把 comparison 補齊
4. **可選**：跑 FedALC-AP-LWC α=0.3（之前沒跑，comparison 缺這格）

## 歷史修正紀錄

| 時間 | 修正事項 |
|---|---|
| 2026-04-20 之前 | 全用 server-side `server_eval.tsv`，結論有多項錯誤 |
| 2026-04-20 | 切換到 client-side `eval_metrics.tsv`，重寫 findings（這個文件） |
| 原 server-side findings | 標記廢棄：FedSA QNLI 50.54% 是 artifact、FedALC-AP α=0.3 QNLI 90.23% 低估、FedAvg 看似最強是因為 server-side view 偏好 global method |

## 工具

- `plot_method_comparison.py`（repo root）：新 client-side 版本
- `tools/scan_log_inventory.py`：掃 wandb 配對 log dir → alpha
- `tools/apply_log_rename.sh`：rename subdir 加 alpha tag
- `tools/apply_batch_dir_rename.sh`：rename 外層 timestamp dir 加 mode+alpha
- `tools/reconstruct_server_eval_from_wandb.py`：從 wandb output.log 重建 server_eval（僅 server-side）
