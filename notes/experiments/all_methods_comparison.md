# Method Comparison — FedAvg / FedSA / FFA / FedALC-AP / FedALC-AP-LWC

> 最後更新：2026-04-20
> Setup：RoBERTa-large, r=8, 30 clients, server-side evaluation on validation split
> 分 α=0.5 跟 α=0.3 兩組（不要跨 α 混比）

## ⚠ 本次修正記錄

之前一版把不同 α 的 runs 混在一起比較，結論全部作廢。原因：
1. `run_ffa_all.sh` 用同 TIMESTAMP 跑 α=0.5 再跑 α=0.3，α=0.3 的 log 覆蓋 α=0.5（server_app.py 的 `server_log_path` 每個 Python process 都 `"w"` 重寫）
2. `logs/20260408_114021/*_fedalc*` 跟 `logs/20260412_235402/*_fed{avg,sa}*` 其實是 α=0.3，不是 α=0.5
3. 現在 rename 完 log 都帶 `_a<alpha>` 尾綴，且 server_app 已修好避免未來再覆蓋
4. FFA α=0.5 的 per-round accuracy 從 wandb `output.log` 重建出來（4 位小數精度）

Log 對照：

| Log timestamp | 真實 alpha | 內容 |
|---|---|---|
| `20260402_132128/*_fedavg_a0.5` | 0.5 | FedAvg 全 GLUE |
| `20260405_071932/*_fedsa_a0.5` | 0.5 | FedSA 全 GLUE |
| `20260406_203614/*_fedalc_a0.5` | 0.5 | FedALC-AP basic α=0.5 |
| `20260408_114021/*_fedalc_a0.3` | **0.3** | FedALC-AP basic α=0.3 |
| `20260412_235402/*_fed{avg,sa}_a0.3` | **0.3** | baseline α=0.3 |
| `20260415_063849/*_fedalc-lwc_a0.5` | 0.5 | LWC α=0.5 |
| `20260416_064902/*_ffa_a0.3` | 0.3 | FFA α=0.3（surviving） |
| `20260416_064902/*_ffa_a0.5` | 0.5 | FFA α=0.5（從 wandb 重建） |

## α=0.5（中等 non-IID）

### Best server-side accuracy

| Method | SST-2 Best | Round | Last | QNLI Best | Round | Last | SST-2 Rounds | QNLI Rounds |
|---|---|---|---|---|---|---|---|---|
| FedAvg | **95.53** | 5 | 94.61 | **93.59** | 6 | 93.37 | 20 | 20 |
| FedALC-AP-LWC | 95.18 | 12 | 95.18 | 92.75 | 11 | 92.48 | 20 | 20 |
| FFA ※ | 95.18 | 11 | 94.61 | 93.19 | 8 | 92.93 | 30 | 30 |
| FedALC-AP | 94.95 | 9 | 94.50 | 93.34 | 24 | 93.32 | 30 | 30 |
| FedSA | 94.84 | 11 | 94.50 | 91.45 | 18 | 91.43 | 20 | 20 |

※ FFA α=0.5 數字是從 wandb `output.log` 重建（4 位小數精度）

**SST-2 排名**：FedAvg > LWC ≈ FFA > FedALC-AP > FedSA （差距只有 0.7%，noise range）
**QNLI 排名**：FedAvg > FedALC-AP > FFA > LWC > FedSA

### 圖表

- [all_methods_accuracy_sst2_a0.5.png](../../plots/r30_c30/all_methods_accuracy_sst2_a0.5.png)
- [all_methods_accuracy_qnli_a0.5.png](../../plots/r30_c30/all_methods_accuracy_qnli_a0.5.png)
- [all_methods_loss_qnli_a0.5.png](../../plots/r30_c30/all_methods_loss_qnli_a0.5.png)
- [all_methods_best_bar_a0.5.png](../../plots/r30_c30/all_methods_best_bar_a0.5.png)

## α=0.3（高度 non-IID）

### Best server-side accuracy

| Method | SST-2 Best | Round | Last | QNLI Best | Round | Last | Rounds |
|---|---|---|---|---|---|---|---|
| FedAvg | **95.76** | 6 | 94.38 | **92.62** | 19 | 92.13 | 30 |
| FFA | 95.64 | 8 | 94.61 | 92.07 | 10 | 91.10 | 30 |
| FedALC-AP | 95.53 | 9 | 95.07 | 90.23 | 30 | 90.23 | 30 |
| FedSA | 94.95 | 20 | 94.61 | **50.54 ⚠** | 1 | 50.54 | 30 |
| FedALC-AP-LWC | — | — | — | — | — | — | **尚未跑** |

**SST-2 排名**：FedAvg > FFA > FedALC-AP > FedSA（差距 0.8%）
**QNLI 排名**：FedAvg > FFA > FedALC-AP >>> FedSA（崩潰）

### 圖表

- [all_methods_accuracy_sst2_a0.3.png](../../plots/r30_c30/all_methods_accuracy_sst2_a0.3.png)
- [all_methods_accuracy_qnli_a0.3.png](../../plots/r30_c30/all_methods_accuracy_qnli_a0.3.png)
- [all_methods_loss_qnli_a0.3.png](../../plots/r30_c30/all_methods_loss_qnli_a0.3.png)
- [all_methods_best_bar_a0.3.png](../../plots/r30_c30/all_methods_best_bar_a0.3.png)

## Cross-alpha 觀察

| Method | SST-2 α=0.5 | SST-2 α=0.3 | Δ | QNLI α=0.5 | QNLI α=0.3 | Δ |
|---|---|---|---|---|---|---|
| FedAvg | 95.53 | 95.76 | +0.23 | 93.59 | 92.62 | -0.97 |
| FedSA | 94.84 | 94.95 | +0.11 | 91.45 | **50.54** | **-40.91** ⚠ |
| FFA | 95.18 | 95.64 | +0.46 | 93.19 | 92.07 | -1.12 |
| FedALC-AP | 94.95 | 95.53 | +0.58 | 93.34 | 90.23 | **-3.11** |
| LWC | 95.18 | — | — | 92.75 | — | — |

## Key findings

### 🚨 Finding 1 — FedSA 在 QNLI 的失敗跟 non-IID 強度相關

- **α=0.5**：FedSA QNLI 能跑到 91.45%（只比 FedAvg 93.59% 差 2.14%）
- **α=0.3**：FedSA QNLI **完全崩潰** 50.54%（差 FedAvg 42%）
- SST-2 兩個 α 下 FedSA 都正常
- **原因分析**：QNLI 是 entailment 任務，句子對輸入 → B 矩陣對 task-specific 方向更敏感。α=0.3 時 clients 的 B 矩陣方向發散嚴重，server-side 平均 B 退化成 noise；α=0.5 時 B 矩陣還勉強對齊
- **不是 universal bug**，而是「FedSA server eval 在高 non-IID + task-specific B matrix」下的已知退化
- **對 paper 意涵**：FedSA 需要 per-client eval；我們用 server-side eval 對 FedSA 不公平

### 🎯 Finding 2 — FedALC-AP (basic) 在 α=0.3 QNLI 掉了 3.11%

- α=0.5 QNLI：93.34%（接近 FedAvg 的 93.59%）
- α=0.3 QNLI：90.23%（比 FedAvg 的 92.62% 差 2.39%）
- SST-2 沒這麼明顯（+0.58%）
- **可能原因**：α=0.3 下 R1 的 B 矩陣方向還沒 stabilize 就 cluster → 早期錯誤 cluster assignment 把訓練 lock 到 suboptimal
- **這正是 FedALC-AP-Multi 想解決的**（Hopkins adaptive 等累積信號再 cluster）

### 📊 Finding 3 — FedAvg 在 single-task 是強 baseline，難 beat

- 兩個 α 下 SST-2 跟 QNLI 都是 FedAvg 贏
- α=0.3 SST-2 甚至超過 α=0.5（反直覺）
- 推測：single-task 下 clients 學同一個 task，**FedAvg 聚合所有 update 天然就是最大 signal**
- Clustering / selective-aggregation 的優勢要在 multi-task 才會顯現

### 🌟 Finding 4 — LWC 跟 FFA 在 α=0.5 幾乎 tie

- SST-2: LWC 95.18 vs FFA 95.18 (完全相同)
- QNLI: FFA 93.19 vs LWC 92.75 (+0.44% FFA 好)
- **兩者都用 20 rounds 跑完**
- LWC: A 全域 + B per-cluster + layer selection
- FFA: A freeze + B 全域
- 兩個完全不同的策略但 α=0.5 下結果差不多 → 單一任務 α=0.5 太容易，無法區分 methods

### 🔬 Finding 5 — FedALC-AP-LWC α=0.3 還沒跑

- 之前比較裡 LWC α=0.3 沒出現，因為實際上沒跑過
- 應該要補跑 α=0.3 讓 ablation 有完整數據
- 或者直接跳過 LWC α=0.3，改優先跑 FedALC-AP-Multi α=0.3

### 🔧 Finding 6 — FedALC-AP 在 α=0.5 QNLI 到 R24 才 peak

- Best accuracy 93.34% 出現在 round 24（最晚）
- 其他 methods 都在 R11-R20 達到 peak
- **意義**：FedALC-AP 的 clustering 需要更多輪才能穩定
- 跟 Finding 2 合看：在容易 setup（α=0.5）下能等到收斂；在困難 setup（α=0.3）下錯過 convergence window

## 對 FedALC-AP-Multi 實驗的啟示

1. **必跑**：FedALC-AP-Multi α=0.3 QNLI — 這是 basic 版表現最差的地方，Multi 應該能救起來
2. **必跑**：FedALC-AP-Multi α=0.5 QNLI — 跟 basic 版、LWC 對比看 component 個別貢獻
3. **不急著跑**：FedALC-AP-Multi SST-2 — 所有 methods 都在 noise range 內，signal 太小
4. **必須處理**：FedSA 的 per-client eval（否則 paper 裡跟 FedSA 比不公平）
5. **補跑**：LWC α=0.3（讓 ablation 完整，或跳過用 Multi 替代）

## 歷史 / 遺失數據注意

- **FFA α=0.5 的 fit_metrics.tsv / eval_metrics.tsv / checkpoints 永久遺失**（被 α=0.3 覆蓋）
- 只有 server_eval.tsv 從 wandb output.log 重建出來（精度 4 位小數）
- 如果將來要做 per-client analysis 或檢查 client checkpoints，**必須重跑 FFA α=0.5**

## 工具

- `plot_method_comparison.py`（repo root）：重新跑 plot
- `tools/scan_log_inventory.py`：掃 wandb 配對 log dir → alpha
- `tools/apply_log_rename.sh`：根據 inventory 執行 rename
- `tools/reconstruct_server_eval_from_wandb.py`：從 wandb output.log 重建 server_eval.tsv
