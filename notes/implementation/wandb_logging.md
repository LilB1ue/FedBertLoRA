# Wandb Logging 設計

## 概述

本專案使用雙軌 logging：TSV 檔案（per-client 細節）+ Wandb（即時監控聚合指標）。Wandb 只在 server 端開 1 個 run，不在 client 端開。

---

## Flower 官方做法（參考）

Flower `advanced-pytorch` 範例（`flwr.serverapp` 新版 API）：

```python
# strategy.py — 在自定義 FedAvg 的 start() 方法裡
wandb.init(project="FLOWER-advanced-pytorch", name=f"{run_dir}-ServerApp")

# 每輪記錄 3 種 aggregated metrics：
wandb.log(dict(agg_train_metrics), step=current_round)      # client 訓練聚合
wandb.log(dict(agg_evaluate_metrics), step=current_round)    # client eval 聚合
wandb.log(dict(res), step=current_round)                     # server-side global eval
```

特點：
- Server 端 **1 個 wandb run**，不是每個 client 各開
- 只記 **aggregated metrics**，不記 per-client
- 同時存 `results.json` 到磁碟（雙軌）
- Best model checkpoint 存檔

---

## 我們的做法

### 架構差異

我們使用舊版 Flower API（`flwr.server`），透過 `fit_metrics_aggregation_fn` 和 `evaluate_metrics_aggregation_fn` 拿到聚合結果。官方範例用新版 API（`flwr.serverapp`）在 strategy 的 `start()` 裡直接控制。

### Wandb 記錄位置

全部在 `server_app.py`，3 個記錄點：

| 記錄點 | 函數 | Wandb prefix | 觸發時機 |
|---|---|---|---|
| Client fit 聚合 | `get_metrics_aggregation_fn(phase="fit")` | `fit/` | 每輪 aggregate_fit 後 |
| Client eval 聚合 | `get_metrics_aggregation_fn(phase="evaluate")` | `evaluate/` | 每輪 aggregate_evaluate 後 |
| Server global eval | `get_evaluate_fn()` | `server/` | 每輪 evaluate_fn 後 |

### Wandb Run 命名

```
{task_name}_{aggregation_mode}_{MMdd_HHMM}
例: sst2_fedavg_0401_1430
例: qnli_fedsa_0402_0900
例: mnli_fedavg_0403_1100
```

每個實驗（不同 task 或不同 strategy）是一個獨立的 run，用時間戳區分同一設定的多次執行。

### 記錄的 Metrics

#### `fit/*` — Client 訓練後的聚合（每輪）

```
fit/train_loss          # weighted avg train loss
fit/train_loss_std      # 跨 client 標準差
fit/train_loss_min
fit/train_loss_max
fit/eval_loss           # weighted avg local eval loss（聚合前 model）
fit/eval_loss_std
fit/eval_accuracy       # weighted avg local eval accuracy
fit/eval_accuracy_std
fit/eval_accuracy_min   # worst client（personalization 核心指標）
fit/eval_accuracy_max   # best client
fit/eval_f1             # QQP only
```

#### `evaluate/*` — Global model 在各 client local test 的聚合（每輪）

```
evaluate/accuracy
evaluate/accuracy_std
evaluate/accuracy_min
evaluate/accuracy_max
evaluate/f1             # QQP only
```

#### `server/*` — Global model 在完整 validation set（每輪）

```
server/accuracy
server/loss
server/f1               # QQP only
server/accuracy_mm      # MNLI mismatched only
```

### Run Config

Wandb run 自動記錄 `pyproject.toml` 的所有參數：

```python
wandb.init(config=dict(cfg))  # cfg = context.run_config
```

包含：task-name, aggregation-mode, lora-r, lora-alpha, dirichlet-alpha, learning-rate, num-server-rounds, batch-size, seed, ...

---

## 設定方式

### pyproject.toml

```toml
# 開啟 wandb
wandb-enabled = true
wandb-project = "bert-federated"

# 關閉（預設）
wandb-enabled = false
```

### 前置條件

```bash
pip install wandb
wandb login
```

---

## TSV vs Wandb 的分工

| 需求 | TSV | Wandb |
|---|---|---|
| Per-client 每輪的詳細 metrics | fit_metrics.tsv, eval_metrics.tsv | 不記（太多） |
| Aggregated metrics 曲線 | 可事後畫 | 即時看 |
| 多實驗比較 | 手動載入比較 | Dashboard 疊圖 |
| 離線分析 | 主要用途 | 輔助 |
| Overfitting 判斷（per-client） | fit_metrics.tsv 的 train_loss vs eval_loss | fit/train_loss vs fit/eval_loss（聚合版） |

**原則：TSV 是 ground truth（per-client 細節），Wandb 是即時監控（聚合概覽）。**

---

## Wandb Dashboard 圖表說明

Wandb 會根據記錄的 metrics 自動生成 line charts。以下列出每個 section 對應的圖表及其用途。

### Server Section（全域模型品質）

| 圖表 | Metric | 說明 |
|------|--------|------|
| Server Accuracy | `server/accuracy` | **最重要的曲線**。全域模型在完整 validation set 的 accuracy，用來判斷 best round 和 overfitting |
| Server Loss | `server/loss` | 全域模型 validation loss，通常比 accuracy 更早反映 overfitting |
| Server F1 | `server/f1` | QQP 任務專用（accuracy + F1 雙指標） |
| Server Accuracy MM | `server/accuracy_mm` | MNLI 任務專用，mismatched validation set 的 accuracy |

### Fit Section（Client 訓練後的聚合指標）

| 圖表 | Metric | 說明 |
|------|--------|------|
| Client Eval Accuracy | `fit/eval_accuracy` | 所有 client 本地 eval accuracy 的加權平均 |
| Client Eval Accuracy Std | `fit/eval_accuracy_std` | client 間 accuracy 差異程度，反映 non-IID 的影響 |
| Client Eval Accuracy Min/Max | `fit/eval_accuracy_min`, `fit/eval_accuracy_max` | 最弱/最強 client，min 是 personalization 的核心指標 |
| Client Train Loss | `fit/train_loss` | client 訓練 loss 加權平均 |
| Client Train Loss Std | `fit/train_loss_std` | 訓練 loss 跨 client 分散程度 |
| Client Eval Loss | `fit/eval_loss` | client 本地 eval loss 加權平均 |
| Client Eval F1 | `fit/eval_f1` | QQP 任務專用 |

### Evaluate Section（全域模型在各 client 本地 test 的聚合）

| 圖表 | Metric | 說明 |
|------|--------|------|
| Evaluate Accuracy | `evaluate/eval_accuracy` | 全域模型在各 client 本地 test set 的加權平均 accuracy |
| Evaluate Accuracy Std/Min/Max | `evaluate/eval_accuracy_std`, `_min`, `_max` | 全域模型在不同 client 上的表現差異 |
| Evaluate Loss | `evaluate/eval_loss` + `_std`/`_min`/`_max` | 對應的 loss 分佈 |

### 關鍵圖表用途對照

| 分析目的 | 看哪些圖 |
|----------|----------|
| 判斷 best round / 是否 overfitting | `server/accuracy` + `server/loss` |
| 比較不同策略（FedAvg vs FedSA） | 多 run 疊 `server/accuracy` |
| 評估 non-IID 影響程度 | `fit/eval_accuracy_std` + `fit/eval_accuracy_min` |
| 評估 personalization 效果 | `fit/eval_accuracy_min`（FedSA 應讓 worst client 提升） |
| 判斷 client drift / divergence | `fit/train_loss_std` 是否隨 round 持續增大 |
| Overfitting 細節 | `fit/train_loss`↓ 但 `server/loss`↑ 就是 overfitting |

### Wandb 多 Run 比較技巧

- 在同一個 project (`bert-federated`) 下，可用 **Group by** run name prefix（如 `sst2_*`）來比較同任務不同策略
- 用 **Filter** 篩選 `config.aggregation_mode` 或 `config.task_name`
- **Panel** 疊圖時選 x-axis = `round`（不是 `_step`），確保對齊
