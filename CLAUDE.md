# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 專案概述

基於 Flower 框架實作聯邦 LoRA 微調實驗。使用 `roberta-large` (355M) + PEFT LoRA 在 GLUE benchmark (SST-2, QNLI, MNLI, QQP, RTE) 上進行分散式微調，以 Dirichlet 分佈模擬 non-IID 資料分割 (30 clients)。

支援四種聚合策略：
- **FedAvg**: 標準 FedAvg，A+B 全聚合
- **FedSA-LoRA**: A 聚合，B 留本地
- **FFA-LoRA**: A freeze，只聚合 B
- **FedALC-AP**: A 全域聚合，B 按 AP clustering 群內聚合，others 留本地（原簡單版 FedALC）
- **FedALC-AP-LWC**: FedALC-AP + silhouette warm-up + Metric B layer selection + clustering freeze（ablation baseline）
- **FedALC-AP-Multi**: FedALC-AP + 內建 LWC layer selection + Hopkins adaptive warm-up + cumulative ΔB + freeze（主方法，targets multi-task FL）

### 相關論文
- **FedSA-LoRA** (ICLR 2025): A 矩陣聚合、B 矩陣留本地
- **FedADC** (Computer Networks 2026): MADC + AP clustering + 交替 A/B 分離
- **HiLoRA** (CVPR 2026): 三層 LoRA hierarchy (root/cluster/leaf) + orthogonality
- 完整列表見 `notes/papers/related_papers.md`
- PDF 存放在 `papers/` 目錄

## 常用指令

```bash
# 安裝依賴
pip install -e .

# 聯邦學習 (預設 FedSA-LoRA, SST-2, 30 clients, 20 rounds)
flwr run .

# GPU 模式
flwr run . localhost-gpu

# 集中式訓練 — 跑全部 GLUE 任務 (r=8, early stopping, log 存檔)
bash centralized_learning/run_all.sh
bash centralized_learning/run_all.sh --wandb   # 加 wandb logging

# 單一任務
python centralized_learning/train.py --task sst2
python centralized_learning/train.py --task qnli --epochs 10 --wandb

# 批次跑（含 wandb）
bash run_fedavg_all.sh              # FedAvg 全任務
bash run_fedsa_all.sh               # FedSA-LoRA 全任務
bash run_fedalc_all.sh              # FedALC-AP α=0.5 (SST-2 + QNLI, 30 rounds)
bash run_fedalc_alpha03.sh          # FedALC-AP α=0.3
bash run_baseline_alpha03.sh        # FedAvg + FedSA α=0.3 baselines
bash run_fedalc_ap_lwc.sh           # FedALC-AP-LWC α=0.5 (SST-2 + QNLI, 20 rounds)
bash run_fedalc_ap_lwc.sh local-simulation 30 0.3  # FedALC-AP-LWC α=0.3
bash run_fedalc_ap_multi.sh         # FedALC-AP-Multi (adaptive warm-up + cumulative ΔB + freeze)
bash run_fedalc_ap_multi_smoke.sh   # FedALC-AP-Multi 5-round smoke test
```

### 快速切換實驗設定 (修改 pyproject.toml)

```toml
# 切換策略
aggregation-mode = "fedsa"   # "fedavg" | "fedsa" | "ffa" | "fedalc-ap" | "fedalc-ap-lwc" | "fedalc-ap-multi"

# 切換任務 + LoRA rank
task-name = "sst2"           # "sst2" | "qnli" | "mnli" | "qqp" | "rte"
lora-r = 8                   # 全部任務統一 r=8

# 切換 non-IID 程度
dirichlet-alpha = 0.5        # 0.3(較強) / 0.5(中等) / 1.0(輕微)
# 注意: α=0.1 在 30 clients + binary task 下 DirichletPartitioner 分割失敗
```

## 架構

```
bert/
├── models.py          # LoRA 模型載入 + A/B 矩陣分離 + cosine_annealing LR schedule
├── dataset.py         # GLUE 資料載入 + DirichletPartitioner non-IID 分割
├── strategy.py        # FedAvgStrategy: 標準 FedAvg 策略
├── fedsa_strategy.py  # FedSALoRAStrategy: selective aggregation (FedSA/FFA 模式)
├── fedalc_ap_strategy.py # FedALCAPStrategy: AP clustering B + global A + local others (basic)
├── fedalc_ap_lwc_strategy.py # FedALCAPLWCStrategy: silhouette warm-up → layer selection → freeze (ablation baseline)
├── fedalc_ap_multi_strategy.py # FedALCAPMultiStrategy: adaptive warm-up (Hopkins) → cumulative ΔB clustering → freeze + built-in layer selection (main method for multi-task)
├── client_app.py      # FlowerClient: 本地 LoRA 訓練 + checkpoint 存儲
├── server_app.py      # ServerApp: 初始化模型 + strategy 選擇 + server-side evaluation + wandb
└── __init__.py
centralized_learning/
├── train.py           # 集中式訓練 (HF Trainer, early stopping, wandb, MNLI-mm eval)
└── run_all.sh         # 批次跑全部 GLUE 任務 + log 存檔
run_fedavg_all.sh      # FedAvg 全任務批次跑
run_fedsa_all.sh       # FedSA-LoRA 全任務批次跑
run_fedalc_all.sh      # FedALC-AP α=0.5 批次跑 (SST-2 + QNLI)
run_fedalc_alpha03.sh  # FedALC-AP α=0.3
run_fedalc_ap_lwc.sh   # FedALC-AP-LWC (ablation baseline)
run_fedalc_ap_multi.sh # FedALC-AP-Multi (main method)
run_fedalc_ap_multi_smoke.sh # FedALC-AP-Multi 5-round smoke test
run_baseline_alpha03.sh # FedAvg + FedSA α=0.3 baselines
papers/                # 相關論文 PDF
notes/                 # 筆記（方法設計、實驗結果、論文整理、研究規劃）
plots/                 # 實驗圖表（按 r{rounds}_c{clients}/ 子目錄分類）
pyproject.toml         # Flower 配置 + 所有超參數
```

### 參數分離邏輯

基於 PEFT 參數命名慣例，以 `lora_param_keys` 做 name-based 分離：
- `"lora_A" in key` → A 矩陣
- `"lora_B" in key` → B 矩陣
- 其餘（classifier, score 等）→ others

不同 strategy 的處理方式：

| 參數 | FedAvg | FedSA/FFA | FedALC-AP* |
|---|---|---|---|
| A | global avg | global avg (FedSA) / freeze (FFA) | global avg |
| B | global avg | local (FedSA) / global avg (FFA) | AP clustering → per-cluster avg |
| Others | global avg | 與 B 綁定 | per-client local |

### 資料流 (FedALC-AP 系列模式)

1. `server_app` 初始化 `roberta-large` + LoRA → 提取初始參數 + parameter key ordering
2. `FedALCAPStrategy.configure_fit()` → 為每個 client 組裝 **global A + cluster B + own others**
3. 各 client 收到個人化參數 → 存 received_checkpoints → 本地訓練 → 存 client_checkpoints → 回傳
4. `FedALCAPStrategy.aggregate_fit()` →
   - A: 全域 FedAvg
   - B: cosine similarity → AP clustering → per-cluster FedAvg
   - Others: 存回各 client 字典（不聚合）
   - 寫 clustering.jsonl（每輪的 cluster 數、silhouette、client 分配）
5. `evaluate_fn` 用 global A + avg B + avg others 在 centralized valid set 評估

### Checkpoint 存儲規則

| 存儲位置 | 內容 | FedAvg | FedSA/FFA/FedALC-AP* |
|---|---|---|---|
| `global_checkpoints/` | 聚合後 global model | 每輪存 | 不存 |
| `client_checkpoints/` | 訓練後、聚合前 adapter | 每輪存 | 每輪存 |
| `received_checkpoints/` | 聚合後、訓練前 adapter | 不存 | 每輪存 |

### Log 結構

```
logs/{timestamp}/{task}_{strategy}/
├── clustering.jsonl          # FedALC-AP* only: 每輪 cluster 分配
# NOTE: main comparison metric is client-side (eval_metrics.tsv), NOT server_eval.tsv,
# because the project targets personalized models. server_eval is sanity check only.
├── fit_metrics.tsv           # per-client train/eval metrics
├── eval_metrics.tsv          # per-client eval metrics
├── server_eval.tsv           # server-side global eval
├── summary.log               # Flower summary
├── global_checkpoints/       # FedAvg only
├── client_checkpoints/       # 所有策略
└── received_checkpoints/     # 非 FedAvg 策略
```

## 配置參數 (pyproject.toml)

| 參數 | 預設值 | 說明 |
|---|---|---|
| `model-name` | `roberta-large` | HuggingFace 模型 |
| `task-name` | `sst2` | GLUE 任務 |
| `lora-r` | `8` | LoRA rank (全部任務統一 r=8) |
| `lora-alpha` | `16` | LoRA scaling |
| `lora-target-modules` | `query,key,value,dense` | LoRA 套用的模組（attention Q/K/V + 所有 dense） |
| `aggregation-mode` | `fedsa` | 聚合策略: fedavg / fedsa / ffa / fedalc-ap / fedalc-ap-lwc / fedalc-ap-multi |
| `warmup-sil-threshold` | `0.5` | AP-LWC/AP-Multi: warm-up 結束的 silhouette 門檻 |
| `freeze-sil-threshold` | `0.8` | AP-LWC/AP-Multi: clustering freeze 的 silhouette 門檻 |
| `layer-selection-k` | `10` | AP-LWC/AP-Multi: top-K 層數 |
| `layer-reselect-every` | `1` | AP-LWC/AP-Multi: 每 N 輪重新選層（0=one-shot） |
| `hopkins-threshold` | `0.75` | AP-Multi: Hopkins H > threshold → 結束 warm-up |
| `warmup-max-rounds` | `10` | AP-Multi: warm-up 硬上限 |
| `freeze-stable-rounds` | `3` | AP-Multi: cluster 連續 N 輪不變 → freeze |
| `layer-score-feature` | `cumulative_delta_b` | AP-Multi: layer scoring feature (`cumulative_delta_b` \| `current_b`) |
| `dirichlet-alpha` | `0.5` | Non-IID 程度 |
| `num-server-rounds` | `20` | FL 總輪數 |
| `fraction-fit` | `1.0` | 每輪參與比例 |
| `learning-rate` | `0.001` | AdamW 最大學習率 (cosine annealing lrate_max) |
| `batch-size` / `grad-accum-steps` | `32` / `4` | Effective batch = 128 |
| `max-seq-length` | `128` | 最大 token 長度 |
| `local-epochs` | `1` | 每輪本地訓練 epoch |

## 集中式訓練 (Centralized Baseline)

- 使用 HuggingFace `Trainer` + `EarlyStoppingCallback(patience=3)` 監控 accuracy
- LR: 1e-4, cosine schedule, 6% warmup, AdamW (weight_decay=0.01)
- Logging: 每 epoch 記錄 train loss + eval accuracy（不記錄每 step）
- Wandb: `--wandb` 啟用，關閉硬體監控 (`_disable_stats`, `_disable_meta`)
- MNLI 額外評估 validation_mismatched split
- Checkpoint: `save_total_limit=3`, `load_best_model_at_end=True`
- GLUE 任務: SST-2, QNLI, MNLI, QQP, RTE（全部 r=8）
- 資料來源: `nyu-mll/glue` (QNLI/MNLI/QQP/RTE), `stanfordnlp/sst2` (SST-2)，資料內容相同
- GLUE test set 的 label 是 hidden (-1)，因此用 validation set 做評估

## 設計備註

- `dataset.py` 中 `_fds_cache` 以 `(task, num_partitions, alpha)` 為 key 做全域快取
- `fedsa_strategy.py` 中 `client_b_matrices: Dict[str, List[np.ndarray]]` 追蹤每個 client 的 B 矩陣狀態
- `fedalc_ap_strategy.py` 中用 AP clustering 自動決定 cluster 數量，silhouette score 衡量分群品質
- `fedalc_ap_strategy.py` 的 clustering metrics 獨立 log 到 wandb（因為 fit_metrics_aggregation_fn 在 clustering 之前就已 log）
- 集中式和聯邦式共用 `bert/models.py` + `bert/dataset.py`，確保模型架構和資料處理一致
- FL client 使用 `lr_scheduler_type="constant"`，LR 由 server-side cosine annealing 每輪控制
- FL 的 wandb 設定在 `pyproject.toml` 的 `wandb-enabled` / `wandb-project`
