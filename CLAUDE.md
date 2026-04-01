# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 專案概述

基於 Flower 框架實作 **FedSA-LoRA**（Selective Aggregation for Low-Rank Adaptation）聯邦學習實驗。使用 `roberta-large` (355M) + PEFT LoRA 在 GLUE benchmark (SST-2, QNLI, MNLI) 上進行分散式微調，以 Dirichlet 分佈模擬 non-IID 資料分割 (20 clients)。

核心策略：只聚合 LoRA A 矩陣（general knowledge），B 矩陣留在 client 端（client-specific knowledge）。支援切換為 FFA-LoRA、標準 FedAvg 等模式。

### 相關論文
- **FedSA-LoRA** (ICLR 2025): A 矩陣聚合、B 矩陣留本地
- **FedADC** (Computer Networks 2026): 交替 similarity/dissimilarity clustering + A/B 分離
- 完整列表見 `related_papers.md`

## 常用指令

```bash
# 安裝依賴
pip install -e .

# 聯邦學習 (預設 FedSA-LoRA, SST-2, 20 clients, 100 rounds)
flwr run .

# GPU 模式
flwr run . localhost-gpu

# 集中式訓練 — 跑全部 GLUE 任務 (r=8, early stopping, log 存檔)
bash centralized_learning/run_all.sh
bash centralized_learning/run_all.sh --wandb   # 加 wandb logging

# 單一任務
python centralized_learning/train.py --task sst2
python centralized_learning/train.py --task qnli --epochs 10 --wandb
```

### 快速切換實驗設定 (修改 pyproject.toml)

```toml
# 切換策略
aggregation-mode = "fedsa"   # "fedavg" | "fedsa" | "ffa"

# 切換任務 + LoRA rank
task-name = "sst2"           # "sst2" | "qnli" | "mnli" | "qqp" | "rte"
lora-r = 8                   # SST-2 用 8, QNLI/MNLI 用 16

# 切換 non-IID 程度
dirichlet-alpha = 0.5        # 0.1(嚴重) / 0.5(中等) / 1.0(輕微)
```

## 架構

```
bert/
├── models.py       # LoRA 模型載入 + A/B 矩陣分離 + cosine_annealing LR schedule
├── dataset.py      # GLUE 資料載入 + DirichletPartitioner non-IID 分割
├── strategy.py     # FedSALoRAStrategy(FedAvg): selective aggregation 核心邏輯
├── client_app.py   # FlowerClient: 本地 LoRA 訓練 (AdamW, LR from server-side cosine annealing)
├── server_app.py   # ServerApp: 初始化模型 + strategy + server-side evaluation
└── __init__.py
centralized_learning/
├── train.py        # 集中式訓練 (HF Trainer, early stopping, wandb, MNLI-mm eval)
└── run_all.sh      # 批次跑全部 GLUE 任務 + log 存檔
pyproject.toml      # Flower 配置 + 所有超參數
```

### 資料流 (FedSA-LoRA 模式)

1. `server_app` 初始化 `roberta-large` + LoRA → 提取初始參數 + parameter key ordering
2. `FedSALoRAStrategy.configure_fit()` → 為每個 client 組裝 **global A + client 自己的 B**，並透過 config 傳送 round-level cosine annealing LR
3. 各 client 收到個性化參數 + 當前 round 的 LR → 本地訓練 → 回傳更新後的 A+B
4. `FedSALoRAStrategy.aggregate_fit()` → A 矩陣做 FedAvg，B 矩陣存回各 client 字典
5. `evaluate_fn` 在 server 端用 GLUE validation set 評估全域模型

### A/B 矩陣分離

基於 PEFT 參數命名慣例（`"lora_A"` / `"lora_B"` in parameter name），在 `strategy.py` 中以 `lora_param_keys` 做 name-based 分離。此設計參考自 `/data/experiment/exp-fed/flowertune-clustering/` 的 `_extract_lora_matrices_by_name()` 方法。

## 配置參數 (pyproject.toml)

| 參數 | 預設值 | 說明 |
|---|---|---|
| `model-name` | `roberta-large` | HuggingFace 模型 |
| `task-name` | `sst2` | GLUE 任務 |
| `lora-r` | `8` | LoRA rank (全部任務統一 r=8) |
| `lora-alpha` | `16` | LoRA scaling |
| `lora-target-modules` | `query,key,value,dense` | LoRA 套用的模組（attention Q/K/V + 所有 dense） |
| `aggregation-mode` | `fedsa` | 聚合策略 |
| `dirichlet-alpha` | `0.5` | Non-IID 程度 |
| `num-server-rounds` | `100` | FL 總輪數 |
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
- `strategy.py` 中 `client_b_matrices: Dict[str, List[np.ndarray]]` 追蹤每個 client 的 B 矩陣狀態
- `aggregation_mode` 設計為可擴展：未來可加入 `"cluster"` 模式實作 clustering-based B aggregation
- 集中式和聯邦式共用 `bert/models.py` + `bert/dataset.py`，確保模型架構和資料處理一致
- FL client 使用 `lr_scheduler_type="constant"`，LR 由 server-side cosine annealing 每輪控制
- FL 的 wandb 設定在 `pyproject.toml` 的 `wandb-enabled` / `wandb-project`
