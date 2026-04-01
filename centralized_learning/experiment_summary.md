# Centralized LoRA Fine-tuning — Experiment Summary

## 1. Hyperparameter Comparison: Ours vs LoRA Paper

| Hyperparameter | **Ours** | **LoRA Paper (Hu et al. 2021)** |
|----------------|----------|----------------------------------|
| Model | roberta-large (355M) | roberta-large (355M) |
| LoRA rank (r) | 8 | 8 |
| LoRA alpha | 16 | 16 |
| Alpha/r (scaling) | 2 | 2 |
| Target modules | q, k, v, dense | q, v only |
| Trainable params | 4.59M (1.28%) | ~0.8M (0.23%) |
| Learning rate | 1e-4 | ~4e-4 (task-specific) |
| Effective batch size | 128 (32 × 4 accum) | 16 |
| LR schedule | cosine | linear decay |
| Warmup | 6% | 6% |
| Weight decay | 0.01 | 0.1 |
| Optimizer | AdamW | AdamW |
| Max epochs | 10 + early stop (patience=3) | fixed (task-specific) |
| Seed | 42 (single run) | median of 5 seeds |
| Max seq length | 128 | 128 |

**Note**: Our trainable params are ~5.7× more than the paper due to additional target modules (key + dense). LR difference is partially offset by larger batch size (linear scaling rule: effective LR ≈ 1e-4 × 8 = 8e-4).

---

## 2. GLUE Dataset Sizes

| Task | Train | Validation | Test | **# Labels** | Label Names | Note |
|------|------:|----------:|-----:|:------------:|-------------|------|
| **SST-2** | 67,349 | 872 | 1,821 | **2** | negative / positive | test labels hidden |
| **QNLI** | 104,743 | 5,463 | 5,463 | **2** | entailment / not_entailment | test labels hidden |
| **MNLI** | 392,702 | 9,815 (matched) + 9,832 (mismatched) | 9,796 + 9,847 | **3** | entailment / neutral / contradiction | test labels hidden |
| **QQP** | 363,849 | 40,430 | 390,965 | **2** | not_duplicate / duplicate | test labels hidden |
| **RTE** | 2,490 | 277 | 3,000 | **2** | entailment / not_entailment | test labels hidden |

All GLUE test set labels are hidden (-1). Evaluation uses **validation set**.

**Dirichlet non-IID 相關**：
- 二分類 (SST-2, QNLI, QQP, RTE)：Dirichlet(α) 的主要變異軸是一個 scalar（label 比例 p），分布越不均衡 α 越小
- 三分類 (MNLI)：分布在 2D simplex 上，α 控制三個 class 的同時不均衡程度，異質性更複雜

---

## 3. Completed Experiments — Results & Timing

| Task | Best Acc | Best Epoch | Early Stop | Train Time | Epochs Run |
|------|---------|-----------|------------|-----------|-----------|
| **SST-2** | **95.99%** | 5 | epoch 8 | ~66 min | 8 |
| **QNLI** | **94.78%** | 3 | epoch 6 | ~120 min | 6 |
| **MNLI** | **90.73%** | 6 | epoch 9 | ~10 hr | 9 |
| **QQP** | - | - | - | - | - |
| **RTE** | - | - | - | - | - |

### Per-epoch Training Time Estimate

| Task | Steps/epoch | Time/epoch |
|------|------------|-----------|
| SST-2 | 527 | ~8 min |
| QNLI | 820 | ~20 min |
| MNLI | 3,068 | ~66 min |
| QQP | 2,843 | ~60 min |
| RTE | 19 | <1 min |

---

## 4. Related Papers Hyperparameter Comparison (RoBERTa-large + GLUE)

### Training Hyperparameters

| | **Ours** | **LoRA (2021)** | **FFA-LoRA (2024)** | **FedSA-LoRA (2025)** | **RoLoRA (2024)** | **ADF-LoRA (2025)** |
|---|---|---|---|---|---|---|
| Setting | Centralized | Centralized | Federated (3 clients) | Federated (3 clients) | Federated (3 clients) | Decentralized (10 clients) |
| Optimizer | AdamW | AdamW | SGD | SGD | SGD | AdamW |
| LR | 1e-4 | 2e-4~4e-4 (task-specific) | 0.01~0.1 (tuned) | 5e-3~1e-2 (tuned) | tuned | 5e-4~5e-3 (tuned) |
| LR schedule | Cosine | Linear decay | Constant | Constant | Constant | Constant |
| Warmup | 6% | 6% | — | — | — | — |
| Weight decay | 0.01 | ~0.01 (AdamW default) | — | — | — | — |
| Batch size | 128 (32×4 accum) | 4~8 | 200 | 128 | 32~64 | 32 |
| **Max seq length** | **128 (全 tasks)** | **512 (MNLI/QNLI/QQP/RTE)** / 128 (SST-2) | **128** | **128** (follows FFA-LoRA) | 未明確 | **128** |
| Epochs/Steps | 10 epochs + early stop | 10~20 epochs | 10 steps/round × 1000 rounds | 10 steps/round × 1000 rounds | 20 epochs/round × 200~500 rounds | 20 steps/round × 150 rounds |
| Seed | 42 (×1) | median of 5 seeds | — | — | — | — |

### LoRA Hyperparameters

| | **Ours** | **LoRA (2021)** | **FFA-LoRA (2024)** | **FedSA-LoRA (2025)** | **RoLoRA (2024)** | **ADF-LoRA (2025)** |
|---|---|---|---|---|---|---|
| Rank r | 8 | 8 | 8 | 8 | 1/2/4/8 | 8 |
| Alpha α | 16 | 16 | **8** | 16 | 未報告 | 16 |
| Dropout | 0 | 0 | 0 | 0 | 0 | **0.1** |
| **Target modules** | **Q, K, V, dense** | **Q, V** | **Attention + FFN** | **Q, V** | **Q, V** | **Q, V** |
| Trainable params | 4.59M (1.28%) | ~0.8M (0.23%) | 更多（含FFN） | ~0.8M | ~0.8M | ~0.8M |

### 重點觀察

- **Max seq length**: 原始 LoRA 論文對 MNLI/QNLI/QQP/RTE 用 **512**，我們和 federated 論文都用 **128** → 我們的 QNLI/MNLI 可能因截斷而略有損失
- **Optimizer**: FL 論文偏好 SGD（收斂更穩定），原始 LoRA 和我們用 AdamW
- **Target modules**: 我們多加了 K 和 dense，trainable params 是論文的 ~5.7 倍，但結果相近甚至更好
- **Alpha**: FFA-LoRA 用 α=r=8（scaling=1），其他包含我們都用 α=16（scaling=2）

---

## 5. Results vs LoRA Paper

| Task | **Ours** | **LoRA Paper** | **Diff** |
|------|---------|----------------|---------|
| SST-2 | 95.99% | 96.2 ± 0.5 | -0.21% |
| QNLI | 94.78% | 94.8 ± 0.3 | -0.02% |
| MNLI | 90.73%* | ~90.2 | +0.53%* |
| QQP | - | ~91.6 | - |
| RTE | - | 85.2 ± 1.1 | - |



All results within or exceeding the LoRA paper's reported range. Slight advantage on MNLI likely due to more target modules.
