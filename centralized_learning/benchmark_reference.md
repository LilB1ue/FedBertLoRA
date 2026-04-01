# RoBERTa-large + LoRA GLUE Benchmark Reference

## LoRA 論文 (Hu et al. 2021) — RoBERTa-large, r=8, alpha=16

| Task | LoRA (r=8) | Full Fine-tuning |
|------|-----------|-----------------|
| SST-2 | 96.2 ± 0.5 | ~96.4 |
| QNLI | 94.8 ± 0.3 | ~94.7 |
| MNLI | ~90.2 | ~90.2 |
| QQP | ~91.6 | ~92.2 |
| RTE | 85.2 ± 1.1 | ~86.6 |

- 論文只用 Wq, Wv 兩個 module
- Trainable params: ~0.8M (r=8), 佔全部 355M 的 0.23%
- Dev set 結果, median over 5 seeds

## 我們的設定

- Model: `roberta-large` (355M)
- LoRA: r=8, alpha=16, target modules: `query, key, value, dense` (比論文多 key + dense)
- Trainable params: 4.59M (1.28%) — 比論文多因為 target modules 更多
- LR: 1e-4, cosine schedule, 6% warmup, AdamW (weight_decay=0.01)
- Batch: 32 × 4 grad_accum = effective 128
- Max epochs: 10, early stopping patience=3

## 我們的實驗結果

### SST-2 (2026-03-31) — 完成

| Epoch | Train Loss | Eval Accuracy | Eval Loss | Note |
|-------|-----------|---------------|-----------|------|
| 1 | 0.178 | 95.30% | 0.149 | |
| 2 | 0.140 | 95.07% | 0.165 | |
| 3 | 0.117 | 95.64% | 0.141 | |
| 4 | 0.104 | 95.87% | 0.126 | |
| **5** | **0.093** | **95.99%** | 0.153 | **best** |
| 6 | 0.080 | 95.99% | 0.153 | patience=1 |
| 7 | 0.071 | 95.64% | 0.163 | patience=2 |
| 8 | 0.065 | 95.76% | 0.161 | early stop |

- **Final best: 95.99% (epoch 5)**，early stop 於 epoch 8
- 訓練時間: ~41 分鐘
- Wandb: https://wandb.ai/louischen0609-/bert-centralized/runs/jnpsjhsj

### QNLI
(pending)

### MNLI (2026-04-01) — 完成

| Epoch | Train Loss | Eval Accuracy | Eval Loss | Note |
|-------|-----------|---------------|-----------|------|
| 1 | 0.337 | 89.64% | 0.286 | |
| 2 | 0.311 | 90.11% | 0.284 | |
| 3 | 0.282 | 90.16% | 0.272 | |
| 4 | 0.260 | 90.39% | 0.266 | |
| 5 | 0.241 | 90.33% | 0.278 | |
| **6** | **0.222** | **90.73%** | 0.271 | **best** |
| 7 | 0.198 | 90.56% | 0.283 | patience=1 |
| 8 | 0.182 | 90.50% | 0.296 | patience=2 |
| 9 | 0.166 | 90.61% | 0.301 | early stop |

- **Final best: 90.73% (epoch 6)**，early stop 於 epoch 9
- 訓練時間: ~10 hr

### QQP
(pending)

### RTE
(pending)

## 參考資料

- [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
- [Microsoft LoRA GitHub - NLU Examples](https://github.com/microsoft/LoRA/blob/main/examples/NLU/README.md)
- [LoRA-XS (arxiv 2405.17604)](https://arxiv.org/abs/2405.17604)
- [Dual LoRA (arxiv 2512.03402)](https://arxiv.org/abs/2512.03402)
