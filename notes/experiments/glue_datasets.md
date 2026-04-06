# GLUE Datasets 介紹

> 本專案使用的 4 個 GLUE benchmark 子任務（不含 RTE，因 Dirichlet 分割在 30 clients 下會失敗）。

## 總覽

| Task | 類型 | Labels | Train | Validation | 來源 |
|------|------|--------|------:|----------:|------|
| SST-2 | 情感分析 | 2 (positive/negative) | 67,349 | 872 | stanfordnlp/sst2 |
| QNLI | 自然語言推論 | 2 (entailment/not_entailment) | 104,743 | 5,463 | nyu-mll/glue (qnli) |
| MNLI | 自然語言推論 | 3 (entailment/neutral/contradiction) | 392,702 | 9,815 + 9,832 | nyu-mll/glue (mnli) |
| QQP | 語意相似度 | 2 (not_duplicate/duplicate) | 363,846 | 40,430 | nyu-mll/glue (qqp) |

## 各 Dataset 說明

### SST-2 (Stanford Sentiment Treebank)

- **任務**: 判斷電影評論句子是 positive 還是 negative
- **輸入**: 單句（例如 "This movie is great"）
- **資料量**: 67K train，最小的 dataset
- **特點**: Binary classification，最簡單的 GLUE 任務，baseline accuracy 通常 >93%

### QNLI (Question Natural Language Inference)

- **任務**: 給定一個問題和一個句子，判斷句子是否包含問題的答案
- **輸入**: 兩句（question + sentence）
- **資料量**: 105K train
- **特點**: 從 SQuAD 轉換而來，Binary classification

### MNLI (Multi-Genre Natural Language Inference)

- **任務**: 給定 premise 和 hypothesis，判斷關係是 entailment / neutral / contradiction
- **輸入**: 兩句（premise + hypothesis）
- **資料量**: 393K train，最大的 dataset
- **特點**:
  - **3-class classification**（唯一非 binary 的任務）
  - 有兩個 validation set：matched（同 genre）和 mismatched（不同 genre）
  - 在 FL non-IID 設定下最具挑戰性（3 個 class 的 Dirichlet 分佈更不均）

### QQP (Quora Question Pairs)

- **任務**: 判斷兩個問題是否語意相同（duplicate）
- **輸入**: 兩句（question1 + question2）
- **資料量**: 364K train
- **特點**: Binary classification，label 不平衡（約 63% not_duplicate, 37% duplicate）

## 在 FL 中的注意事項

- **GLUE test set 的 label 是 hidden (-1)**，所以用 validation set 做評估
- SST-2 來源不同（stanfordnlp/sst2），其他三個來自 nyu-mll/glue，但資料內容相同
- FL 中用 Dirichlet α=0.5 分割 train set 給各 client，每個 client 再做 80/20 的 train/eval split
- MNLI 的 3-class 在 non-IID 下最容易出現某些 client 完全缺少某個 class 的情況

## 30 Clients + Dirichlet α=0.5 下的分割情況

| Task | 平均/client | 最小 client | 最大 client | Max/Min ratio |
|------|----------:|----------:|----------:|-------------:|
| SST-2 | 2,245 | 42 | 9,279 | 221x |
| QNLI | 3,491 | 71 | 15,337 | 216x |
| MNLI | 13,090 | 295 | 38,938 | 132x |
| QQP | 12,128 | 278 | 60,362 | 217x |

詳細 per-client 分佈見 [data_distribution_30clients.md](data_distribution_30clients.md)。
