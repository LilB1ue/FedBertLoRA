# 文件索引

所有筆記整理在 `` 資料夾下，按主題分類。

## papers/ — 論文整理

| 文件 | 說明 |
|------|------|
| [related_papers.md](papers/related_papers.md) | FL + LoRA 相關論文完整列表（含實驗設定、超參數比較表） |
| [personalized_fl_papers.md](papers/personalized_fl_papers.md) | Personalized FL 經典論文：FedPer、FedRep、LG-FedAvg、FedBN 等 |
| [FedALC-LoRA.md](papers/FedALC-LoRA.md) | FedALC-LoRA 方法設計筆記 |

## implementation/ — 實作細節

| 文件 | 說明 |
|------|------|
| [fedsa_detailed_implementation.md](implementation/fedsa_detailed_implementation.md) | FedSA-LoRA 論文的詳細評估協議（含與本專案的差異分析） |
| [fedsa_code_review.md](implementation/fedsa_code_review.md) | FedSA-LoRA 實作的 code review 結果（6 個 issues） |
| [wandb_logging.md](implementation/wandb_logging.md) | Wandb 設定與 logging 說明 |

## concepts/ — 概念筆記

| 文件 | 說明 |
|------|------|
| [lora_vs_full_finetuning.md](concepts/lora_vs_full_finetuning.md) | LoRA vs 全參數微調架構圖解、classifier/logits 位置、參數數量對比 |
| [why_local_classifier.md](concepts/why_local_classifier.md) | 為什麼 classifier 要留本地不聚合（gradient 分析、經典論文） |

## experiments/ — 實驗數據

| 文件 | 說明 |
|------|------|
| [glue_datasets.md](experiments/glue_datasets.md) | GLUE 四個任務介紹、資料量、FL 分割情況 |
| [data_distribution_sst2.md](experiments/data_distribution_sst2.md) | SST-2 資料分佈分析（40 clients, α=0.5） |
| [data_distribution_30clients.md](experiments/data_distribution_30clients.md) | 全 GLUE 任務資料分佈分析（30 clients, α=0.5） |
| [fedavg_results.md](experiments/fedavg_results.md) | FedAvg 實驗結果（30 clients, 20 rounds, 全 GLUE 任務） |
| [fedavg_vs_fedsa_comparison.md](experiments/fedavg_vs_fedsa_comparison.md) | FedAvg vs FedSA-LoRA 完整比較（accuracy 表格 + 曲線圖） |

## 專案文件（根目錄）

| 文件 | 說明 |
|------|------|
| [README.md](../README.md) | 專案基本說明 |
| [CLAUDE.md](../CLAUDE.md) | Claude Code 使用指引、架構設計、常用指令 |
