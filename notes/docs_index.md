# 文件索引

所有筆記整理在 `notes/` 資料夾下，按主題分類。

## papers/ — 論文整理

| 文件 | 說明 |
|------|------|
| [related_papers.md](papers/related_papers.md) | FL + LoRA 相關論文完整列表（含實驗設定、超參數比較表） |
| [personalized_fl_papers.md](papers/personalized_fl_papers.md) | Personalized FL 經典論文：FedPer、FedRep、LG-FedAvg、FedBN 等 |
| [FedALC-LoRA.md](papers/FedALC-LoRA.md) | FedALC-AP-* 家族方法設計（含 FedADC/FedLEASE/HiLoRA 比較、layer selection motivation） |
| [HiLoRA_detailed.md](papers/HiLoRA_detailed.md) | HiLoRA 方法詳解（三層 LoRA + clustering + 實驗設定） |
| [comparison_methods.md](papers/comparison_methods.md) | FedALC 比較方法定位：Tier 1-3 分類、場景比較、實作優先順序（寫於 rename 前，舊名 FedALC/FedALC-LWC = 今 FedALC-AP/FedALC-AP-LWC） |
| [task_vector_connection.md](papers/task_vector_connection.md) | LoRA ↔ Task Vector 連結：FedALC-AP-Multi 作為 federated task vector clustering 的理論基礎 |
| [fedadc_detailed.md](papers/fedadc_detailed.md) | *(計畫中)* FedADC (Computer Networks 2026) 方法詳解 + 跟 FedALC-AP-Multi 比較 |

## plans/ — 研究規劃

| 文件 | 說明 |
|------|------|
| [research_plan.md](plans/research_plan.md) | FedALC-LoRA 三階段研究規劃（Phase 1-3） |
| [next_steps.md](plans/next_steps.md) | 下一步行動（指向最新 action plan） |
| [fedalc_ap_multi_action_plan.md](plans/fedalc_ap_multi_action_plan.md) | **當前**：git commit 策略 + single-task 實驗計畫 (P0-P4) |
| [fedalc_lwc_design.md](plans/fedalc_lwc_design.md) | FedALC-AP-LWC 設計歷史（寫於 rename 前，原名 FedALC-LWC） |

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
| [clustering_methods_comparison.md](concepts/clustering_methods_comparison.md) | Related work 的 clustering 方法比較（AP/Agglomerative/Spectral + similarity metric） |
| [evaluation_metrics.md](concepts/evaluation_metrics.md) | 評估指標說明：資料切分、evaluate vs server accuracy、FedALC server_eval 問題 |
| [fedalc_naming_convention.md](concepts/fedalc_naming_convention.md) | FedALC-AP / FedALC-AP-LWC / FedALC-AP-Multi 命名規則：family 結構、variant 定位、ablation 設計 |

## experiments/ — 實驗數據

| 文件 | 說明 |
|------|------|
| [glue_datasets.md](experiments/glue_datasets.md) | GLUE 四個任務介紹、資料量、FL 分割情況 |
| [data_distribution_sst2.md](experiments/data_distribution_sst2.md) | SST-2 資料分佈分析（40 clients, α=0.5） |
| [data_distribution_30clients.md](experiments/data_distribution_30clients.md) | 全 GLUE 任務資料分佈分析（30 clients, α=0.5） |
| [fedavg_results.md](experiments/fedavg_results.md) | FedAvg 實驗結果（30 clients, 20 rounds, 全 GLUE 任務） |
| [fedavg_vs_fedsa_comparison.md](experiments/fedavg_vs_fedsa_comparison.md) | FedAvg vs FedSA-LoRA 完整比較（accuracy 表格 + 曲線圖） |
| [fedalc_lwc_results.md](experiments/fedalc_lwc_results.md) | FedALC-AP-LWC 實驗結果（寫於 rename 前，舊名 FedALC-LWC）：warm-up 失敗分析 |
| [fedalc_vs_lwc_comparison.md](experiments/fedalc_vs_lwc_comparison.md) | FedALC-AP vs AP-LWC 比較實驗設計（寫於 rename 前） |
| [fedalc_vs_lwc_no_warmup_results.md](experiments/fedalc_vs_lwc_no_warmup_results.md) | FedALC-AP vs AP-LWC（無 warm-up）SST-2 結果（寫於 rename 前） |
| [findings_summary.md](experiments/findings_summary.md) | 實驗 Findings 總結：8 個 key findings + 6 個優先度排序的下一步（寫於 rename 前） |

## 專案文件（根目錄）

| 文件 | 說明 |
|------|------|
| [README.md](../README.md) | 專案基本說明 |
| [CLAUDE.md](../CLAUDE.md) | Claude Code 使用指引、架構設計、常用指令 |
