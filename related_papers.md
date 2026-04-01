# Federated Learning + LoRA Selective Aggregation 相關論文

## 核心論文

### 1. FedSA-LoRA: Selective Aggregation for Low-Rank Adaptation in Federated Learning
- **發表**: ICLR 2025
- **連結**: [arXiv](https://arxiv.org/abs/2410.01463) | [GitHub](https://github.com/Pengxin-Guo/FedSA-LoRA)
- **簡介**: 發現 A 矩陣學習 general knowledge、B 矩陣學習 client-specific knowledge。提出只聚合 A 矩陣到 server，B 矩陣留在本地。擴展到 rsLoRA 和 VeRA 變體。
- **實驗設定**:
  - 模型: RoBERTa-large (NLU), LLaMA-3-8B (NLG)
  - GLUE 子任務: SST-2, QNLI, MNLI, QQP, RTE
  - 非 GLUE: GSM8K, CodeSearchNet
  - Clients: 3（ablation: 10, 20, 100）
  - Non-IID: Dirichlet α=0.5（也測 α=1.0 和 IID）
  - Optimizer: SGD; LR=0.02
  - LoRA: r=8, α=16, target=Q,V; rounds=1000; local epochs=10; batch=128

### 2. FedADC: Federated Fine-Tuning on Heterogeneous Data with Alternating Device-to-Device Collaboration
- **發表**: Computer Networks, 2026
- **連結**: [ScienceDirect](https://doi.org/10.1016/j.comnet.2025.111931)
- **簡介**: 兩階段交替訓練 — Stage 1: similarity clustering + freeze A + 訓練 B（個性化適應）；Stage 2: dissimilarity clustering + freeze B + 訓練 A（全域泛化）。使用 affinity propagation 聚類 + MAB/UCB 自適應排程。
- **實驗設定**:
  - 模型: RoBERTa-base, DeBERTa-large
  - GLUE（具體子任務需查全文，論文在 Elsevier paywall 後無 preprint）
  - Clients: 80 devices（實體異質平台，非模擬）
  - Non-IID: 實體環境自然異質性

---

## 密切相關論文

### 3. FFA-LoRA: Federated Freeze-A LoRA
- **發表**: ICLR 2024
- **連結**: [arXiv](https://arxiv.org/abs/2403.12313)
- **簡介**: Freeze 隨機初始化的 A 矩陣，只訓練和聚合 B 矩陣，將通訊成本減半。缺點是固定 A 會削弱 LoRA 學習能力，導致次優表現。是 FedSA-LoRA 的主要 baseline。
- **實驗設定**:
  - 模型: RoBERTa-large (NLU), LLaMA (NLG), ViT (CV)
  - GLUE 子任務: SST-2, QNLI, MNLI, QQP
  - 非 GLUE: GSM8K, Food-101 (ViT)
  - Clients: 3
  - Non-IID: 固定 label 比例分配（二分類: [0.1,0.9]/[0.9,0.1]/[0.5,0.5]；MNLI 三分類類似）
  - LoRA: r=8, α=16, target=attention + FFN
  - Optimizer: SGD (DP-SGD for privacy); rounds=1000; local steps=10; batch=200

### 4. RoLoRA: Robust Federated Finetuning of Foundation Models via Alternating Minimization of LoRA
- **發表**: arXiv 2024
- **連結**: [arXiv](https://arxiv.org/abs/2409.02346)
- **簡介**: 奇偶輪交替更新 A 和 B 矩陣來緩解聚合偏差。比 FFA-LoRA 更穩健，因為兩個矩陣都有機會被更新，但在高 non-IID 下仍有效能下降。
- **實驗設定**:
  - 模型: RoBERTa-large; DeBERTa-XLarge (附錄)
  - GLUE 子任務: SST-2, QNLI, MNLI, QQP, RTE
  - Clients: 3（ablation: 20, 50）
  - Non-IID: 非重疊 sample split（非 Dirichlet、非 label-based proportion）
  - LoRA: r∈{1,2,4,8}, target=Q,V; rounds=500 (RTE=200); local epochs=20; batch=32 (SST-2=64)

### 5. ADF-LoRA: Alternating Low-Rank Aggregation for Decentralized Federated Fine-Tuning
- **發表**: arXiv 2025
- **連結**: [arXiv](https://arxiv.org/abs/2511.18291)
- **簡介**: 將交替更新 A/B 的方法擴展到去中心化 (peer-to-peer) 聯邦學習。每輪只同步一個矩陣，並混合兩個矩陣以維持一致的參數狀態。
- **實驗設定**:
  - 模型: RoBERTa-large
  - GLUE 子任務: SST-2, QNLI, MNLI, QQP
  - Clients: 10
  - Non-IID: 固定 label 比例分配（同 FFA-LoRA 方式）
  - LoRA: r=8, α=16, dropout=0.1, target=Q,V; rounds=150; local steps=20; batch=32
  - Optimizer: AdamW; LR∈{5e-4, 1e-3, 2e-3, 5e-3}

### 6. LoRA-FAIR: Federated LoRA Fine-Tuning with Aggregation and Initialization Refinement
- **發表**: ICCV 2025
- **連結**: [arXiv](https://arxiv.org/abs/2411.14961) | [GitHub](https://github.com/jmbian/LoRA-FAIR)
- **簡介**: 同時解決 server 端聚合偏差 (aggregation bias) 和 client 端初始化滯後 (initialization lag) 兩個問題。引入校正項改善聚合效率和準確度。
- **⚠️ 純 CV 論文，無 NLP/GLUE 實驗**
- **實驗設定**:
  - 模型: ViT-B/16, MLP-Mixer-B/16 (pretrained on ImageNet-21k)
  - 資料集: DomainNet (top 100 categories), NICO++
  - Clients: 6 (feature non-IID) 或 30 (feature+label non-IID)
  - Non-IID: domain-based 自然分割 + Dirichlet α=0.5 (label 異質性)

### 7. SFed-LoRA: Stabilized Fine-Tuning with LoRA in Federated Learning (Scaling Factor 分析)
- **發表**: arXiv 2025
- **連結**: [arXiv](https://arxiv.org/abs/2603.08058)
- **簡介**: 分析 client 數量和 LoRA rank 對 FL 中 LoRA 微調穩定性的影響，提出 scaling factor γ_z = α·√(N/r) 來緩解 gradient collapse。
- **實驗設定**:
  - 模型: RoBERTa-large (GLUE), LLaMA-2-7B (NLG)
  - GLUE 子任務: MNLI-m only
  - 非 GLUE: Alpaca, GSM8K
  - Clients: N∈{5, 10, 15, 20} (GLUE); 33 (Alpaca/GSM8K, IID)
  - Non-IID: Dirichlet α=0.5 (GLUE); IID (Alpaca/GSM8K)
  - LoRA: r∈{4,8,32,128,512}, 極端測試 r=2048; rounds=100; local steps=10

### 8. PF2LoRA: Personalized Federated Fine-Tuning via Two-Level LoRA (Automatic Rank Learning)
- **發表**: arXiv 2025
- **連結**: [arXiv](https://arxiv.org/abs/2503.03920)
- **簡介**: 提出自動 rank 學習的兩層 LoRA 方法 — 第一層 {A,B} 學習所有 client 共用的 adapter (FedAvg)，第二層 {C_k,D_k} 留本地促進個性化。W_k = W_0 + BA + D_k·C_k。
- **⭐ Personalization 導向**：評估 per-client 個性化表現
- **實驗設定**:
  - 模型: RoBERTa-base（主實驗）, RoBERTa-large（附錄）, DeBERTa-v3, GPT-2 Medium/XL
  - GLUE 子任務: CoLA, MNLI, SST-2, QQP, QNLI
  - 非 GLUE: SQuAD v1/v2, WebNLG, E2E
  - Clients: 8 (GLUE), 4 (SQuAD)
  - Non-IID: Label-sorting (s 參數, s∈[0,1])，非 Dirichlet
  - LoRA: common r=8, client-specific r̃=2; rounds: CoLA=50, SST-2/QNLI=100, MNLI/QQP=300
  - Optimizer: SGD (client-specific) + AdamW (common); communication interval=10 local steps

### 9. FedDPA: Dual-Personalizing Adapter for Federated Foundation Models
- **發表**: NeurIPS 2024
- **連結**: [arXiv](https://arxiv.org/abs/2403.19211) | [GitHub](https://github.com/Lydia-yang/FedDPA)
- **簡介**: 每個 client 配備兩套完整 LoRA — global adapter (FedAvg 聚合) + local adapter (留本地)。推論時用 instance-wise cosine similarity 動態加權兩個 adapter。與 FedSA-LoRA 的差異：FedSA-LoRA 在單一 LoRA 內分 A/B，FedDPA 是兩套完整 LoRA。
- **⭐ Personalization 導向**：instance-wise 動態加權 global/local adapter
- **實驗設定**:
  - 模型: LLaMA-7B（⚠️ 非 RoBERTa，非 GLUE）
  - 資料集: FLAN（instruction-tuning 集合，8 個不同 NLP 任務各分給一個 client）
  - Clients: 8（主實驗），40（ablation）
  - Non-IID: natural task heterogeneity（每 client 一個不同任務），非 Dirichlet
  - LoRA: r=8, target=Q,V; rounds=20; local epochs=10; batch=64; lr=0.0003

### 10. pFedLoRA: Model-Heterogeneous Personalized Federated Learning with LoRA Tuning
- **發表**: arXiv 2023
- **連結**: [arXiv](https://arxiv.org/abs/2310.13283)
- **簡介**: 每個 client 有不同架構的 CNN + 同質的小 adapter（LoRA-inspired 但非標準 PEFT LoRA）。只聚合 adapter，本地模型留 client 端。Personalization 來自模型架構異質性。
- **⚠️ 純 CV 論文，非標準 LoRA，與 NLP 場景無直接可比性**
- **實驗設定**:
  - 模型: 5 種異質 CNN（非預訓練語言模型）
  - 資料集: CIFAR-10, CIFAR-100（無 NLP/GLUE）
  - Clients: 10 / 50 / 100
  - Non-IID: label sorting（每 client 只拿部分 class）

---

## Clustering 相關論文

### 11. FedLEASE: Adaptive LoRA Experts Allocation and Selection
- **發表**: NeurIPS 2025
- **連結**: [arXiv](https://arxiv.org/abs/2509.15087)
- **簡介**: 用 LoRA-B cosine similarity 進行 client clustering，silhouette 分析找最佳 K。實驗中 16 clients × 4 GLUE 任務 → 最佳 K=4（反映任務差異，非 label 差異）。B 矩陣 similarity heatmap 顯示 block-diagonal structure。
- **與 B 矩陣 clustering 分析直接相關**
- **實驗設定**:
  - 模型: 需確認
  - GLUE 子任務: 4 個任務（跨任務 clustering，非單任務內 non-IID）
  - Clients: 16（每任務 4 個）

### 12. CORNFLQS: Robust Clustered Federated Learning for Non-IID
- **發表**: arXiv 2025
- **連結**: [arXiv](https://arxiv.org/abs/2510.03380)
- **簡介**: CV 上 CIFAR-10（10 classes）最佳 cluster 數是 4，**不等於 label 數**。Overestimate cluster 數比 underestimate 影響小。
- **⚠️ CV 論文**
- **實驗設定**:
  - 資料集: CIFAR-10, CIFAR-100 等 CV 資料集
  - 結論: 最佳 K 與 label 數無直接對應

### 13. HiLoRA: Hierarchical LoRA for Personalized Federated Learning
- **發表**: arXiv 2025
- **連結**: [arXiv](https://arxiv.org/abs/2603.02785)
- **簡介**: 三層 hierarchy — root (global) → cluster → leaf (client)。LoRA-Subspace Adaptive Clustering 機制，按 subspace similarity 分群。
- **實驗設定**:
  - 資料集: CIFAR-100（NLP 結果不確定）

### 14. FL-TAC: Task-Specific Adapter Clustering for Federated Learning
- **發表**: arXiv 2024
- **連結**: [arXiv](https://arxiv.org/abs/2404.15384)
- **簡介**: Server-side adapter clustering on GLUE。Task-aware clustering。報告 QNLI +11.7 pts, QQP +29.7 pts。
- **實驗設定**:
  - GLUE + CIFAR-10/100
  - 跨任務 clustering（非單任務內 non-IID clustering）

### 15. Federated Low-Rank Adaptation for Foundation Models: A Survey
- **發表**: IJCAI 2025
- **連結**: [arXiv](https://arxiv.org/abs/2505.13502)
- **簡介**: 全面綜述 Federated LoRA 的研究現況，涵蓋聚合策略、通訊效率、隱私保護、異質性處理等面向。適合作為背景閱讀和相關工作整理。

---

## 開放研究問題

### LoRA-B 矩陣 clustering 與 label 數的關係
- **文獻現狀**: 無人在 NLP 單任務 + Dirichlet non-IID 下直接研究 B 矩陣自然分群數是否與 label 數相關
- **CV 觀察**: CORNFLQS 在 CIFAR-10 (10 classes) 上最佳 K=4，不等於 label 數
- **NLP 觀察**: FedLEASE 在 4 GLUE 任務上 K=4，但反映任務差異非 label 差異
- **假說 A**: NLP 文本分類中，同 label 內特徵多樣性高（如 positive sentiment 可來自演技/劇情/攝影等不同面向），B 矩陣差異可能不由 label 主導 → cluster 數與 label 數無關
- **假說 B**: Dirichlet 在二分類上的主要變異軸是 label 比例（一維 scalar），B 矩陣的變異可能被 label ratio 主導 → cluster ≈ label 數
- **驗證方式**: 收集訓練後 B 矩陣 → pairwise cosine similarity heatmap → silhouette analysis → 對照各 client label 比例

---

## 實驗設定比較總覽

### 模型與資料集

| 論文 | NLP 模型 | GLUE 子任務 | 非 GLUE 資料集 |
|---|---|---|---|
| FedSA-LoRA | RoBERTa-large, LLaMA-3-8B | SST-2, QNLI, MNLI, QQP, RTE | GSM8K, CodeSearchNet |
| FFA-LoRA | RoBERTa-large, LLaMA | SST-2, QNLI, MNLI, QQP | GSM8K, Food-101(ViT) |
| RoLoRA | RoBERTa-large, DeBERTa-XL | SST-2, QNLI, MNLI, QQP, RTE | — |
| ADF-LoRA | RoBERTa-large | SST-2, QNLI, MNLI, QQP | — |
| LoRA-FAIR | ViT, MLP-Mixer (⚠️ CV only) | — | DomainNet, NICO++ |
| SFed-LoRA | RoBERTa-large, LLaMA-2-7B | MNLI-m | Alpaca, GSM8K |
| PF2LoRA | RoBERTa-base/large, GPT-2 | CoLA, MNLI, SST-2, QQP, QNLI | SQuAD, WebNLG, E2E |
| FedDPA | LLaMA-7B (⚠️ 非 GLUE) | — | FLAN (8 NLP tasks) |
| pFedLoRA | CNN (⚠️ CV only, 非標準 LoRA) | — | CIFAR-10, CIFAR-100 |
| FedADC | RoBERTa-base, DeBERTa-large | GLUE（需查全文） | 需查全文 |
| FedLEASE | 需確認 | 4 GLUE tasks (跨任務) | — |
| FL-TAC | 需確認 | GLUE (跨任務) | CIFAR-10/100 |

### Clients 與 Non-IID 方法

| 論文 | Clients | Non-IID 方法 | Personalization |
|---|---|---|---|
| FedSA-LoRA | 3 (ablation: 10/20/100) | Dirichlet α=0.5 | 否（評估全域模型） |
| FFA-LoRA | 3 | 固定 label 比例分配 | 否 |
| RoLoRA | 3 (ablation: 20/50) | 非重疊 sample split | 否 |
| ADF-LoRA | 10 | 固定 label 比例分配 | 否 |
| LoRA-FAIR | 6 或 30 | Domain-based + Dirichlet | 否 |
| SFed-LoRA | 5–20 | Dirichlet α=0.5 | 否 |
| PF2LoRA | 8 | Label-sorting (s 參數) | **是** |
| FedDPA | 8 (ablation: 40) | Natural task heterogeneity | **是** |
| pFedLoRA | 10/50/100 | Label sorting | **是**（模型異質性） |
| FedADC | 80（實體平台） | 自然異質性 | 否 |
| FedLEASE | 16 (4 tasks × 4) | 跨任務異質性 | 否（clustering 分析） |

### 訓練超參數

| 論文 | Optimizer | LoRA rank | LoRA target | Rounds | Local epochs/steps | Batch |
|---|---|---|---|---|---|---|
| FedSA-LoRA | SGD, lr=0.02 | r=8, α=16 | Q, V | 1000 | 10 epochs | 128 |
| FFA-LoRA | SGD | r=8, α=16 | attention + FFN | 1000 | 10 steps | 200 |
| RoLoRA | — | r∈{1,2,4,8} | Q, V | 500 (RTE=200) | 20 epochs | 32 |
| ADF-LoRA | AdamW | r=8, α=16 | Q, V | 150 | 20 steps | 32 |
| SFed-LoRA | — | r∈{4,8,32,128,512} | — | 100 | 10 steps | — |
| PF2LoRA | SGD + AdamW | common r=8, local r̃=2 | — | 50–300 | 10 steps | — |
| FedDPA | AdamW(?) | r=8 | Q, V | 20 | 10 epochs | 64 |

---

## 方法分類總覽

| 方法 | A 矩陣 | B 矩陣 | 聚類 | 特點 |
|---|---|---|---|---|
| **LoRA (FedAvg)** | 聚合 | 聚合 | 無 | Baseline |
| **FFA-LoRA** | Freeze | 聚合 | 無 | 通訊減半，但 A 不學習 |
| **FedSA-LoRA** | 聚合 | 本地保留 | 無 | A+B 都訓練，只聚合 A |
| **RoLoRA** | 奇數輪聚合 | 偶數輪聚合 | 無 | 交替更新 |
| **ADF-LoRA** | 交替同步 | 交替同步 | 無 | 去中心化版本 |
| **FedADC** | Dissim. cluster 聚合 | Sim. cluster 聚合 | 有 | 兩階段交替 + 動態聚類 |
| **LoRA-FAIR** | 聚合+校正 | 聚合+校正 | 無 | 解決聚合偏差+初始化滯後 (CV only) |
| **PF2LoRA** | 聚合 (共用層) | 聚合 (共用層) + 本地層 | 無 | 兩層 LoRA, personalization |
| **FedDPA** | 聚合 (global LoRA) | 聚合 (global LoRA) | 無 | 兩套完整 LoRA (global+local), personalization |
| **pFedLoRA** | — | — | 無 | 異質 CNN + 同質 adapter, CV only |
| **FedLEASE** | 按 cluster 聚合 | 按 cluster 聚合 | **有** | LoRA-B similarity clustering, silhouette 選 K |
| **FL-TAC** | 按 cluster 聚合 | 按 cluster 聚合 | **有** | Task-specific adapter clustering |
| **HiLoRA** | 階層式聚合 | 階層式聚合 | **有** | 三層 hierarchy (global→cluster→client) |
