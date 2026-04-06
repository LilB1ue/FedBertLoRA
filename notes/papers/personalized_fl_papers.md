# Personalized Federated Learning 經典論文

> 聚焦於「哪些層該聚合、哪些層該留本地」這個問題的論文。

---

## 核心論文：Local Head / Personalization Layers

### 1. FedPer: Federated Learning with Personalization Layers
- **作者**: Arivazhagan et al.
- **年份**: 2019 (NeurIPS Workshop)
- **連結**: [arXiv:1912.00818](https://arxiv.org/abs/1912.00818)
- **核心觀點**: 最早提出把模型切成 base layers（聚合）+ personalization layers（留本地）。實驗發現把最後幾層留本地，在 non-IID 下效果顯著優於 FedAvg。
- **方法**: 手動指定哪些層是 base、哪些是 personalization，base 做 FedAvg，personalization 不上傳。

### 2. LG-FedAvg: Think Locally, Act Globally
- **作者**: Liang et al.
- **年份**: 2020
- **連結**: [arXiv:2001.01523](https://arxiv.org/abs/2001.01523)
- **核心觀點**: 把模型分成 global representation（encoder，聚合）+ local head（classifier，留本地）。跟 FedPer 方向相同但更明確地把 local head 定義為 classifier。
- **實驗**: 在 CIFAR-10/100 上證明 local head 比聚合整個模型好 5-10%。

### 3. FedRep: Exploiting Shared Representations for Personalized FL
- **作者**: Collins, Hassani, Mokhtari, Shakkottai
- **年份**: 2021 (ICML)
- **連結**: [arXiv:2102.07078](https://arxiv.org/abs/2102.07078)
- **核心觀點**: 提供了**理論證明**——在 heterogeneous 設定下，shared representation + local head 的收斂性優於全模型聚合。訓練分兩階段：先固定 head 更新 representation，再固定 representation 更新 head。
- **重要性**: 這篇是 personalized FL 中最常被引用的理論基礎。

### 4. FedBN: Federated Learning on Non-IID Features via Local Batch Normalization
- **作者**: Li et al.
- **年份**: 2021 (ICLR)
- **連結**: [arXiv:2102.07623](https://arxiv.org/abs/2102.07623)
- **核心觀點**: 連 BatchNorm 這種統計量層也不該聚合。不同 client 的 feature distribution 不同，BN 的 mean/variance 聚合後反而有害。
- **延伸**: 說明了 non-IID 的影響不只在 label 分佈，feature 分佈的差異也需要本地適應。

---

## LoRA 特化論文：A/B 矩陣的聚合策略

### 5. FedSA-LoRA: Selective Aggregation for Low-Rank Adaptation in FL
- **作者**: Guo et al.
- **年份**: 2025 (ICLR)
- **連結**: [arXiv:2410.01463](https://arxiv.org/abs/2410.01463) | [GitHub](https://github.com/Pengxin-Guo/FedSA-LoRA)
- **核心觀點**: A 矩陣學 general knowledge（聚合），B 矩陣學 client-specific knowledge（留本地），classifier 也留本地。
- **評估方式**: 每個 client 用 personalized model (global_A + own_B + own_classifier) 在 local validation split 上測，取 weighted average。
- **設定**: RoBERTa-large, 3 clients, Dirichlet α=0.5, 1000 rounds

### 6. FFA-LoRA: Federated Freeze-A LoRA
- **作者**: Sun et al.
- **年份**: 2024 (ICLR)
- **連結**: [arXiv:2403.12313](https://arxiv.org/abs/2403.12313)
- **核心觀點**: 凍結隨機初始化的 A 矩陣不訓練，只訓練和聚合 B 矩陣。通訊成本減半，但 A 不學習導致次優表現。是 FedSA-LoRA 的主要 baseline。

### 7. FedDPA: Dual-Personalizing Adapter for Federated Foundation Models
- **作者**: Yang et al.
- **年份**: 2024 (NeurIPS)
- **連結**: [arXiv:2403.19211](https://arxiv.org/abs/2403.19211) | [GitHub](https://github.com/Lydia-yang/FedDPA)
- **核心觀點**: 每個 client 配備兩套完整 LoRA — global adapter（FedAvg 聚合）+ local adapter（留本地）。推論時用 instance-wise cosine similarity 動態加權兩個 adapter。

---

## 聚類策略論文

### 8. FedADC: Federated Fine-Tuning with Alternating Device-to-Device Collaboration
- **年份**: 2026 (Computer Networks)
- **連結**: [ScienceDirect](https://doi.org/10.1016/j.comnet.2025.111931)
- **核心觀點**: 兩階段交替——Stage 1: similarity clustering + freeze A + 訓練 B；Stage 2: dissimilarity clustering + freeze B + 訓練 A。用 affinity propagation 聚類。

### 9. CFL: Clustered Federated Learning
- **作者**: Sattler et al.
- **年份**: 2020 (IEEE TNNLS)
- **連結**: [arXiv:1910.01991](https://arxiv.org/abs/1910.01991)
- **核心觀點**: 用 cosine similarity 對 client gradient 做層次聚類，相似的 client 分成同一群各自做 FedAvg。是 clustering-based FL 的經典基線。

---

## 理論觀察：層級越深，heterogeneity 越大

多篇論文共同的實驗發現：

```
底層 encoder (靠近 input):   學通用特徵     → 各 client 差不多 → 適合聚合
高層 encoder (靠近 output):  學任務相關特徵  → 開始分歧
classifier (最後一層):       直接對應 label  → 差異最大 → 不適合聚合
```

**原因**：
- Classifier 的 gradient 方向直接由 label 決定（`∝ predicted_prob - one_hot_label`）
- Encoder 的 gradient 是間接傳回的，經過多層 chain rule 稀釋，對 label 分佈敏感度較低
- Classifier 的每一行只被對應 label 的資料訓練，non-IID 下某些行幾乎沒資料可練

---

## 方法總覽

| 方法 | Encoder / LoRA-A | Classifier / LoRA-B | 聚類 | 特點 |
|------|-----------------|---------------------|------|------|
| FedAvg | 聚合 | 聚合 | 無 | Baseline |
| FedPer | 底層聚合 | 頂層留本地 | 無 | 最早的 layer-wise 分離 |
| LG-FedAvg | 聚合 | 留本地 | 無 | 明確定義 local head |
| FedRep | 聚合 | 留本地 | 無 | 理論證明收斂性 |
| FFA-LoRA | 凍結 | 聚合 | 無 | 通訊減半但 A 不學習 |
| FedSA-LoRA | 聚合 | 留本地 | 無 | A/B 都訓練，只聚合 A |
| FedDPA | 聚合 (global adapter) | 留本地 (local adapter) | 無 | 兩套完整 LoRA |
| FedADC | 交替聚合 | 交替聚合 | 有 | 動態聚類 + 交替訓練 |
| CFL | 聚合 (cluster 內) | 聚合 (cluster 內) | 有 | Gradient-based 層次聚類 |
