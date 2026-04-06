# 為什麼 Classifier 要留本地不聚合

## 核心問題

在 Federated Learning 中，為什麼很多論文主張把 classifier head 留在本地，不參與 FedAvg 聚合？

---

## Encoder vs Classifier 的角色差異

### Encoder 做什麼

Attention + FFN 疊 24 層，把原始 token 轉成融合了整句語意的 vector：

```
"This movie is great"
    │
    ▼ tokenize + embedding
每個 token 一個初始 vector（互相獨立，不知道上下文）
    │
    ▼ Layer 1~24: Attention + FFN 重複
    │
    ▼ Attention: 讓 token 之間交換資訊（"great 跟 movie 有關"）
    ▼ FFN:       在每個位置做非線性特徵轉換（"great 是正面語意"）
    │
    ▼
[CLS] token 的 vector 已經融合整句話的語意
    │
    ▼ pooling
pooled_output (1024-dim)
```

### Classifier 做什麼

一個矩陣乘法，把 1024 維的語意 vector 投影到 num_labels 維：

```
W = [ w_positive ]   ← 第 0 行 (1024 個數字): "positive 的特徵模板"
    [ w_negative ]   ← 第 1 行 (1024 個數字): "negative 的特徵模板"

logits[0] = w_positive · pooled_output = positive 分數
logits[1] = w_negative · pooled_output = negative 分數
```

W 的每一行是 1024 維空間裡的一個「方向」，代表「feature 長這樣就是這個 label」。

---

## 為什麼 Non-IID 對 Classifier 傷害更大

### Encoder：不管 label 是什麼，都在學同一件事

```
positive 句子: "This movie is great"
  → attention 學到 great 修飾 movie
  → FFN 學到 great 是正面語意

negative 句子: "This movie is terrible"
  → attention 學到 terrible 修飾 movie
  → FFN 學到 terrible 是負面語意
```

兩種 label 的資料都在教 encoder「理解英文句子的結構和語意」。
Client A 只有 positive 資料 → encoder 也在學理解英文。
Client B 只有 negative 資料 → encoder 也在學理解英文。
方向差不多，聚合起來不會互相抵消。

### Classifier：每一行只被對應 label 的資料訓練

```
Client A (90% positive):
  → w_positive 被充分訓練，學得很好
  → w_negative 幾乎沒看到資料，學得很差

Client B (90% negative):
  → w_positive 學得差
  → w_negative 學得好

聚合後:
  w_positive = (好 + 差) / 2 = 半成品
  w_negative = (差 + 好) / 2 = 半成品
  → 兩行都被拉向平庸
```

### 從 Gradient 角度理解

```
Loss = CrossEntropy(softmax(classifier(encoder(x))), label)

反向傳播:
  classifier 的 gradient ∝ (predicted_prob - one_hot_label)
                            ──────────────────────────────
                            方向直接由 label 決定

  encoder 的 gradient = classifier gradient × chain rule 傳回來
                        ──────────────────────────────────────
                        經過 classifier weight 矩陣「稀釋」，
                        對 label 分佈的敏感度低很多
```

Classifier 的梯度方向直接由 label 決定 → non-IID 下各 client 方向差異最大。
Encoder 的梯度是間接傳回的 → 經過多層 chain rule 稀釋，方向差異較小。

---

## 類比

> 所有人戴同一副眼鏡看世界（shared encoder），但每個人根據自己的經驗做判斷（local classifier）。
>
> 眼鏡可以共享 — 「看清楚東西」是通用能力。
> 判斷標準不能平均 — 每個人面對的 label 分佈不同。
>
> 正是因為 encoder 被聚合了（大家共享同一個 feature space），
> classifier 才能安全地留本地（在共享空間裡各自畫決策邊界）。

---

## 經驗法則：越靠近 output 的層，heterogeneity 越大

```
底層 encoder (靠近 input):   學通用語法、語意特徵     → 各 client 差不多 → 適合聚合
高層 encoder (靠近 output):  學任務相關特徵           → 開始分歧
classifier (最後一層):        直接對應 label 分佈      → 差異最大 → 不適合聚合
```

---

## 經典論文

| 論文 | 核心觀點 |
|------|---------|
| **FedPer** (Arivazhagan et al., 2019) | 最早提出 base layers 聚合 + personalization layers 留本地 |
| **LG-FedAvg** (Liang et al., 2020) | 模型切成 global representation + local head，實驗證明 local head 更好 |
| **FedRep** (Collins et al., 2021, ICML) | 理論分析：shared representation + local head 在 heterogeneous 下收斂更好 |
| **FedBN** (Li et al., 2021) | 連 BatchNorm 統計量層也不該聚合 |
| **FedSA-LoRA** (Guo et al., 2025, ICLR) | LoRA 版本：A 聚合 + B 和 classifier 留本地 |

---

## 在 FedSA-LoRA 中的對應

| 參數 | 角色 | 聚合？ | 原因 |
|------|------|--------|------|
| LoRA A | 投影下降 (靠近 input 側) | 聚合 | 學 shared knowledge |
| LoRA B | 投影回升 (靠近 output 側) | 留本地 | Client-specific adaptation |
| Classifier | 最終決策層 | 留本地 | 直接受 label 分佈影響 |
| Base model W | 預訓練權重 | 凍結 | 不參與訓練 |
