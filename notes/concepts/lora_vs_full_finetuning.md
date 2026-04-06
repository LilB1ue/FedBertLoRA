# LoRA vs 全參數微調：架構圖解

## 全參數微調

```
Input tokens: "This movie is great"
         │
         ▼
┌─────────────────────────────────┐
│     RoBERTa Encoder (24層)       │
│                                  │
│  Attention: Q, K, V, Dense       │  ◄── 全部參數都更新 ✏️
│  FFN: intermediate, output       │
│                                  │
│  每層的 W 直接被 gradient 修改    │
└─────────────────────────────────┘
         │
         │  pooled_output (1024-dim vector)
         ▼
┌─────────────────────────────────┐
│        Classifier Head           │  ◄── 這就是 classifier ✏️
│                                  │
│  Linear(1024→1024) + tanh        │  classifier.dense
│           │                      │
│  Linear(1024→num_labels)         │  classifier.out_proj
└─────────────────────────────────┘
         │
         │  ← 這裡出來的就是 logits (num_labels 維)
         │     例如 SST-2: [0.8, -1.2] (2維)
         ▼
    softmax (無參數)
         │
         ▼
    機率: [0.88, 0.12]
         │
         ▼
    預測: positive
```

## LoRA 微調

```
Input tokens: "This movie is great"
         │
         ▼
┌──────────────────────────────────────────┐
│        RoBERTa Encoder (24層)             │
│                                           │
│  每層 Attention:                          │
│  ┌──────────────────────────────────┐     │
│  │ Q: output = W_q·x + B_q·A_q·x   │     │
│  │            ─────   ───────────   │     │
│  │            凍結 🧊   LoRA ✏️      │     │
│  │                                   │     │
│  │ K: output = W_k·x + B_k·A_k·x   │     │
│  │ V: output = W_v·x + B_v·A_v·x   │     │
│  │ Dense: 同上                       │     │
│  └──────────────────────────────────┘     │
│                                           │
│  W (原始權重): 凍結 🧊 不動               │
│  A (rank↓):   可訓練 ✏️  ← FedSA 聚合這個 │
│  B (rank↑):   可訓練 ✏️  ← FedSA 留本地   │
└──────────────────────────────────────────┘
         │
         │  pooled_output (1024-dim vector)
         ▼
┌──────────────────────────────────────────┐
│        Classifier Head                    │  ◄── 可訓練 ✏️
│                                           │
│  Linear(1024→1024) + tanh                 │  classifier.dense
│           │                               │     (weight: 1024×1024)
│  Linear(1024→num_labels)                  │  classifier.out_proj
│                                           │     (weight: num_labels×1024)
│                                           │
│  FedSA 論文: 留本地 (跟 B 一樣)            │
└──────────────────────────────────────────┘
         │
         │  ← logits (num_labels 維)
         ▼
    softmax (無參數，不需要訓練)
         │
         ▼
    機率 → 預測
```

## 參數數量對比

```
                        全參數微調          LoRA 微調
                        ──────────          ─────────
RoBERTa Encoder (W)     ~355M ✏️           ~355M 🧊 (凍結)
LoRA A matrices         不存在              ~2.3M ✏️
LoRA B matrices         不存在              ~2.3M ✏️
Classifier head         ~1M ✏️             ~1M ✏️
                        ──────────          ─────────
可訓練參數              ~356M               ~4.6M (1.3%)
```

## 名詞釐清

| 名詞 | 是什麼 | 有參數嗎 |
|------|--------|---------|
| Encoder | RoBERTa 的 24 層 transformer | 有（~355M），LoRA 時凍結 |
| LoRA A | 投影下降矩陣 (hidden_dim → rank) | 有，可訓練 |
| LoRA B | 投影回升矩陣 (rank → hidden_dim) | 有，可訓練 |
| Classifier | 分類頭，兩層 Linear | 有（~1M），永遠要訓練 |
| Logits | classifier 的輸出，num_labels 維的原始分數 | 無，只是中間值 |
| Softmax | 把 logits 轉成機率 | 無 |

## FedSA-LoRA 中各參數的聚合方式

| 參數 | FedAvg | FedSA-LoRA | FFA-LoRA |
|------|--------|-----------|----------|
| LoRA A | 聚合 | 聚合 (shared knowledge) | 凍結不動 |
| LoRA B | 聚合 | **留本地** (client-specific) | 聚合 |
| Classifier | 聚合 | **留本地** (跟 B 一樣) | 聚合 |

> **為什麼 classifier 要留本地？**
> 在 non-IID 設定下（Dirichlet α=0.5），不同 client 的 label 分佈差異大。
> 例如 client 1 幾乎全是 positive，client 9 幾乎全是 negative。
> classifier 會往各自的分佈偏，聚合反而互相干擾。
>
> **為什麼預訓練模型沒有 classifier？**
> 預訓練的 RoBERTa 是做 Masked Language Modeling（遮字預測），
> 沒有下游分類頭。微調時（不管 LoRA 或全參數）classifier 都是從零初始化的新參數。
