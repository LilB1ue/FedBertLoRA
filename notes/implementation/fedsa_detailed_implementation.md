# FedSA-LoRA 詳細實作與評估協議

> 來源：[FedSA-LoRA (ICLR 2025)](https://arxiv.org/abs/2410.01463) + [官方 GitHub](https://github.com/Pengxin-Guo/FedSA-LoRA)（基於 FederatedScope 框架）

---

## 實驗設定

| 參數 | 值 |
|------|-----|
| 模型 | RoBERTa-large (NLU), LLaMA-3-8B (NLG) |
| GLUE 任務 | SST-2, QNLI, MNLI, QQP, RTE |
| 非 GLUE | GSM8K, CodeSearchNet |
| Clients | **3**（ablation: 10, 20, 100） |
| Non-IID | Dirichlet **α=0.5**（也測 α=1.0 和 IID） |
| Optimizer | SGD, lr=0.02 |
| LoRA | r=8, α=16, target=Q,V |
| FL Rounds | 1000 |
| Local epochs | 10 |
| Batch size | 128 |

---

## 核心機制：哪些參數留本地

論文的 FederatedScope config：

```yaml
personalization:
  local_param: ['lora_B', 'classifier']
```

含義：
- **lora_A**: 聚合到 server（FedAvg weighted average）
- **lora_B**: **留本地**，不上傳、不聚合
- **classifier**: **留本地**，分類頭也不聚合（因為不同 client 的 label 分佈不同，classifier 也有 client-specific 偏好）

### 什麼是 classifier？

RoBERTa 用於 GLUE 分類任務時，最上層有一個 classification head（`roberta.classifier`），將 pooled output 映射到 `num_labels` 維度。在 FedSA-LoRA 中，這個分類頭跟 lora_B 一樣被視為 client-specific，不參與聚合。

---

## 每輪流程

```
Round N:

1. Server → Clients: 廣播 global 參數（只有 lora_A + 其他共用參數）

2. Client 端合併:
   收到的 global_A  →  替換本地的 A
   本地的 lora_B    →  保留不動
   本地的 classifier →  保留不動
   結果: 每個 client 持有 (global_A + own_B + own_classifier)

3. Client 本地訓練:
   用合併後的 personalized model 在 local train split 上訓練 10 epochs
   A、B、classifier 都會被更新

4. Client → Server: 只上傳 lora_A（B 和 classifier 不傳）

5. Server 聚合:
   aggregated_A = weighted_average(所有 client 的 A, 按樣本數加權)

6. 評估（見下方詳細說明）
```

---

## 評估協議（關鍵！論文沒明說）

### 論文怎麼測的

**不是**用一個 global model 在 centralized validation set 上測。

實際做法：

1. Server 廣播 global 參數給所有 client
2. 每個 client 合併為 personalized model: **global_A + own_B + own_classifier**
3. 每個 client 在**自己的 local validation split** 上評估（Dirichlet 分割後的驗證集）
4. Server 收集結果，算 **weighted average**（按樣本數加權）

```python
# FederatedScope 輸出格式
{
    'Results_weighted_avg': {'val_accuracy': 0.934},   # ← 論文報告這個
    'Results_avg': {'val_accuracy': 0.921},             # 簡單平均
    'Results_fairness': {
        'val_accuracy_std': 0.015,
        'val_accuracy_min': 0.908,
        'val_accuracy_max': 0.952,
    }
}
```

等效計算：
```
reported_accuracy = sum(client_i_correct) / sum(client_i_total)
```

### 為什麼這很重要

如果用 `(global_A + averaged_B)` 在 centralized validation set 上測：
- 把 B 平均掉了 → 失去 personalization 的意義
- 本質上等於測一個 FedAvg 等效模型
- **無法展現 FedSA 保留 B 矩陣的優勢**

FedSA 的 global model（avg_B）表現跟 FedAvg 可能差不多甚至更差，但 personalized model（own_B）才是重點。

---

## 與我們實作的差異

| 面向 | 論文（FederatedScope） | 目前實作（Flower） |
|------|----------------------|-------------------|
| local_param | `['lora_B', 'classifier']` | 只處理 `lora_B`，classifier 有聚合 |
| 評估方式 | 每個 client 用 personalized model 在 local valid split 上測 | server 用 `(global_A + avg_B)` 在 centralized valid set 上測 |
| 報告 metric | per-client accuracy 的 weighted average | centralized accuracy |
| Clients | 3（主實驗） | 30 |
| Local epochs | 10 | 1 |
| Optimizer | SGD, lr=0.02 | AdamW, lr=0.001 |
| Rounds | 1000 | 20 |

### 需要修改的項目

1. **評估改為 personalized evaluation**: `configure_evaluate` 要跟 `configure_fit` 一樣，為每個 client 組裝 `(global_A + own_B)`，讓 client 在自己的 local validation split 上測
2. **Classifier 也應留本地**: 在 `_separate_a_b()` 中把 classifier 參數歸類到跟 B 一樣的處理方式
3. **Server eval 的角色**: 可保留作為輔助參考（global model 表現），但主要 metric 應改用 client-side weighted average

---

## 參考

- 論文: https://arxiv.org/abs/2410.01463
- 官方代碼: https://github.com/Pengxin-Guo/FedSA-LoRA
- 框架: FederatedScope（阿里巴巴）
- 關鍵 config: `federatedscope/glue/yamls/fedsa-lora.yaml`
