# Plots 說明

## 共通設定

- Model: RoBERTa-large (355M) + LoRA r=8
- Clients: 30, Dirichlet α=0.5
- Local epochs: 1
- Optimizer: AdamW, cosine annealing, lr=0.001

## 目錄結構

```
plots/
├── r20_c30/          # 20 rounds, 30 clients
├── r50_c30/          # (planned) 50 rounds, 30 clients
└── README.md
```

### 命名規則

```
{比較對象}_{metric}_{task}.png
fedavg_vs_fedsa_accuracy_sst2.png
```

rounds / clients 資訊由目錄名表達，不重複放在檔名中。

## r20_c30/ — 20 Rounds 實驗

### Accuracy 曲線（evaluate/accuracy, weighted avg across clients）

| 檔案 | 內容 |
|------|------|
| `fedavg_vs_fedsa_accuracy_sst2.png` | SST-2: FedSA 0.952 > FedAvg 0.946, 接近 centralized 0.960 |
| `fedavg_vs_fedsa_accuracy_qnli.png` | QNLI: FedSA 0.924 > FedAvg 0.919, centralized 0.948 |
| `fedavg_vs_fedsa_accuracy_mnli.png` | MNLI: FedSA 0.765 << FedAvg 0.882, FedSA 未收斂 |
| `fedavg_vs_fedsa_accuracy_qqp.png` | QQP: FedSA 0.820 << FedAvg 0.892, FedSA 未收斂 |

### Server Loss 曲線（server/loss, global model on centralized valid set）

| 檔案 | 內容 |
|------|------|
| `fedavg_vs_fedsa_server_loss_sst2.png` | SST-2 server loss |
| `fedavg_vs_fedsa_server_loss_qnli.png` | QNLI server loss |
| `fedavg_vs_fedsa_server_loss_mnli.png` | MNLI server loss |
| `fedavg_vs_fedsa_server_loss_qqp.png` | QQP server loss |

### 主要發現

1. **SST-2 / QNLI (binary, 小~中資料量)**: FedSA 略勝 FedAvg (+0.5~0.6%)
2. **MNLI / QQP (3-class / 大資料量)**: FedSA 大幅落後 FedAvg (-7~12%)，20 rounds 不夠收斂
3. **需要更多 rounds**: 計畫跑 50 輪 FedSA 觀察 MNLI/QQP 能否追上

## r50_c30/ — 50 Rounds 實驗 (planned)

待補。重點觀察 MNLI/QQP 在更多 rounds 後 FedSA 是否能收斂。

## 指標說明

- **accuracy 曲線**: `evaluate/accuracy` = 每個 client 在 local valid split 上測，取 weighted average
  - FedAvg: 所有 client 用同一個 global model
  - FedSA: 每個 client 用個人化 model (global_A + own_B + own_classifier)
- **server loss**: `server/loss` = server 用 (agg_A + avg_B) 在完整 centralized valid set 上測
- **centralized 虛線**: centralized training 的 best epoch accuracy（upper bound）
