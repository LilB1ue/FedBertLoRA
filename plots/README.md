# Plots 說明

## 實驗設定

- Model: RoBERTa-large (355M) + LoRA r=8
- Clients: 30, Dirichlet α=0.5
- Rounds: 20
- Local epochs: 1
- Optimizer: AdamW, cosine annealing, lr=0.001

## 檔案列表

### Accuracy 曲線（evaluate/accuracy, weighted avg across clients）

| 檔案 | 內容 |
|------|------|
| `fedavg_vs_fedsa_r20_c30_accuracy_sst2.png` | SST-2: FedSA 0.952 > FedAvg 0.946, 接近 centralized 0.960 |
| `fedavg_vs_fedsa_r20_c30_accuracy_qnli.png` | QNLI: FedSA 0.924 > FedAvg 0.919, centralized 0.948 |
| `fedavg_vs_fedsa_r20_c30_accuracy_mnli.png` | MNLI: FedSA 0.765 << FedAvg 0.882, FedSA 未收斂 |
| `fedavg_vs_fedsa_r20_c30_accuracy_qqp.png` | QQP: FedSA 0.820 << FedAvg 0.892, FedSA 未收斂 |

### Server Loss 曲線（server/loss, global model on centralized valid set）

| 檔案 | 內容 |
|------|------|
| `fedavg_vs_fedsa_r20_c30_server_loss_sst2.png` | SST-2 server loss |
| `fedavg_vs_fedsa_r20_c30_server_loss_qnli.png` | QNLI server loss |
| `fedavg_vs_fedsa_r20_c30_server_loss_mnli.png` | MNLI server loss |
| `fedavg_vs_fedsa_r20_c30_server_loss_qqp.png` | QQP server loss |

## 命名規則

```
{比較對象}_{rounds}_{clients}_{metric}_{task}.png
fedavg_vs_fedsa_r20_c30_accuracy_sst2.png
```

## 主要發現 (20 rounds)

1. **SST-2 / QNLI (binary, 小~中資料量)**: FedSA 略勝 FedAvg (+0.5~0.6%)
   - FedSA 的個人化 B 矩陣在簡單任務上有效
   - 兩者都接近 centralized baseline

2. **MNLI / QQP (3-class / 大資料量)**: FedSA 大幅落後 FedAvg (-7~12%)
   - FedSA best round 在 R19，曲線還在爬升，**20 rounds 不夠收斂**
   - FedAvg best round 在 R12，已收斂
   - FedSA 的 std 極大（MNLI=0.206），跨 client 差異嚴重

3. **需要更多 rounds**: 計畫跑 50 輪 FedSA 觀察 MNLI/QQP 能否追上

## 指標說明

- **accuracy 曲線**: `evaluate/accuracy` = 每個 client 在 local valid split 上測，取 weighted average
  - FedAvg: 所有 client 用同一個 global model
  - FedSA: 每個 client 用個人化 model (global_A + own_B + own_classifier)
- **server loss**: `server/loss` = server 用 (agg_A + avg_B) 在完整 centralized valid set 上測
- **centralized 虛線**: centralized training 的 best epoch accuracy（upper bound）
