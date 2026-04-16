# FedALC vs FedALC-LWC（無 warm-up）實驗結果

> 日期：2026-04-15
> 設定：SST-2, α=0.5, 30 clients, 20 rounds
> LWC 設定：layer-selection-k=10, layer-reselect-every=1, freeze-sil-threshold=0.8, 無 warm-up

## Accuracy 比較

[FedALC vs LWC Accuracy](../../plots/r30_c30/fedalc_vs_lwc_no_warmup_accuracy_sst2.png)

| 方法 | Best Accuracy | Round |
|---|---|---|
| FedALC | **0.9547** | R26 |
| FedALC-LWC | 0.9522 | R20 |
| FedSA-LoRA | 0.9520 | R17 |
| FedAvg | 0.9457 | R13 |

**觀察**：FedALC 和 FedALC-LWC 非常接近（差 0.25%）。LWC 略低但仍高於 FedSA 和 FedAvg。在 single task + α=0.5 下，layer selection 沒有帶來明顯提升。

## Silhouette Score 比較

[FedALC vs LWC Silhouette](../../plots/r30_c30/fedalc_vs_lwc_no_warmup_silhouette_sst2.png)

**觀察**：
- R1-R4：兩者 silhouette 相近（0.05 → 0.69）
- R4：LWC 的 silhouette 超過 freeze threshold (0.8) → 觸發 Phase 2（frozen）
- R5+：LWC 不再做 AP，silhouette 不再變化
- FedALC 繼續爬升到 0.99，但 R22+ 開始 AP 震盪

**關鍵差異**：LWC 的 freeze 機制在 R4 就生效，完全避開了 FedALC 後期的 AP 震盪問題。

## Cluster 數量

[Cluster Count](../../plots/r30_c30/fedalc_vs_lwc_no_warmup_cluster_count_sst2.png)

**觀察**：
- 兩者都穩定在 5 群
- FedALC R22+ 暴增到 19 群（AP 震盪）
- FedALC-LWC 20 輪全穩定（R4 已 freeze）

## Selected Layers 分析

[Selected Layers Heatmap](../../plots/r30_c30/fedalc_lwc_selected_layers_sst2.png)

**觀察**：
- R1-R4 每輪的 top-10 layers 有變化（layer reselect 每輪觸發）
- 選到的主要是高 index 的 B 矩陣（ffn_inter 層），跟之前的 per-layer discriminability 分析一致
- R5+ frozen 不再做 layer selection

## 結論

### FedALC-LWC 在 single task 下的價值

1. **Accuracy**：跟 FedALC 幾乎相同（差 0.25%），layer selection 沒有帶來明顯提升
2. **Clustering 穩定性**：freeze 機制有效避免了 AP 後期震盪 → 這是 LWC 的實際貢獻
3. **Layer selection 觀察**：top-K 在 R1-R4 有變化，但最終穩定在 ffn_inter 層

### 跟之前 warm-up 版本的比較

| | LWC (warm-up) | LWC (no warm-up) |
|---|---|---|
| Best accuracy | 0.9513（= FedSA） | **0.9522**（> FedSA） |
| 進入 clustering | 從未進入 | R1 就開始 |
| Freeze | 從未觸發 | R4 觸發 |
| 問題 | silhouette 永遠觸發不了 | 無 |

### 下一步

- **QNLI 跑完後補上對比**
- **α=0.3 的比較**：FedALC 在 α=0.3 優勢更大（+1.5%/+4.0%），LWC 是否也如此？
- **Multi-task 場景**：layer selection 在跨任務下可能有更大差異
