# FedSA-LoRA Code Review 結果

> 2026-04-06, 針對 `fedsa_strategy.py`, `server_app.py`, `client_app.py`, `models.py` 的完整 review

## 正確的部分

- A/B 分離邏輯正確，292 個參數 key 的順序全程保持一致
- `_separate_a_b` 和 `_reconstruct_parameters` 使用三個獨立 index (a_idx, b_idx, o_idx) 正確交錯還原
- `configure_evaluate` 正確 mirror `configure_fit`，送個人化參數給每個 client
- Classifier (4 個參數) 歸類到 B 類，跟論文 `local_param: ['lora_B', 'classifier']` 一致
- `_weighted_average` 實作正確，處理了 zero weight edge case
- Fallback 邏輯合理（新 client 用 avg_B）

## 問題列表

### Issue 1 — 潛在 bug：缺少 "score" 檢查

**檔案**: `fedsa_strategy.py` line 77

`_separate_a_b` 檢查 `"classifier" in key` 但沒檢查 `"score"`。

PEFT 對 `task_type="SEQ_CLS"` 自動設 `modules_to_save: ["classifier", "score"]`：
- RoBERTa 用 `classifier` → 目前沒問題
- GPT-2 等模型用 `score` → 會被誤歸到 `other_params` 被聚合

**修法**: `elif "lora_B" in key or "classifier" in key or "score" in key:`

**嚴重度**: 低（目前用 RoBERTa 不影響，換模型才會出問題）

### Issue 2 — Server eval 用 avg_B 產生誤導指標

**檔案**: `server_app.py` evaluate_fn

Server-side `evaluate_fn` 用 `(global_A + avg_B + avg_classifier)` 在 centralized validation set 上測。

實際觀察：
- Centralized loss 在 round 8 後持續上升（0.177 → 0.418）
- 但 distributed personalized accuracy 穩定在 0.952

看起來像訓練崩了，其實是 avg_B 模型本來就不是 FedSA 的重點。

**建議**: 明確標記哪個 metric 是主要的。Distributed evaluate 的 weighted average 才是論文對應的指標。

### Issue 3 — 每輪傳全部參數（A+B 都傳）

**檔案**: `fedsa_strategy.py` configure_fit / client_app.py fit

論文的通訊協議：
- Server → Client: 只傳 A
- Client → Server: 只傳 A

目前實作：兩邊都傳全部 292 個參數，約 2x 多餘通訊量。

**影響**: Simulation 模式下無差（記憶體內傳），但寫論文比通訊量時不能這樣算。

**嚴重度**: 功能正確，效能問題。

### Issue 4 — FFA 模式 client 端沒真的凍結 A

**檔案**: `client_app.py` fit

FFA-LoRA 的 A 應該凍結不訓練。目前 client 端仍然訓練 A，只是 server 端丟掉不用。浪費 compute。

**修法**: 如果要用 FFA 模式，需傳 `aggregation_mode` 給 client，條件式 freeze A 的 `requires_grad`。

**嚴重度**: FedSA 模式不影響，只影響 FFA。

### Issue 5 — `del self.net` 模式偏脆弱

**檔案**: `client_app.py` line 108, 121

`fit()` 和 `evaluate()` 都 `del self.net`。Flower simulation 每輪建新 client instance 所以安全，但如果 client 被重用會 crash。

**嚴重度**: 低，目前安全。

### Issue 6 — 後期有 overfitting 跡象

實驗觀察：
- Distributed loss 在 round 5 後上升（0.174 → 0.810）
- Accuracy 在 round 13 後持平（~0.952）
- 模型變得過度自信（loss 升但 acc 不降）

**可能原因**: Cosine annealing 後期 LR 太低鎖住了 overfit 狀態。

## 參數分佈確認

```
292 個 PEFT state dict 參數:
  144 個 lora_A  → A 類（聚合）
  144 個 lora_B  → B 類（留本地）
    4 個 classifier → B 類（留本地）
    0 個 other    → 空（不存在）
```

## 結論

核心的 FedSA-LoRA 邏輯（A/B 分離、per-client B 追蹤、個人化 configure_fit/configure_evaluate）**概念正確**，跟論文演算法一致。最需要注意的是 Issue 2（server eval 指標誤導）和 Issue 3（通訊量計算）。
