# Rule: Single-task 只做 sanity check，main result 留給 multi-task

## Rule

**Paper 主戰場是 multi-task FL**（對標 FedLEASE / HiLoRA / FedADC），single-task 實驗只做 code 驗證與 component 觀察，**不**在 single-task 上跑完整 ablation / 多 seeds。

## Why

- FedALC-AP-* 的 component（Hopkins adaptive trigger / cumulative ΔB / layer selection / freeze）在 single-task 下發揮空間小：
  - 同 task client 的 task vector 方向本質上一樣，cluster 結構弱
  - Hopkins 難達 threshold，多半 fallback 到 `warmup_max_rounds`
  - Layer selection 的 dissim × norm 在 single-task 下 ranking 模糊
- Single-task α=0.5 所有 non-broken methods 擠在 ~1% window（signal < noise）
- Multi-task 下每個 component 都有信號：不同 task client 的 ΔB 方向差異大 → Hopkins 達標快、layer 選擇有明顯 differentiation、per-cluster B 有 task-level 意義

## How to apply

### Single-task 實驗要跑

1. Smoke test（~10 分鐘）確認 code 沒壞
2. Main comparison 跟既有 baseline 做 single-round sanity check
3. **不**做 component ablation（signal 太弱不可信）
4. **不**跑多 seeds

### Multi-task 實驗要準備（當前還沒 setup）

1. Dataset partition 混 GLUE 多 task（SST-2 + QNLI + MNLI + RTE）
2. Per-task classifier head
3. FedLEASE baseline 實作或重用
4. Per-task eval + aggregated eval

### 既有 single-task 結果的定位

- 視為 Section 5.2 "sanity check" 的支撐材料
- Main result 跟完整 ablation 留 Section 5.1 multi-task
- 不要用 single-task 結果否定某 component（可能是 single-task 本身不適合）

## Related reference

- `notes/plans/fedalc_ap_multi_action_plan.md` — single-task 實驗優先級 P0-P4
- `notes/papers/FedALC-LoRA.md` — paper positioning
