# Rule: FedALC-AP-* family naming + variant positioning

## Rule

FedALC-* family 命名規則：`FedALC-{clustering_algorithm}[-{variant}]`

- **clustering_algorithm**: AP / Spectral / Agglo（只有 AP 已實作）
- **variant**（optional）：
  - `LWC` — Metric B layer selection + freeze（無 warm-up；ablation baseline）
  - `Multi` — Hopkins adaptive + cumulative ΔB + internal layer selection + freeze（主方法，target multi-task）

## Why

- 舊命名（FedALC, FedALC-LWC, FedALC-AP）沒把 clustering algorithm 暴露在名字 → 之後想加 Spectral / Agglo 變體時命名衝突
- 把 variant 作為 suffix 讓 ablation 容易命名（`FedALC-Spectral-Multi` / `FedALC-Agglo-LWC` 都 extensible）

## How to apply

### Code 對應

| Config mode | Class | File |
|---|---|---|
| `"fedalc-ap"` | `FedALCAPStrategy` | `bert/fedalc_ap_strategy.py` |
| `"fedalc-ap-lwc"` | `FedALCAPLWCStrategy` | `bert/fedalc_ap_lwc_strategy.py` |
| `"fedalc-ap-multi"` | `FedALCAPMultiStrategy` | `bert/fedalc_ap_multi_strategy.py` |

### 不要做

- 不要把 LWC 的 layer selection 當「正交 variant」— 它在 Multi 裡是**必要的 Hopkins 降維前處理**，不是可選外掛
- 不要再給 basic FedALC-AP 加新 feature — 新 feature 加到 Multi variant
- 不要把 Multi 稱為 "FedALC-AP"（會跟 basic 混淆）

### 討論 paper 定位時

- Basic FedALC-AP = baseline
- FedALC-AP-LWC = ablation（AP without Hopkins / cumulative）
- FedALC-AP-Multi = main method

## Authoritative reference

- `notes/concepts/fedalc_naming_convention.md` — 完整 family 結構 + ablation 表
- `notes/papers/FedALC-LoRA.md` — 方法變體描述
- `bert/fedalc_ap_multi_strategy.py` — module docstring 有 theoretical framing
