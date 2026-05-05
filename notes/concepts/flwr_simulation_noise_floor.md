---
tags: [flwr, determinism, verification, noise_floor]
related: [../experiments/all_methods_comparison.md, ../../verify_refactor_bit_exact.sh, ../../diagnose_flwr_determinism.sh]
---

# flwr local-simulation noise floor

> 最後更新：2026-05-04
> 結論：在我們的 setup（30 clients × CPU local-simulation × multi-thread BLAS），**flwr simulation 對同 config 兩次 run 不是 byte-deterministic**。R1 通常一致，R2+ 部分 client 飄 ~1-2pp。這是 framework + PyTorch + BLAS 的 inherent property，**不是 bug**。

## TL;DR

跟同 branch 跑兩次同 config 的 diagnose 結果(SST-2 fedsa α=0.5 3-round):

### Per-partition view(看單個 client 飄多少)

| | 一致性 |
|---|---|
| R1 全 30 partition eval accuracy | ✅ byte-identical(小數第 16 位)|
| R2 partition 0/3/6/7/8/10/13/19/23 accuracy | ❌ 飄 0.2–2.4 pp |
| R2 其他 partition | ✅ byte-identical |
| Partition num_examples(dataset 切割) | ✅ 完全一致 |

### Aggregated view(paper main metric 飄多少)

| Round | Unweighted mean drift | Weighted mean drift |
|---|---|---|
| R1 | 0pp | 0pp |
| R2 | **+0.12pp** | **+0.17pp** |
| R3 | **+0.04pp** | **+0.04pp** |

**關鍵 insight**:per-partition 飄看起來大(最大 2.4pp),但 30 個 partition 平均後**飄向 +/- 抵消**,**mean accuracy 只飄 ~0.1-0.2pp**。

→ Refactor 等 pure deterministic 改動的 bit-exact verification **不可靠**,要降級成 statistical equivalence。但**aggregated metric 的 noise floor 比想像中小很多**(< 0.5pp)。

---

## 是怎麼發現的

2026-05-04 做 `bert/lora_utils.py` refactor（5 strategy file 的 `_separate_a_b_others / _reconstruct_parameters / _weighted_average / _compute_layer_scores` 抽出去）。

寫了 `verify_refactor_bit_exact.sh` 預期 baseline branch vs refactor branch 在同 config 下 `eval_metrics.tsv` 應該 byte-identical（因為抽出去的 4 個 function 都是 pure deterministic）。

跑下去 FAIL：R1 全等但 R2 部分 partition 飄 ~1-2pp。

無法直接判斷是 refactor bug 還是 flwr 本身吵 → 寫 `diagnose_flwr_determinism.sh` 在**同一個 branch** 跑兩次，看 baseline 自己跟自己會不會也飄。

結果：**同 branch 兩次 run 也飄一樣的 9 個 partition、一樣的 magnitude**。確認 refactor 沒問題，是 flwr noise floor。

---

## 為什麼**剛好**這 9 個 partition 飄

對照 SST-2 α=0.5 30-client partition 的 label 分佈：

| 飄的 partition（典型） | n | acc | label distribution |
|---|---|---|---|
| pid 0 | 559 | ~91% | mixed |
| pid 13 | 421 | ~82% | mixed |
| pid 19 | 378 | ~82% | mixed |
| pid 23 | 946 | ~93% | mixed |

| 不飄的 partition（典型） | n | acc | label distribution |
|---|---|---|---|
| pid 16 | 1566 | ~99.6% | extreme skew（~99.6% 一類） |
| pid 9 | 267 | ~99.6% | extreme skew |
| pid 21 | 134 | ~59.7% | extreme skew（另一類為主） |
| pid 2 | 43 | ~93% | tiny + skewed |

**Pattern**：
- **Label 極端偏的 client**：training 退化成「預測 majority class」的 trivial solution → 不論 dropout / batch order 怎麼飄都收斂到同一點 → deterministic
- **正常分類的 client**：有真的 gradient 訊號，model 在學 → **dropout / batch order 飄就讓 trajectory 飄** → noise 顯現

→ noise 不是均勻分散在所有 partition，而是**集中在「真的在學」的 partition**。對 paper 的 unweighted-mean accuracy 影響存在但有限（飄的 partition 是一部分 + 飄向有 +/- 抵消）。

---

## 為什麼 R1 不飄、R2 才飄

| | R1 | R2+ |
|---|---|---|
| `set_seed(42)` 設過? | ✅（client_fn 開頭）| ✅（每 round client_fn 重新觸發）|
| BLAS thread pool 內部 state | 乾淨（process 剛啟動）| 帶有 R1 計算殘留 |
| Ray worker process 是否 warm | 冷啟動順序固定 | warm，scheduling 動態 |
| Process-level OS state | 乾淨 | 帶有 R1 殘留 |

R1 是「冷啟動 → 排程相對 deterministic」，R2 之後底層 state 累積各種 source 的微小擾動 → 飄。

---

## `set_seed(42)` 沒涵蓋的非決定性源頭

`bert/models.py:set_seed` 只設了 Python-level RNG：

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
```

**沒**設到的：

| 源頭 | 對 CPU simulation 影響 |
|---|---|
| BLAS / MKL multi-thread reduction 順序 | 強（matmul 主要計算路徑） |
| Ray worker process scheduling | 強（client → worker assignment 變動） |
| PyTorch DataLoader workers | 視 num_workers 設定 |
| `PYTHONHASHSEED`（dict / set 順序） | 弱 |
| CUDA / cuDNN | 不適用（CPU） |

要強制 byte-deterministic 需要：

```bash
# 在 process 啟動 前 設
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONHASHSEED=42
# Python 內
torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)
```

代價：訓練時間 ×（thread 數）→ 30-client simulation 慢很多。**目前不啟用**，因為：
1. 啟用後既有所有 baseline 數字失效（要重跑全部 ablation）
2. Paper 級 reproducibility 不靠 bit-exact，靠多 seed 平均

---

## 給未來的 implication

### Refactor verification 該怎麼做

❌ **不要**用 bit-exact diff（`verify_refactor_bit_exact.sh` 過度樂觀）

✅ **正確做法**：
1. 跑 `diagnose_flwr_determinism.sh` 量 noise floor（baseline 自己跑兩次的飄幅）
2. 跑 verify（baseline vs refactor 各一次）
3. 如果 verify 飄幅在 noise floor 範圍內 → pass
4. 如果飄幅遠大於 noise floor → 真 bug

### Multi-seed evaluation 是必須的

對 paper main comparison（FedALC vs FedSA vs FFA）：
- 同一個 method 跑多 seed（建議 3-5 seeds）
- 報 mean ± std
- 差距 < 2 × max(per-method std) 視為 noise

不要單一 seed 對單一 seed 比 raw accuracy。

### Strategy 改動的 sanity check

每次改完 strategy 重要邏輯，在 unweighted-mean accuracy 上應該：
- 跟同 strategy 不同 seed 的 result 在 noise range 內（~2pp）
- 跟之前 commit 跑的 result 落在類似分佈

不能期待 byte-exact match。

---

## Tools

| Script | 用途 |
|---|---|
| `verify_refactor_bit_exact.sh` | 比兩個 branch 的 eval_metrics.tsv（**現在知道太嚴格，當「快速 sanity」用即可**） |
| `diagnose_flwr_determinism.sh` | 量 noise floor（同 branch 跑兩次比結果） |

兩個都在 repo root。跑完的 logs 在 `logs/REFACTOR_BL_*` / `REFACTOR_VR_*` / `DETERM_A_*` / `DETERM_B_*`，用完可刪。

---

## 相關紀錄

- 2026-05-04 ACTIVITY entry：refactor + 此調查
- Refactor commit：`7840e9c refactor(strategies): extract shared LoRA utilities to bert/lora_utils`
- 上游討論：`~/.claude/plans/lora-buzzing-seal.md`（plan 從 LR schedule 議題演進到 refactor）
