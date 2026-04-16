# FedALC-AP-Multi Action Plan

> 最後更新：2026-04-16
> Branch: `fedalc-awc`（準備 rename → `fedalc-ap-multi`）

涵蓋兩件事：
1. [Git commit 策略](#git-commit-策略) — 把這次 rename + 新方法的改動推到 repo
2. [Single-task 實驗計畫](#single-task-實驗計畫) — 確認實作正確再轉 multi-task

---

## Git commit 策略

現在 working tree 混合了 (a) 這次 session 的 FedALC-AP family rename + Multi 實作，(b) 前次 session 的 LWC 實驗結果 + plots，(c) 前次 session 的 FFA/models 修復。建議拆成 3 個 commits 保持 review 友善。

### Commit A — FedALC-AP family rename + Multi 實作（主要）

檔案清單：
```
bert/fedalc_ap_strategy.py           # rename from fedalc_strategy.py + class rename
bert/fedalc_ap_lwc_strategy.py       # rename from fedalc_lwc_strategy.py + class rename
bert/fedalc_ap_multi_strategy.py     # 新檔，主方法
bert/server_app.py                   # imports + aggregation-mode 分支
pyproject.toml                       # rename config mode + 新 layer-score-feature
run_fedalc_all.sh                    # aggregation-mode 從 fedalc 改 fedalc-ap
run_fedalc_alpha03.sh                # 同上
run_fedalc_ap_lwc.sh                 # rename from run_fedalc_lwc.sh
run_fedalc_ap_multi.sh               # 新檔
run_fedalc_ap_multi_smoke.sh         # 新檔
CLAUDE.md                            # 全面更新命名
notes/concepts/fedalc_naming_convention.md  # rewrite family 結構
notes/papers/task_vector_connection.md      # FedALC-AP → FedALC-AP-Multi
notes/papers/FedALC-LoRA.md                  # 方法變體描述
```

Commit message：
```
Rename FedALC family to FedALC-AP-* and implement FedALC-AP-Multi

Rename
- FedALC{Strategy,LWCStrategy} -> FedALCAP{Strategy,LWCStrategy} (also file
  and config string renames: fedalc -> fedalc-ap, fedalc-lwc -> fedalc-ap-lwc)

FedALC-AP-Multi (new, main method targeting multi-task FL)
- Built-in Metric B layer selection as dimensionality-reduction preprocessing
  so Hopkins statistic stays meaningful (fixes C1 numerical overflow at D≈50K,
  plus C2 n<4 guard, C3 Phase 2 empty-cluster fallback, I1 b_init=zeros)
- Snapshot selected layers at Phase 0→1 trigger into frozen_layer_indices;
  Phase 1/2 clustering uses frozen set, observation reselect keeps running
  each round and logs Jaccard drift vs frozen for analysis
- Enhanced clustering.jsonl: trigger_fired/reason, freeze_trigger, ap_converged,
  rounds_stable, cumulative_count stats, n_fallback_assignments, per-round
  layer drift

Configs
- New layer-score-feature ("cumulative_delta_b" | "current_b"), wired through
  server_app and pyproject
- New run_fedalc_ap_multi.sh + run_fedalc_ap_multi_smoke.sh (5-round SST-2
  verification with Hopkins/layer-dim checks)

Docs
- fedalc_naming_convention rewritten for new family structure (AP = basic,
  AP-LWC = ablation baseline, AP-Multi = main method)
- task_vector_connection adds high-D Hopkins rationale
- FedALC-LoRA updates method variant descriptions
```

### Commit B — 前次 session 累積的 notes + plots

檔案清單：
```
notes/docs_index.md
notes/plans/next_steps.md
notes/papers/related_papers.md
notes/concepts/evaluation_metrics.md
notes/experiments/findings_summary.md
notes/experiments/fedalc_lwc_results.md
notes/experiments/fedalc_vs_lwc_comparison.md
notes/experiments/fedalc_vs_lwc_no_warmup_results.md
notes/papers/comparison_methods.md
plots/r30_c30/
```

Commit message：
```
Add FedALC-LWC experiment results, findings summary, and comparison notes

- Experiment results on SST-2 (FedALC vs LWC, with and without warmup)
- findings_summary.md: 8 key findings + next-step priorities
- comparison_methods.md: positioning FedALC vs baselines
- Plots for r30_c30 experiments (accuracy, silhouette, cluster counts, etc.)
- Related docs index and related_papers list updated
```

### Commit C — 前次 session 的 code 修復

檔案清單：
```
bert/client_app.py      # received_checkpoints saving
bert/models.py          # freeze_lora_a() helper
run_ffa_all.sh          # 新檔
```

Commit message：
```
FFA-LoRA fixes and FedSA freeze utility in models

- client_app: save received_checkpoints in non-FedAvg modes
- models: freeze_lora_a() helper for FFA mode
- run_ffa_all.sh: batch runner for FFA
```

### 執行順序

```bash
# 先 verify 變更
git status

# Commit A
git add bert/fedalc_ap_strategy.py \
        bert/fedalc_ap_lwc_strategy.py \
        bert/fedalc_ap_multi_strategy.py \
        bert/server_app.py \
        pyproject.toml \
        run_fedalc_all.sh \
        run_fedalc_alpha03.sh \
        run_fedalc_ap_lwc.sh \
        run_fedalc_ap_multi.sh \
        run_fedalc_ap_multi_smoke.sh \
        CLAUDE.md \
        notes/concepts/fedalc_naming_convention.md \
        notes/papers/task_vector_connection.md \
        notes/papers/FedALC-LoRA.md
git status    # 確認只有這些 staged
git commit -m "...(A 的 message)"

# Commit B
git add notes/docs_index.md \
        notes/plans/next_steps.md \
        notes/papers/related_papers.md \
        notes/concepts/evaluation_metrics.md \
        notes/experiments/findings_summary.md \
        notes/experiments/fedalc_lwc_results.md \
        notes/experiments/fedalc_vs_lwc_comparison.md \
        notes/experiments/fedalc_vs_lwc_no_warmup_results.md \
        notes/papers/comparison_methods.md \
        plots/r30_c30/
git commit -m "...(B 的 message)"

# Commit C
git add bert/client_app.py bert/models.py run_ffa_all.sh
git commit -m "...(C 的 message)"

# Branch rename
git branch -m fedalc-awc fedalc-ap-multi

# 驗證
git log --oneline -5
git branch -v
```

---

## Single-task 實驗計畫

### 前提與定位

- **主戰場是 multi-task**（對標 FedLEASE），現在的 single-task 實驗是 **sanity check + component 觀察**
- 目標：確認 FedALC-AP-Multi 實作正確、Hopkins adaptive trigger 有合理行為
- **不在 single-task 上跑大量 seeds / 完整 ablation**，留給 multi-task 再深挖

### Priority 順序

#### P0 — Smoke test（立即跑，~10 分鐘）

**目的**：確認 code 沒壞、Hopkins 值合理。

```bash
bash run_fedalc_ap_multi_smoke.sh localhost-gpu
```

**驗收標準**（smoke test 內建自動檢查）：
1. 跑完 5 rounds 不 crash
2. Hopkins 值落在 [0, 1] 區間，不是 NaN / Inf
3. `selected_layer_indices` 每輪都有 log
4. `n_params_hopkins` / `n_params_clustering` 是 ~10K 等級（top-K 降維生效），不是 ~50K（full B）
5. Phase trajectory 有合理轉變（Phase 0 → 1）

手動檢查 `logs/<timestamp>/sst2_fedalc-ap-multi/clustering.jsonl`：
- `trigger_fired` 在某輪變 `true`
- `trigger_reason` 是 `"hopkins_threshold"` 或 `"max_rounds"`
- 後續 Phase 1 entries 有 `is_first_clustering`, `freeze_trigger`, `layer_drift_jaccard_vs_frozen`

#### P1 — Main single-task comparison（1-2 天）

**目的**：確認 FedALC-AP-Multi 相對 basic 版 at least not worse。

**Methods**：FedAvg, FedSA, FedALC-AP (basic), FedALC-AP-LWC, **FedALC-AP-Multi**

**Setup**：SST-2 + QNLI，α=0.5，30 rounds，單 seed

```bash
# 新方法必跑
bash run_fedalc_ap_multi.sh localhost-gpu 30 0.5

# Baseline 舊結果可重用（在 logs/ 目錄找）
# 如果要確認 rename 後 basic AP 結果一致：
bash run_fedalc_all.sh localhost-gpu 30   # 現在 aggregation-mode=fedalc-ap

# LWC 也重跑一次確認 rename 無影響：
bash run_fedalc_ap_lwc.sh localhost-gpu 30 0.5
```

**預期**：FedALC-AP-Multi 跟 basic 差 -1% ~ +1%（single-task 下本來就不會差太多）。如果**落後 > 2%** → 有 bug，要 debug。

#### P2 — Non-IID 強化（α=0.3，1 天）

**目的**：看 clustering 在更異質資料下效益是否擴大。

```bash
bash run_fedalc_ap_multi.sh localhost-gpu 30 0.3
bash run_fedalc_alpha03.sh localhost-gpu 30   # FedALC-AP basic α=0.3
bash run_baseline_alpha03.sh localhost-gpu    # FedAvg + FedSA α=0.3
```

**預期**：non-IID 越強 clustering 越有用，Multi vs basic 的差距應變大（仍可能邊際）。

#### P3 — Component ablation（1-2 天，可選）

**目的**：分析 Multi 各 component 的貢獻。在 SST-2 + α=0.3 跑變體：

| Ablation | Config override | 預期 |
|---|---|---|
| no-warmup | `warmup-max-rounds=1` | 強制 R1 就進 Phase 1，像 basic。應比 Multi 差 |
| no-freeze | `freeze-sil-threshold=1.1 freeze-stable-rounds=999` | Permanent 每輪 cluster，後期可能震盪 |
| current-B layer scoring | `layer-score-feature='current_b'` | Layer 選擇抖動多，觀察 drift Jaccard 是否偏低 |

⚠️ **如果 ablation 顯示「只要 freeze 就拿到 80% 效益」** → 承認 single-task 下 Hopkins 跟 cumulative 沒發揮空間，轉 multi-task 前不要再花時間在 single-task ablation。

#### P4 — Layer drift analysis（不用新跑，從 P1 log 分析）

直接讀 P1 跑完的 `clustering.jsonl`：

**要繪的圖**：
1. **Hopkins 隨 round 演化**（Phase 0 每輪值 + trigger 位置標記）
2. **Silhouette 隨 round 演化**（Phase 1 + freeze 點）
3. **Layer drift Jaccard 隨 round 演化**（trigger 後每輪 vs frozen 的 Jaccard）
4. **Cumulative count mean 隨 round**（確認累積有持續）

**分析要得出**：
- Trigger 發生在第幾輪？
- Freeze 是 silhouette 還是 stable-rounds 觸發？
- Layer drift Jaccard 穩定還是下降？（下降 = 後期選的層跟 frozen 差很多 = frozen 可能 suboptimal）

### 不建議的跑法

- 重跑舊 FedALC-LWC 實驗：rename 不影響行為，舊結果可用
- Single-task 跑多 seeds：main result 留給 multi-task
- QQP / MNLI：資料量大耗時，single-task 階段不必要

### 時間估計

| 優先級 | 時間 | 能得到什麼 |
|---|---|---|
| P0 smoke | 10 min | 確認實作沒壞 |
| P1 main | 1 天（5 runs × 30-60 min GPU） | Main comparison（Multi vs basic/LWC/baselines） |
| P2 α=0.3 | 1 天 | Non-IID sensitivity |
| P3 ablation | 1-2 天 | Component 貢獻 |
| P4 drift | 0 (從 log 讀) | Layer stability 分析 |

**總計：3-4 天**，之後停 single-task 轉 multi-task setup。

### Deliverables（跑完能寫進 paper / notes 的東西）

1. `notes/experiments/fedalc_ap_multi_results_sst2.md` — P1 結果表格 + 曲線
2. `notes/experiments/fedalc_ap_multi_results_qnli.md` — 同上 QNLI
3. `notes/experiments/fedalc_ap_multi_alpha_sensitivity.md` — P2 α=0.3 vs 0.5 對比
4. `notes/experiments/fedalc_ap_multi_ablation.md` — P3 ablation（若跑）
5. `notes/experiments/fedalc_ap_multi_drift_analysis.md` — P4 drift 分析 + plots
6. `plots/r30_c30/fedalc_ap_multi_*.png` — 圖表

---

## 下下步（超出 single-task 範圍）

做完 P0-P2 + P4 後，**停 single-task**，開始：

1. **Multi-task dataset partition 設計** — 看完 FedLEASE paper 之後決定怎麼切
2. **Per-task classifier head** — model.py 要改
3. **FedLEASE baseline 實作** — 從 repo 或 paper 還原
4. **Multi-task evaluation protocol** — per-task acc vs weighted avg

這是下一個階段的工作，這份 action plan 不涵蓋。
