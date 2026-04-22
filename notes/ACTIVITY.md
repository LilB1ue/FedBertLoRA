# Activity Log

> 本檔記錄本 repo 內 notes / plots / methods / tooling 的**有感變更**。
>
> **寫法**：新條目放**最上方**（newest first）。每條 = `## [YYYY-MM-DD] <type> | <title>` + 幾個結構化欄位。
>
> **Type**：`archive` / `add-note` / `add-plot` / `method` / `experiment` / `refactor` / `pivot` / `ingest` / `meta`
>
> **維護**：目前手動 append；之後 `/research-wiki:log-recent` 會做 retrospective 掃描補條目（見 `~/projects/research-wiki/.claude/development_log.md`「Activity log 需求」）。
>
> **前身**：2026-04-21 前的條目在 [`_archived/log.md`](_archived/log.md)（舊檔名 `log.md`，格式微異）。

---

## [2026-04-21] add-note | FedALC-AP 三變體同異比較

**Files added**：
- `notes/concepts/fedalc_methods_comparison.md`

**Why**：`fedalc_naming_convention.md` 聚焦命名與 ablation 設計，缺一份 code-grounded 的「同/異」對照。新檔切四張表（架構維度 / clustering feature / config 參數 / `aggregate_fit` pseudocode），把「A/B/others 聚合原則三者一致」「差異全集中在 clustering feature + warm-up + layer selection」講清楚。

**Follow-up**：
- `docs_index.md` 尚未加條目（等 Tier 2 歸檔決定後再一起整理）
- 若之後實作 `FedALC-Spectral-*` / `FedALC-Agglo-*`，需回來補第五列到表 A 與表 B

---

## [2026-04-21] meta | research-wiki skill retrofit + dev log

**Files changed**：
- 改 skill repo（`~/projects/research-wiki/`）：10 個檔（3 新 + 7 改），詳見該 repo `.claude/development_log.md`
- 本 repo：無（skill 未套用）

**Why**：討論「如何把 research-wiki skill 套用到本 repo」，發現 skill 假設跟本 repo 慣例衝突（`raw/` vs `papers/`、`wiki/` vs `notes/`、`/plot-run` 以 wandb 為主 vs 本 repo 以 `logs/eval_metrics.tsv` 為主、`.claude/rules/` 被忽略、無 archive 概念、linear `-vN` 命名不支援 FedALC variant family）。改 skill 適配既有架構，未實際套用（待 trigger A/B/C）。

**Follow-up**：
- Skill 大改未實測；第一次套用前先在 sandbox 跑一遍 `/init` 確認 config 正確生成
- 待 paper 寫完 + 觸發 archive / batch-plot / ablation-table 場景再 enable

---

## [2026-04-21] archive | notes + plots 大批歸檔

**Files moved**：
- `notes/experiments/{fedalc_lwc_results, fedalc_vs_lwc_comparison, findings_summary}.md` → `notes/_archived/experiments/`（3 份）
- `notes/plans/{fedalc_lwc_design, research_plan}.md` → `notes/_archived/plans/`（2 份）
- `plots/r30_c30/*.png`（4 張 bar + heatmap + bubble + membership + layer_hm，共 9 張）→ `plots/_archived/r30_c30/`

**Files added / updated**：
- 新 `notes/_archived/README.md`
- 更新 `plots/_archived/README.md`（加 2026-04-21 批次）
- 更新 `plots/README.md`（主 paper 圖源分類改按折線 / box）
- 更新 `notes/docs_index.md`（移除歸檔條目、加 `_archived/` footer pointer、補 `all_methods_comparison.md` 條目）

**Why**：
- Notes：5 份寫於 FedALC-AP-* rename 前 或 2026-04-20 pivot 推翻的 plan
- Plots：使用者決定 paper 圖源收斂到「折線 + box」兩種，bar / heatmap / bubble 先歸檔（需要時可重繪）

**Tier 2 未處理（等決定）**：`fedalc_vs_lwc_no_warmup_results.md`、`comparison_methods.md`、`fedalc_ap_multi_action_plan.md`、`next_steps.md`、`data_distribution_sst2.md`

**Git**：全部未 stage（遵守 `.claude/rules/git_permission.md`）。

---

## [2026-04-20] pivot | Single-task 升為 primary，multi-task 降 future work

**Files changed**：
- 新增 `.claude/rules/_deprecated_experiment_scope.md`（記錄原 rule「single-task sanity only」廢棄）
- Memory `feedback_single_task_main_method.md` 更新

**Why**：投稿目標調整（非頂會）。α=0.3 下 FedALC vs FedSA 在 QNLI +4% / SST-2 +1.5% 的 signal 夠強，直接當 paper main result；不再堅持 multi-task 當主戰場。FedALC-AP-Multi 的 component（Hopkins / cumulative ΔB / layer selection）在 single-task 下仍弱，降為 optional variant。

**Impact**：basic FedALC-AP 升為主方法；single-task 可跑完整 ablation + 多 seeds。

---

## [2026-04-20] refactor | Client-side eval + log dir fix

**Files changed**：
- `bert/server_app.py`：batch dir 加 `{mode}_a{alpha}` tag
- 新 rule `.claude/rules/evaluation_metric.md`
- 重繪 16 張新 plots（client-side eval, unweighted + weighted mean）
- `notes/experiments/all_methods_comparison.md` 更新

**Why**：發現 server-side eval 對 personalized method 不公平（FedSA 「崩潰」是 artifact，實際 server-side 取 avg B 對保留本地 B 的 FedSA 無意義）。改用 client-side `eval_metrics.tsv` 的 unweighted + weighted mean。同時修 FFA α=0.3 覆蓋 α=0.5 的 log dir overwrite bug。

---

## [2026-04-16] refactor | FedALC family 命名定案

**Files changed**：
- `bert/fedalc_ap_strategy.py`、`bert/fedalc_ap_lwc_strategy.py`、`bert/fedalc_ap_multi_strategy.py`
- `notes/concepts/fedalc_naming_convention.md`（新）
- 各 `run_*.sh` 腳本對應改名

**Why**：FedALC → FedALC-AP（basic）；FedALC-LWC → FedALC-AP-LWC（ablation baseline）；新增 FedALC-AP-Multi（main variant for multi-task）。把 clustering algorithm 暴露在名字 → 未來加 Spectral / Agglo 變體命名可擴展。

---

## 更早期條目

2026-04-02 到 2026-04-16 的原始條目保留在 [`_archived/log.md`](_archived/log.md)（舊檔）。包含：
- FedAvg / FedSA / FFA baseline 跑完（α=0.5, α=0.3）
- FedALC Phase 1（α=0.5）+ Phase 2（α=0.3）
- FedALC-LWC 實驗
- Task vector 連結 + FedALC-AP-Multi 設計初稿
- Paper ingestion 批次
