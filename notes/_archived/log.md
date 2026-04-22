---
tags: [misc]
related: []
---

# Research Log

每筆用 `## [YYYY-MM-DD] <type> | <title>` 格式。Type: experiment / ingest / query / lint / refactor。1-3 行摘要，細節在對應頁面。

> 本檔由 `.claude/rules/wiki_log.md` 規範；歷史部分是從 `tools/log_inventory.tsv`、git log、notes 的「最後更新」日期重建。更早期無法確認日期的事項已跳過。

## [2026-04-02] experiment | FedAvg α=0.5（SST-2 開跑）
首批 FedAvg baseline 實驗，20 rounds。詳見 notes/experiments/fedavg_results.md。

## [2026-04-05] experiment | FedSA-LoRA α=0.5（SST-2 + QNLI + MNLI + QQP）
FedSA-LoRA 全 GLUE 任務實驗，20 rounds，α=0.5。詳見 notes/experiments/fedavg_vs_fedsa_comparison.md。

## [2026-04-06] experiment | FedALC-LoRA α=0.5（SST-2 + QNLI）
Phase 1：首次 AP clustering on B matrices，30 rounds。詳見 notes/experiments/findings_summary.md。

## [2026-04-08] experiment | FedALC-LoRA α=0.3（SST-2 + QNLI）
Phase 2：α=0.3 non-IID 測試；FedALC 相對 baseline 優勢擴大。詳見 notes/experiments/findings_summary.md。

## [2026-04-08] ingest | FedALC-LoRA 方法定位初稿
整合 FedADC/FedLEASE/HiLoRA 的 layer selection motivation + Phase 1 結果。筆記存於 notes/papers/FedALC-LoRA.md。

## [2026-04-12] experiment | FedAvg + FedSA α=0.3 baseline（SST-2 + QNLI）
Phase 2 baseline 補齊；跟 FedALC α=0.3 做對照。詳見 notes/experiments/findings_summary.md。

## [2026-04-15] experiment | FedALC-LWC（layer-wise clustering, α=0.5）
Silhouette warm-up + Metric B layer selection。詳見 notes/experiments/fedalc_lwc_results.md, notes/experiments/fedalc_vs_lwc_comparison.md, notes/experiments/fedalc_vs_lwc_no_warmup_results.md。

## [2026-04-15] ingest | Comparison methods & clustering stability
整合各 baseline 的 Tier 1-3 分類 + AP 震盪分析。筆記存於 notes/papers/comparison_methods.md, notes/plans/fedalc_lwc_design.md。

## [2026-04-15] refactor | LaTeX 化 markdown notes 中的公式
Unify notation for paper writing。

## [2026-04-16] experiment | FFA-LoRA α=0.5 + α=0.3（SST-2 + QNLI）
Freeze A, aggregate B。跑完後 α=0.3 log 覆蓋 α=0.5（log 覆寫 bug，於 2026-04-20 修復）。詳見 notes/experiments/all_methods_comparison.md。

## [2026-04-16] refactor | FedALC family rename → FedALC-AP-*
FedALC → FedALC-AP / FedALC-LWC → FedALC-AP-LWC / 新增 FedALC-AP-Multi。詳見 notes/concepts/fedalc_naming_convention.md。

## [2026-04-16] ingest | Task vector connection + FedALC-AP-Multi 方法設計
LoRA ΔW=BA 的 task vector 視角 + Hopkins adaptive warm-up + 內建 layer selection。筆記存於 notes/papers/task_vector_connection.md, notes/concepts/fedalc_naming_convention.md, notes/plans/fedalc_ap_multi_action_plan.md。

## [2026-04-16] query | Code review 後 8 個 findings + next steps
整理 FedALC/LWC 實驗結論、確認 paper story 走向。歸檔至 notes/experiments/findings_summary.md, notes/plans/next_steps.md。

## [2026-04-20] refactor | Log dir overwrite fix + 加 method/alpha tag
修復 FFA α=0.3 覆蓋 α=0.5 問題；server_app.py 的 batch dir 改成 `{ts}_{mode}_a{alpha}/`；歷史 log dir rename。詳見 tools/log_rename_manifest.tsv。

## [2026-04-20] experiment | 切換到 client-side eval 後重做 method comparison
發現 server-side eval 對 personalized method 不公平（FedSA 「崩潰」是 artifact）。改用 client-side eval_metrics.tsv 的 unweighted + weighted mean，16 張新 plots。詳見 notes/experiments/all_methods_comparison.md。

## [2026-04-20] lint | 初始化 LLM Wiki 架構
掃描 notes/ 目錄，建立 log.md，更新 docs_index.md，為 30 個 notes 頁面補 YAML frontmatter，並執行首次 lint：0 個孤兒頁、0 個 TODO marker、27 張 plot 未被 notes 引用（多為舊實驗 plot）。Top-3 建議補充的頁面：findings_summary.md（server-side 結論待更新）、next_steps.md（正文內容待更新 client-side 數字）、fedalc_vs_lwc_* 系列（rename 前寫的實驗結果）。

