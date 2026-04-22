# Archived notes（2026-04-21）

本目錄收錄已被新內容取代、或因專案方向調整不再作為主線參考的 notes。
**不要在 paper / 報告引用這裡的結論**。需要時從原地重寫更新版。

原則：不刪檔，保留歷史 rationale；docs_index.md 不再列入主表。

## `_archived/experiments/`

| 檔 | 過時原因 |
|---|---|
| `fedalc_lwc_results.md` | 寫於 FedALC-AP-* rename 前；warm-up 失敗結論已被 no-warmup 版取代（見 `experiments/fedalc_vs_lwc_no_warmup_results.md`） |
| `fedalc_vs_lwc_comparison.md` | 寫於 rename 前的 comparison 設計，已實作完畢；當前 comparison 見 `experiments/all_methods_comparison.md` |
| `findings_summary.md` | 寫於 rename 前；基於 server-side eval，違反 `.claude/rules/evaluation_metric.md` 的 client-side 規範 |

## `_archived/plans/`

| 檔 | 過時原因 |
|---|---|
| `fedalc_lwc_design.md` | rename 前的 design doc；FedALC-AP-LWC 已實作，現行命名見 `concepts/fedalc_naming_convention.md` |
| `research_plan.md` | 原 Phase 1-3 三階段規劃；被 2026-04-20 single-task pivot 推翻（見 `.claude/rules/_deprecated_experiment_scope.md` 為何廢棄 multi-task 假設） |

## `_archived/log.md`（2026-04-21）

舊的 research log，格式 `## [YYYY-MM-DD] <type> | <title>`，手動維護負擔大已擺爛。替代品為 `notes/ACTIVITY.md`（重新設計格式，準備接 `/research-wiki:log-recent` 自動化）。舊檔保留供 2026-04-02 到 2026-04-16 的原始歷史條目查詢。

## 恢復方式

```bash
git mv notes/_archived/<category>/<file>.md notes/<category>/<file>.md
```
