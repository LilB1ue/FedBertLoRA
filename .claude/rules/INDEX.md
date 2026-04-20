# .claude/rules/ — Project-specific behavioural rules

這份目錄保存對本專案協作的 **行為指引**（給 Claude、給協作者）。
跟 `notes/` 的差別：`notes/` 是方法論 / 實驗紀錄（給人類讀）；`.claude/rules/` 是 dos and don'ts（短、可 scan、拿來行動）。

## 規則清單

| File | 一句話摘要 |
|---|---|
| [evaluation_metric.md](evaluation_metric.md) | 比較 FL methods 用 client-side eval_metrics.tsv，不用 server-side server_eval.tsv |
| [log_directory_layout.md](log_directory_layout.md) | `logs/{ts}_{mode}_a{alpha}/{task}_{mode}_a{alpha}/` — 外層 batch dir 自描述 |
| [commit_format.md](commit_format.md) | Conventional commits（feat/fix/refactor/docs），scope 分開 commit |
| [experiment_scope.md](experiment_scope.md) | Single-task 只做 sanity check，paper main result 留給 multi-task |
| [fedalc_family.md](fedalc_family.md) | FedALC-AP / AP-LWC / AP-Multi 命名規則 + variant 定位 |

## 維護規則

- 規則衝突時，**這目錄 > `CLAUDE.md` > 其他 notes**
- 加新規則 → 更新 INDEX.md + 確保 `CLAUDE.md` 提到 `.claude/rules/`
- 廢棄規則 → 改檔名加 `_deprecated_` prefix，不要直接刪（保留歷史 rationale）
- 規則裡的路徑 / 數字會過期 → 每次大 refactor 後 grep 一下

## 關聯 memory

User-level memory 在 `~/.claude/projects/-data-experiment-exp-fed-BERT-bert/memory/`。
其中 `feedback_evaluation_metric.md` 跟這邊 `evaluation_metric.md` 內容重疊：
- **Memory** 給 Claude 跨 session 的 behavioural prior
- **`.claude/rules/`** 給 repo 協作者（含 human）的靜態參考
