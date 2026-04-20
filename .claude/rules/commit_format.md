# Rule: Commit message format + scope separation

## Rule

Commit message 用 **conventional commits** 格式：`<type>(<scope>): <subject>`，並依**邏輯相關性**拆 commit（不要把無關改動塞一起）。

## Why

- Conventional format 方便 `git log --oneline` 掃描 + 自動生成 changelog。
- 拆 commit 讓 `git bisect` / revert 精準，reviewer 不用面對巨大 diff。
- 這個 repo 以前 commit 都是一句英文描述（無 prefix），新規範跟以前互補（不強制改舊 commit）。

## How to apply

### Type 常用

| type | 用途 |
|---|---|
| `feat` | 新功能（新 strategy、新 method、新 plot 類型） |
| `fix` | bug 修復（含 log overwrite、eval protocol 錯誤） |
| `refactor` | 重構、rename、結構重組（行為不變） |
| `docs` | 純文件（notes/、CLAUDE.md、README） |
| `chore` | 雜事（依賴升級、config） |

### Scope 常用

`fedalc`、`fedsa`、`ffa`、`logs`、`plots`、`experiments`、`notes`

### 拆 commit 原則

- **一個 commit 一個目的**：rename 跟 feature 分開。
- **Docs 更新可包進相關 code commit**（如果是直接說明該 code），否則獨立 `docs(...)` commit。
- **Plots / generated artifacts** 跟產生它們的 script 一起 commit。

### Body 格式

- 用 `Key: value` 列 context（Background / Fix / Findings / Known data loss 等段落）
- Heredoc 寫多段 message（避免引號 escape）：
  ```bash
  git commit -m "$(cat <<'EOF'
  ...
  EOF
  )"
  ```

### 結尾必加

```
Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

## Push 前檢查

- Branch rename 先做完再 push（例：`fedalc-awc` → `fedalc-ap-multi`）
- 不要 push 到 `main` 除非 explicit 要求
- 不要 `--force` / `--no-verify` 除非用戶明確同意
