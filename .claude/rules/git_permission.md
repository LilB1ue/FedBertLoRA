# Rule: Ask permission before git add / commit / push

## Rule

任何會寫入 git 或修改 remote 的動作，**必須先問使用者同意**再執行。包括：

- `git add ...`
- `git commit ...`（含 `--amend`）
- `git push ...`
- `git merge`, `git rebase`, `git reset --hard`, `git checkout --` 之類 destructive / rewriting 操作
- `git branch -d` / `-D`、`git branch -m` 重新命名

## Why

- 這些動作會改變 repo 狀態，有的甚至影響 remote / 其他協作者
- 一次錯誤 commit / push 的成本遠高於一次 confirm 的成本
- 使用者要 review 改了什麼檔案、commit message 是否貼切，再決定時機

## How to apply

### 需要先問

即使使用者先前說「都執行」，每個新的 git add/commit/push 請求仍要**單獨**確認。**先前 auto 授權不遞延到下一次**。

**該做**的流程：
1. 告知想 stage 哪些檔案（`git status` / `git diff --stat` 先秀）
2. 提出 commit message 草稿
3. **等使用者回「yes / 可以 / 做」之類的明確同意**再執行
4. Push 前再確認一次

### 不需要問（read-only）

下列指令可直接執行，不用問：
- `git status`, `git log`, `git diff`, `git show`
- `git branch -v`, `git branch -a`, `git remote -v`
- `git fetch`（只拉資料，不改 local state）
- `git stash list`

### 特殊情況

- 若是 CI 自動化場景 / 明確在 auto mode 中且已在 session 開頭授權 batch git 動作 → 可沿用至該 batch 結束，但每個新 session / 新一批工作仍要重新確認
- 破壞性操作（`reset --hard`, `push --force`, `branch -D`, `rebase -i`）**一律**要問，無論 mode
- 不要 `--no-verify` / `--no-gpg-sign` 除非使用者明確同意

## Authoritative reference

- `.claude/rules/commit_format.md` — commit message 格式（本 rule 的下游）
- Claude Code 的 Git Safety Protocol（system prompt）亦有相似規定，本 rule 補強 repo-specific 的授權流程
