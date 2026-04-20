# Rule: Log directory layout with method + alpha tag

## Rule

所有 log 目錄遵循：

```
logs/{timestamp}_{mode}_a{alpha}/{task}_{mode}_a{alpha}/
```

- **外層** `{timestamp}_{mode}_a{alpha}` — batch dir，`ls logs/` 一眼辨識
- **內層** `{task}_{mode}_a{alpha}` — run dir，單獨路徑也能 identify

## Why

- 同一個 `run_*_all.sh` 跑多個 alpha 時，沒 alpha tag 會 **overwrite**（不同 alpha 寫同 subdir）。已知事故：FFA α=0.3 覆蓋 α=0.5。
- 外層 batch dir 不帶 mode/alpha 時，`ls logs/` 只看到 timestamp，無法識別 run 性質。
- 內層 subdir 保留冗餘，是為了單獨路徑丟 plot script 還能 parse。

## How to apply

- `server_app.py` 已自動 emit 此結構（`batch_dir = f"{log_timestamp}_{aggregation_mode}_a{alpha}"`）。
- 不要把多個 alpha/mode 的結果塞進同一 batch dir 底下。
- `log-timestamp` 在 shell script 可共享（batch run），但 batch dir 會因 mode/alpha 不同而分開。
- 歷史遺留 dir 用 `tools/apply_log_rename.sh` + `tools/apply_batch_dir_rename.sh` 修復。

## Known edge cases

- Alpha unknown（pre-wandb 的 runs）→ 標 `_aUNKNOWN`
- `{timestamp}_ai2` 類特殊 suffix → 前綴視為完整 timestamp，保留 `_ai2` 在 timestamp 部分
