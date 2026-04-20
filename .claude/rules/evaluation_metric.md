# Rule: Use client-side eval for method comparison

## Rule

比較 FL methods 的 accuracy / convergence 時，主 metric 用 **client-side `eval_metrics.tsv`**（per-client test split, Flower evaluate protocol）。同時報 **unweighted mean**（主）+ **weighted mean by num_examples**（附），並加 per-client distribution。**不要**把 `server_eval.tsv` 當 main metric。

## Why

- 本專案目標是 **personalized models**（每個 client 有自己的 B + others）。
- Server-side eval 取 parameter average 當 global model，對 personalized method 無意義 — 例：FedSA 保留本地 B，server-side 取 avg B 是 noise，呈現假性「崩潰」。
- Client-side eval 反映 **end-user 部署後真正體驗的 accuracy**，對所有 method 公平。

## How to apply

- Plot / table / paper main result 讀 `logs/<batch_dir>/<run_dir>/eval_metrics.tsv`。
- 聚合方式兩個都算：
  - Unweighted = Σaccᵢ / N   （primary，personalized fairness）
  - Weighted   = Σ(accᵢ·nᵢ) / Σnᵢ   （secondary，== global pooled accuracy）
- `server_eval.tsv` 僅作 sanity check（檢查 aggregation 沒有 diverge），不做 main comparison。
- 兩者差距大 = fairness issue（大 client 主導），特別在 α=0.3 下注意。

## Authoritative reference

- `plot_method_comparison.py`（repo root）— 實作範本
- `notes/experiments/all_methods_comparison.md` — 完整結果
