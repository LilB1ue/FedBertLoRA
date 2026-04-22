---
tags: [concept, evaluation, personalized-fl]
related:
  - notes/concepts/evaluation_metrics.md
  - notes/experiments/all_methods_comparison.md
  - .claude/rules/evaluation_metric.md
---

# Personalized FL 的 eval weighting 慣例：weighted vs unweighted

> 最後更新：2026-04-20
> 問題背景：本專案 `all_methods_comparison.md` 同時報 unweighted mean (primary) + weighted mean (secondary) 的 per-client test accuracy，但沒總結其他 paper 用哪個。這份筆記整理常見 paper 的慣例。
>
> ⚠️ **Caveat**：下列摘要基於 `notes/papers/` 內的整理 + 典型 convention，**未一一回去逐頁翻 PDF 核對最新版本的 evaluation protocol 描述**。若要寫進 paper 的 related work，請回去對 PDF 確認 §Experiments/§Evaluation 段落的原文。

## 核心區別

| 名稱 | 公式 | 意義 |
|---|---|---|
| Unweighted (macro) | $\frac{1}{N}\sum_i \text{acc}_i$ | 每個 client 權重相等，對小 client 友善 |
| Weighted (micro) | $\frac{\sum_i n_i \cdot \text{acc}_i}{\sum_i n_i}$ | 等同把所有 client test 合在一起算，大 client 主導；== global pooled accuracy |

兩者差距大 = fairness 問題（大 client 拿高分、小 client 拿低分 or 反之）。

## 常見 paper 的慣例（按 category）

### Personalized FL — 以 per-client 為核心的 paper

| Paper | 報的指標 | 備註 |
|---|---|---|
| **FedPer** (Arivazhagan 2019) | Per-client accuracy distribution + mean | 明確關心 client 公平性，常報 box plot / histogram |
| **LG-FedAvg** (Liang 2020) | Mean per-client accuracy（unweighted） | CIFAR 設定每 client 資料量相同 → weighted = unweighted |
| **FedRep** (Collins 2021) | Per-client mean，有 worst-case 報告 | 理論分析 focus 在 client-level guarantee |
| **pFedMe** (Dinh 2020) | Unweighted mean + std across clients | Personalized model 的 default reporting |
| **Ditto** (Li 2021) | Both: mean + per-client distribution | 有專節討論 fairness，特別看底層 client |

**慣例**：personalized FL paper 多半 report **unweighted mean + per-client distribution**（通常是 std 或 box plot）。Weighted mean 反而少見，因為跟「personalization 讓每個 client 都好」的故事不一致。

### LoRA-based FL — 我們這條線

| Paper | 報的指標 | 備註 |
|---|---|---|
| **FedSA-LoRA** (Guo 2025 ICLR) | "Weighted average across clients" (personalized_fl_papers.md 摘) | 3 clients, α=0.5 → client 數少，weighted 跟 unweighted 差不會太大 |
| **FFA-LoRA** (Sun 2024 ICLR) | Server-side global model on pooled test | 非 personalized（A freeze + 聚合 B）→ 用 global eval 合理 |
| **FedLEASE** (NeurIPS 2025) | Per-task accuracy + aggregate（multi-task FL） | Multi-task 下 "per-task mean" 比 "per-client mean" 更自然 |
| **HiLoRA** (CVPR 2026) | Per-client + global model 都報 | 有 hierarchical structure，兩個角度都有意義 |
| **FedDPA** (Yang 2024 NeurIPS) | Per-client personalized accuracy | 有 global + local adapter，報 per-client 最能反映雙 adapter 的價值 |

**慣例**：LoRA-based FL paper 中，**明確做 personalized 的（FedSA、FedDPA）報 per-client mean**；**做 global 的（FFA）報 server-side**；**做 multi-task 的（FedLEASE）report per-task**。

### Classical clustered FL

| Paper | 報的指標 |
|---|---|
| **CFL** (Sattler 2020) | Per-cluster performance + overall mean |
| **FedADC** (2026) | Per-client accuracy mean |
| **IFCA** (Ghosh 2020) | Per-cluster test accuracy |

## 我們專案的選擇

**Primary metric**：unweighted mean of per-client test accuracy（`eval_metrics.tsv`）
**Secondary**：weighted mean + per-client box plot + fairness gap (weighted − unweighted)

Why：
1. 目標是 personalized models → 每個 client 都應該好 → unweighted 直接反映這個
2. Per-client distribution（box plot）揭露 fairness issue — 這是 `all_methods_comparison.md` Finding 4 發現 FedALC-AP 的 +4.85% gap 的依據
3. Weighted mean 還是要報，因為它 = global pooled accuracy，方便對照 server-side FedAvg 等 global method

這個選擇跟 **pFedMe / FedRep / Ditto 一致**（personalized FL 主流）。不跟 FedSA 原 paper 一致（那邊 weighted），但 FedSA 只有 3 clients，差異可忽略；我們 30 clients 下 weighted vs unweighted 差距可達 4-5%，必須分開報。

## 與 .claude/rules/evaluation_metric.md 的關係

規則檔規定「用 client-side eval、同時報 unweighted + weighted」。這份檔案補上：
- 為什麼兩個都要報（不同社群不同習慣）
- 其他 paper 用哪個（讓 related work 能對齊）
- 本專案選 unweighted primary 的依據（pFedMe/FedRep/Ditto 先例）

## 待驗證事項（寫 paper 前必須回去查原文）

- [ ] FedSA-LoRA 原 paper §5 Experiments 用的到底是 per-client weighted 還是 pooled global eval（`personalized_fl_papers.md` 寫 "weighted average" 但沒引號原文）
- [ ] HiLoRA 報 personalization 的具體公式
- [ ] FedLEASE 的 "per-task accuracy" 是 pool test set 還是 mean of per-client
- [ ] Ditto paper 的 worst-case client 報告方式（我們可能也要補 bottom-k client 的 accuracy）

## 參考

- `notes/papers/personalized_fl_papers.md` — 我們整理的 personalized FL papers 摘要
- `notes/papers/related_papers.md` — 完整 FL+LoRA 論文列表
- `notes/concepts/evaluation_metrics.md` — 本專案的 eval pipeline 說明
- `.claude/rules/evaluation_metric.md` — project rule：用 client-side eval
