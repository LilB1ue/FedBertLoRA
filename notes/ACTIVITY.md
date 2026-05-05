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

## [2026-05-05] meta | LWC 文件修正：實作上沒有 silhouette warm-up

**Files changed**：`CLAUDE.md`, `.claude/rules/fedalc_family.md`, `notes/concepts/fedalc_naming_convention.md`, `notes/concepts/fedalc_methods_comparison.md`, `notes/papers/FedALC-LoRA.md`, `notes/experiments/all_methods_comparison.md`

**Why**：`fedalc_ap_lwc_strategy.py` docstring 寫 3-phase（warm-up → cluster → freeze），實際 code `self.phase = 1` 直接跳過 Phase 0，`warmup_sil_threshold` 從未被任何 logic 讀取。多份 markdown 跟著錯。

**改了什麼**：
- 6 份 md 統一改成「LWC = layer selection + freeze（無 warm-up）」
- `fedalc_methods_comparison.md` 配置表把 LWC 那欄的 `warmup_sil_threshold` 從 ✓ 改成 ✗（殘留參數但未使用）
- `all_methods_comparison.md` Finding 3 是事實錯誤（把功勞算到不存在的 sil warm-up 上）→ 改成 layer selection + early freeze

**Follow-up**：
- code 端 `fedalc_ap_lwc_strategy.py` docstring + 死參數 `warmup_sil_threshold` 待清掉（doc-only 這次先不動）
- 真正的乾淨 ablation 軸（LWC w/ sil-warm-up vs Multi w/ Hopkins）需要先把 sil-warm-up 補回 LWC，現在的「LWC vs Multi」差兩個 component 捆綁

---

## [2026-05-04] refactor | strategy utilities 抽出到 bert/lora_utils + flwr noise floor 量測

**Files added**：
- `bert/lora_utils.py`（新 module，4 個 pure functions：`separate_a_b_others` / `reconstruct_parameters` / `weighted_average` / `compute_layer_scores`）
- `tools/verify_refactor_bit_exact.sh`（refactor 用的 bit-exact 驗證 script）
- `tools/diagnose_flwr_determinism.sh`（flwr 自身決定性診斷 script）
- `notes/concepts/flwr_simulation_noise_floor.md`（noise floor 結構性紀錄）

**Files changed**：
- 5 個 strategy files（`bert/fedsa_strategy.py`, `bert/fedalc_ap_strategy.py`, `bert/fedalc_ap_lwc_strategy.py`, `bert/fedalc_ap_multi_strategy.py`, `bert/fedalc_agglo_lwc_strategy.py`）— 改用 `bert.lora_utils` import 取代 in-class duplicate methods
- `.gitignore` 加 `.vscode`(IDE 自動加)

**LOC 變化**：5 strategy file 累計 -252 LOC（duplicate methods 全砍），加 +100 LOC 在 utils module → 淨 -152。

**Refactor 設計**：
- FedSA 的 `classifier`/`score` 特殊處理從硬寫條件改用參數 `b_extra_keys=("classifier", "score")` 表達
- AP-Multi 的壓縮風格(no docstring + 分號擠行)在抽出後消失
- `_compute_layer_scores` 三個 LWC variant 略有差異(AP-Multi 加 n=1 safeguard,Agglo-LWC 加 float() cast),統一成 AP-Multi 的較 safe 版本(n>=2 實務情境下 bit-identical)

**Why（refactor 動機）**：
- 5 strategy file 共 2,771 行,~520 行是 duplicate(_separate_a_b_others / _reconstruct_parameters / _weighted_average byte-identical 4-5 個檔)
- Naming 不一致(FedSA 用 `_separate_a_b`,FedALC family 用 `_separate_a_b_others`)
- AP-Multi 的壓縮寫法跟其他檔風格不一致 → review 累
- Memory `project_fedalc_unification_plan.md` 早有 unification 規劃,但這次只做最 lite 的 utility 抽出,**不動 phase logic / aggregate_fit**(避開 high-risk 改動)

**Verification 過程(重要的副產品)**：

寫了 `verify_refactor_bit_exact.sh` 預期 baseline 跟 refactor branch 在同 config 下 `eval_metrics.tsv` 應該 byte-identical(因為抽出去的 4 個 function 都 pure deterministic)。

跑下去 **FAIL**:R1 全等但 R2 部分 partition 飄 ~1-2pp。

寫 `diagnose_flwr_determinism.sh` 在**同 branch 跑兩次同 config** 排除 refactor 影響。結果:**baseline 自己跟自己也飄一樣的 9 個 partition、一樣的 magnitude**。

→ **Refactor 沒問題,是 flwr local-simulation 在我們 setup 下的 inherent noise floor**。詳見 `notes/concepts/flwr_simulation_noise_floor.md`。

**Why R1 不飄 R2 才飄、為什麼剛好 9 個 partition**(都在 noise floor 那篇 note 裡):
- R1 是冷啟動,BLAS thread pool 跟 Ray worker pool 都還沒被 R1 計算「弄髒」
- 飄的 9 個都是「label 分佈正常」的 partition(model 在學);不飄的多半是「label 極端偏」的 partition(model 退化成預測 majority class,trivial solution → deterministic)
- 真正源頭:CPU BLAS multi-thread reduction、Ray worker scheduling、`set_seed` 沒涵蓋的層級

**Implication**:
- 之後 refactor / strategy 改動驗證**不能用 bit-exact**,要降級成「飄幅落在 noise floor 範圍 (~2pp) 內」的 statistical equivalence
- Paper main comparison 必須**多 seed mean ± std**,不能單 seed 對單 seed
- 想 byte-deterministic 需要 OMP_NUM_THREADS=1 + `torch.use_deterministic_algorithms(True)` 等(代價:既有 baseline 全失效 + 訓練變慢),**目前不啟用**

**Commit**:
- `7840e9c refactor(strategies): extract shared LoRA utilities to bert/lora_utils`(實際 net -152 LOC)
- 兩個 verification script 移到 `tools/` 並一起 commit

**Follow-up**:
- 推 `refactor/strategy-utils` branch(等用戶決定)
- Memory `project_fedalc_unification_plan.md` 描述的 full unification(統一成單一 strategy + config switch)還沒做,等 paper 實驗收斂後再評估

---

## [2026-05-04] meta | LR schedule design 調查 + paper notes 修正

**Files changed**：
- `notes/papers/related_papers.md`：修正 FedSA-LoRA「local epochs=10」→「10 batches (corrected)」,加上「Optimizer: SGD lr=0.02, no scheduler/no warmup」(查官方 yaml + FederatedScope defaults 確認);FedADC 條目補 cosine schedule 措辭註解(p.9 原文 "decays according to a cosine scheduler",最自然解讀是 within-round HF cosine,但 paper 沒釋出 code 無法 100% 確認)
- `notes/plans/next_steps.md`：新增 Task 6 (deferred) — LR schedule ablation (C1 vs C2 × α=0.3/0.5),paper appendix 用

**Why**：討論「FL LR schedule 該怎麼選」,從質疑現行 C1(cross-round cosine + within-round constant)「不是 LoRA 標準」開始,逐層查證後修正了 4 個 framing 錯誤:
1. **「LoRA 標準 = cosine + warmup」不正確** — 原 LoRA paper(microsoft/LoRA RoBERTa-large SST-2)用 **linear + warmup_ratio=0.06, lr=4e-4**,不是 cosine
2. **「FedSA-LoRA 用 10 epochs」不正確** — 官方 yaml 是 `local_update_steps: 10` + `batch_or_epoch: batch`,實際是 10 batches (= 10 optimizer steps);加上他們是 SGD constant LR,沒 scheduler 沒 warmup
3. **「C1 跟 FedADC 對齊」不正確** — FedADC paper 措辭最自然解讀是 **within-round cosine**(C2),不是 round-level cosine。所以 C1 是 Flower SFT template 獨有設計,沒 paper 直接對應
4. **「FL+LoRA 圈有 LR schedule convention」不成立** — paper 圈普遍 SGD constant 居多,沒人 ablate 過 within-round vs across-round cosine 差異

**核心發現** — within-round cosine(任何衰到 0 的 schedule)會 hurt 小 client:
- 200-example client(2 steps):step 0=peak, step 1≈0 → **50% step 浪費**
- 5000-example client(39 steps):末段 ~3 step 浪費 → ~8%
- C1 對所有 client 中性,**Flower template 巧合地是對 small client 最 friendly 的設計**

**衍生發現**:AdamW 在 FL 下的結構性問題 — optimizer state(m, v)每 round Trainer 重建時 reset → 第 1-3 step 不穩定。Centralized 用 warmup 補償,FL 1 epoch + 17 step 沒空間 warmup。FedSA-LoRA 用 SGD 直接迴避這個問題。真解法是 FedOpt/FedAdam(server-side momentum),不在當前 plan 範圍。

**Decision**:
- **不改現行 config**(C1 維持),理由:對 small client 友善 + 跟 Flower template 對齊 + 改了會 invalidate 既有 baseline 結果
- **C1 vs C2 ablation 列為 deferred future work**(`next_steps.md` Task 6),paper appendix 用

**Follow-up**：
- 未來若需要 ablate(paper appendix),`pyproject.toml` 切 `lr-schedule="constant"` + `lr-scheduler-type="cosine"` 即可,**0 行 code change**
- 若要從根本解決 AdamW state reset → 實作 FedOpt-style server-side optimizer(較大改動,paper extension scope)
- 完整討論記錄在 `~/.claude/plans/lora-buzzing-seal.md`(plan v4)

---

## [2026-04-24] experiment | Singleton 成因驗證 + label 分布 dump tool

**Files added**：
- `tools/dump_partition_stats.py`：per-partition stats dump tool（train/test size + label counts + ratios）。Output JSON + TSV，跟 `bert/dataset.py` 共用 Dirichlet + seeded split，保證跟 FL 訓練實際 seen 的分布一致
- `logs/partition_stats/qnli_c30_a0.5_s42.{json,tsv}`：QNLI α=0.5 完整 30 partition stats
- `notes/experiments/label_skew_singleton_analysis.md`：singleton 成因分析

**Why**：長期追蹤的 QNLI α=0.5 structural singleton (`pid_16` 全 30 輪孤立) 謎團解開 — **pid_16 的 label marginal 是 99.5% label_1，而其他高 accuracy client (pid_1/9/20) 是 99%+ label_0**。label 相反 → LoRA-B 學出反向分類器 → cosine(B) ≈ 負 → AP / Agglomerative 都會把它判為 outlier → singleton 是**結構性正確**，不是 clustering bug。

**Implication**：
- Singleton = feature not bug（在 inverse label marginal 的情境下）
- 強制 `min_cluster_size=2` merge to nearest 會把 pid_16 塞進 label_0 cluster → cluster-averaged B 正負相消 → pid_16 accuracy 預計從 99.5% 掉到 ~50-60%（未驗證）
- Paper 立場：personalized FL 下 singleton 是正確的 personalization decision

**Follow-up**：
- 跑剩 3 組：`for t in sst2 qnli; do for a in 0.3 0.5; do python tools/dump_partition_stats.py --task $t --alpha $a; done; done`
- 實測 min_cluster_size=2 merge 對 pid_16 accuracy 的代價
- `docs_index.md` 已加條目

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
