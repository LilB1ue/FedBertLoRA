# 20 Newsgroups, Agglo-LWC, And Checkpoint Changes Summary

Snapshot based on `git status --short` before creating this file.

After this report is created, `20newsgroups_agglo_checkpoint_changes_summary.md`
is also an untracked file.

## Tracked Modified Files

| File | Summary |
|---|---|
| `bert/client_app.py` | Generalizes wording from GLUE-only to sequence classification; replaces direct GLUE metric loading with `load_metric_for_task()`; reads `total_rounds` from server config so `checkpoint-save-policy="selective"` saves both first and last client/received checkpoints. |
| `bert/dataset.py` | Reworks dataset loading around `bert.task_registry`; keeps legacy `GLUE_TASK_CONFIG` compatibility; adds `20newsgroups` support through generic task specs; casts integer labels to `ClassLabel` for stratified splits; adds clearer Dirichlet retry-limit diagnostics; central data loading now uses task-specific train/eval splits. |
| `bert/experiment_config.py` | Adds `20newsgroups` as an accuracy-compatible checkpoint metric task; changes selective checkpoint retention from first-only to first-and-last; extends `BestCheckpointTracker` with protected rounds so first/final eval checkpoints are not pruned while still keeping `best_round.json`. |
| `bert/fedalc_agglo_lwc_strategy.py` | Changes FedALC-Agglo-LWC from fixed warm-up at round E to a probing state with normed Metric-B top-K layer selection; adds `layer_overlap_trigger`; freezes when consecutive top-K overlap passes threshold or max warm-up is reached; updates clustering JSONL schema with `state`, `event`, overlap, trigger reason, normed score mode, and frozen reuse logging. |
| `bert/lora_utils.py` | Extends `compute_layer_scores()` with `score_mode`; preserves existing raw Metric-B default and adds `metric-b-normed`, dividing Frobenius magnitude by `sqrt(numel)` to reduce tensor-size bias. |
| `bert/server_app.py` | Replaces direct GLUE metric loading with `load_metric_for_task()`; passes `total_rounds` into fit/evaluate config; configures best checkpoint tracking to protect round 1 and final round; forwards `layer-overlap-trigger` into `FedALCAggloLWCStrategy`. |
| `centralized_learning/train.py` | Generalizes centralized training from GLUE-only to sequence-classification tasks; task choices come from `get_task_names()`; metrics load through `load_metric_for_task()`, enabling `20newsgroups`. |
| `pyproject.toml` | Documents `20newsgroups` as a valid `task-name`; notes `min-partition-size=128` for 20NG alpha 0.3; changes Agglo-LWC config comments from fixed warm-up to probing warm-up; adds `layer-overlap-trigger = 7`. |

## Untracked New Files

| File | Summary |
|---|---|
| `20news.md` | Implementation plan/documentation for adding `20newsgroups` support, including registry design, dataset details, metric routing, partition diagnostics, and verification commands. |
| `AGENTS.md` | Repo-level working instructions for agents: project rules, strategy map, commands, evaluation conventions, log layout, FedALC naming discipline, and checkpoint conventions. |
| `agglo_lwc_normed_metric_b_plan.md` | Plan for changing FedALC-Agglo-LWC to normed Metric-B layer scoring and layer-overlap-triggered probing/freeze. |
| `anal_plan.md` | Documentation of a completed log-analysis workflow for recent FedSA all-checkpoint smoke logs, including client-side metric summaries and LoRA-B clustering analysis outputs. |
| `bert/task_registry.py` | New task registry with `TaskSpec`; defines GLUE tasks plus `20newsgroups`; centralizes dataset names, text fields, label fields, train/eval splits, label counts, and metric loading. |
| `centralized_20news.md` | Plan/usage notes for centralized LoRA training on 20 Newsgroups via the existing centralized training framework. |
| `centralized_learning/run_20newsgroups.sh` | Dedicated centralized 20 Newsgroups launcher using RoBERTa-large LoRA with conservative defaults: seq length 256, batch size 16, grad accumulation 4, lr 1e-4, 10 epochs. |
| `dissusion_with_gpt_5_pro.md` | Research discussion notes around clustering choices, Agglomerative/KMeans/Spectral behavior, weighted/unweighted metrics, and personalized aggregation direction. |
| `experiment_without_serverside_lr_code_review.md` | Code-review note for the working-tree checkpoint/server-LR related changes, including risks and suggested fixes. |
| `personalized_aggregation_design.html` | HTML design/proposal page for personalized aggregation, including method framing and evaluation conventions. |
| `reliability_aware_personalized_aggregation_plan.html` | HTML plan for a reliability-aware personalized aggregation variant, tentatively framed as `FedALC-Agglo-RA`. |
| `run_20newsgroups_alpha03.sh` | FL launcher for 20 Newsgroups alpha 0.3, defaulting to `fedalc-agglo-lwc`, `min-partition-size=128`, seq length 256, and configurable warm-up/overlap/LoRA settings. |
| `run_20newsgroups_baselines_alpha05.sh` | Batch FL launcher for 20 Newsgroups alpha 0.5 baselines: `fedavg`, `fedsa`, and `ffa`, with shared timestamp and selective checkpoint policy. |
| `run_20newsgroups_fedsa_10round_allckpt.sh` | FedSA 20 Newsgroups launcher for post-hoc analysis; runs 10 rounds by default with `checkpoint-save-policy='all'` to retain every round's client checkpoints. |
| `run_20newsgroups_ffa_alpha05.sh` | New FFA-only 20 Newsgroups alpha 0.5 launcher, matching the baseline script defaults; W&B defaults to true; supports passing an explicit timestamp to fill the missing `ffa` run from a prior batch. |

## Ignored Local Tests

`git status --ignored --short tests` reports `!! tests/`, so the local tests
under `tests/` are ignored and are not part of the normal untracked-file list.

Notable local tests currently present:

- `tests/test_experiment_config.py`: includes coverage for first/best/last selective checkpoint behavior and protected best-checkpoint rounds.
- `tests/test_task_registry.py`: covers GLUE task preservation and `20newsgroups` task spec/metric loading.
- `tests/test_dataset_class_label.py`: covers integer-label-to-`ClassLabel` casting and partition retry-limit diagnostics.
- `tests/test_fedalc_agglo_lwc_strategy.py`: covers normed Metric-B scoring and Agglo-LWC layer-overlap trigger behavior.

## Verification Already Run

Focused unit tests passed:

```bash
conda run -n exp-flower-bert python -m unittest \
  tests.test_experiment_config \
  tests.test_task_registry \
  tests.test_dataset_class_label \
  tests.test_fedalc_agglo_lwc_strategy
```

Result: `22 tests` passed.
