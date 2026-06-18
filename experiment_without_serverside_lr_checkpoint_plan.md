# Experiment Without Server-Side LR And Checkpoint Retention Plan

> Branch: `experiment-without-serverside-lr`
>
> Scope: planning only. This note describes intended changes; no code has been changed by this plan.

## Goal

Align the FL training protocol with papers that do not use a cross-round server-side learning-rate schedule, while reducing checkpoint storage without losing the model states needed for analysis.

## Current Behavior

Training protocol is configured in `pyproject.toml`.

- Server-side `lr-schedule = "cosine"` currently computes one learning rate per FL round in `bert/server_app.py`.
- Client-side `lr-scheduler-type = "constant"` currently passes that per-round LR into Hugging Face `Trainer`.
- `grad-accum-steps = 4` with `batch-size = 32` gives effective local batch size 128.
- `dirichlet-alpha = 0.5` is the default.
- `min_partition_size` is hard-coded to `10` in `bert/dataset.py`; a config key named `min-partition-size` would currently be ignored.

Checkpoint saving currently happens in three places:

- `client_checkpoints/round_R/client_ID/`: saved after local training and before aggregation for every strategy.
- `received_checkpoints/round_R/client_ID/`: saved before local training for every non-FedAvg strategy.
- `global_checkpoints/round_R/`: saved only for FedAvg server-side global adapter.

Important: `eval_metrics.tsv` corresponds to the model sent in `configure_evaluate()` after round `R` aggregation. It does not correspond exactly to `client_checkpoints/round_R`, which is pre-aggregation.

## Proposed Training Protocol Changes

Use this as the paper-aligned experiment protocol:

```toml
learning-rate = 0.001
lr-schedule = "constant"
lr-scheduler-type = "cosine"
batch-size = 32
grad-accum-steps = 1
local-epochs = 1
min-partition-size = 256
test-split-ratio = 0.2
dirichlet-alpha = 0.3
```

Interpretation:

- The server sends the same base LR every round.
- Each client uses local cosine scheduling inside its one local epoch.
- Effective local batch size becomes 32.
- Strong non-IID partitioning uses alpha 0.3.
- Minimum client partition size becomes configurable instead of hard-coded.

Task caveat:

- This is reasonable for current FL tasks (`sst2`, `qnli`, `mnli`, `qqp`).
- It is not reasonable for FL `rte` with 30 clients because `30 * 256 = 7680`, larger than the RTE train set.

## Proposed Checkpoint Policy

Do not delete the old save logic. Preserve it behind a config option.

Add a config key:

```toml
checkpoint-save-policy = "all"  # "all" | "selective"
```

Policy meanings:

- `all`: current behavior. Save every received/client/global checkpoint exactly as before.
- `selective`: save only the checkpoints needed for analysis and best-model retention.

Suggested selective policy:

```text
Always keep:
- eval_metrics.tsv
- fit_metrics.tsv
- clustering.jsonl for FedALC-family methods

Save adapter checkpoints:
- round_1 received checkpoint, for non-FedAvg methods
- round_1 client checkpoint, for all methods
- best_checkpoints/round_R/client_ID/, aligned with eval_metrics.tsv
```

Do not keep final checkpoint by default.

## Best Checkpoint Definition

Use client-side evaluation as the source of truth.

Best round should be selected by:

```text
primary: unweighted mean client accuracy from eval_metrics.tsv
secondary/reporting: weighted mean by num_examples
```

For tasks with multiple metrics:

- `accuracy` is the default best metric.
- For QQP, still use `accuracy` unless we explicitly decide to optimize `f1`.

The saved best model must be the post-aggregation, evaluate-time personalized model:

```text
round R aggregation finished
strategy configure_evaluate() prepares personalized parameters
clients evaluate those parameters
eval_metrics.tsv records those results
best_checkpoints/round_R/client_ID/ must match those parameters
```

Do not use `client_checkpoints/round_R` as the best checkpoint. That directory stores local post-training, pre-aggregation adapters.

## Implementation Outline

### 1. Make `min-partition-size` real

Modify `bert/client_app.py`:

- Read `min-partition-size` from config.
- Pass it into `load_data(...)`.

Modify `bert/dataset.py`:

- Add a `min_partition_size` argument to `load_data(...)`.
- Include it in the dataset cache key.
- Pass it into `DirichletPartitioner(min_partition_size=...)`.

Reason for cache-key change:

- Partitions generated with min size 10 and 256 are different datasets and must not share cache.

### 2. Add explicit checkpoint policy config

Modify `pyproject.toml`:

```toml
checkpoint-save-policy = "all"  # "all" preserves legacy every-round saving; "selective" saves R1 + best only
checkpoint-best-metric = "accuracy"
checkpoint-best-mode = "max"
```

Use `all` as the default if backward compatibility is more important.

Use `selective` as the default only if this branch is intended to be an experiment-only branch.

### 3. Preserve original checkpoint behavior

Modify `bert/client_app.py` without deleting the old behavior:

- Keep existing every-round save path under `checkpoint-save-policy == "all"`.
- Add a short comment explaining this is the legacy full-retention path.
- Add selective guards:
  - Save `received_checkpoints` only when `current_round == 1`.
  - Save `client_checkpoints` only when `current_round == 1`.

Do not comment out large code blocks. Prefer a small policy branch so both behaviors remain runnable.

### 4. Save evaluation-aligned best checkpoints

Add a server-side mechanism for `best_checkpoints`.

Required behavior:

- The candidate checkpoint must be created from the same parameters sent by `configure_evaluate()`.
- After evaluation metrics are aggregated, keep the candidate only if the round is the new best.
- If a newer best appears, remove the previous `best_checkpoints/round_old/` directory.
- If the current round is not best, remove its candidate directory.

Suggested directory layout:

```text
logs/{ts}_{mode}_a{alpha}/{task}_{mode}_a{alpha}/best_checkpoints/
├── best_round.json
└── round_R/
    ├── client_0/
    ├── client_1/
    └── ...
```

Suggested `best_round.json`:

```json
{
  "round": 7,
  "metric": "accuracy",
  "selection": "unweighted_mean",
  "value": 0.9142,
  "weighted_mean": 0.9178,
  "num_clients": 30
}
```

### 5. Keep metrics and clustering logs unchanged

Do not reduce:

- `eval_metrics.tsv`
- `fit_metrics.tsv`
- `server_eval.tsv`
- `clustering.jsonl`

Reason:

- `eval_metrics.tsv` is the main result source.
- `fit_metrics.tsv` is needed for local training diagnostics.
- `clustering.jsonl` is necessary for FedALC-family analysis.
- `server_eval.tsv` remains a sanity check, not the main metric.

## Open Design Decision

The main implementation choice is how to write `best_checkpoints` without rebuilding large models too often.

Preferred option:

- Save a candidate evaluate-time checkpoint for the current round.
- Use aggregated client eval metrics to decide whether it becomes the best.
- Prune non-best candidates immediately.

Fallback option:

- Keep only metadata for the best round first.
- Add actual adapter saving later.

The preferred option is better for reproducibility, but needs careful implementation because best is only known after evaluation returns.

## Validation Plan

Use smoke runs only. Do not launch full expensive experiments casually.

Suggested checks:

1. Run a 1-2 round `fedsa` smoke test with `checkpoint-save-policy = "selective"`.
2. Confirm only round 1 `received_checkpoints` and `client_checkpoints` are written.
3. Confirm `eval_metrics.tsv` and `fit_metrics.tsv` still have all rounds.
4. Confirm `best_checkpoints/best_round.json` exists after evaluation.
5. Confirm best checkpoint round matches the highest unweighted mean accuracy computed from `eval_metrics.tsv`.
6. Run one FedALC-family smoke test and confirm `clustering.jsonl` is unchanged.
7. Run with `checkpoint-save-policy = "all"` and confirm the old every-round checkpoint layout still works.

## Recommended Commit Split

Do not stage or commit without explicit approval.

When ready, split commits like this:

```text
feat(data): make partition minimum size configurable
feat(checkpoints): add selective checkpoint retention policy
docs(experiments): document paper-aligned protocol
```

