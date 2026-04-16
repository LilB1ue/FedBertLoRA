"""FedALC-AP-Multi Strategy: multi-component FedALC-AP targeting multi-task FL.

Naming convention (FedALC-* family):
  FedALC-AP         = AP clustering on full B (basic, simple baseline)
  FedALC-AP-LWC     = AP + LWC layer selection (silhouette warm-up, ablation)
  FedALC-AP-Multi   = AP + LWC + Hopkins adaptive trigger + cumulative ΔB +
                      freeze (this file; targets multi-task FL scenarios)
  FedALC-Spectral   = Spectral clustering on full B (future)
  FedALC-Agglo      = Agglomerative clustering on full B (future)

Core design shared with all FedALC-* variants:
  - A matrices: global FedAvg (shared subspace basis)
  - B matrices: per-cluster FedAvg (task-specific direction)
  - Others: per-client local (personalized classifier)

What FedALC-AP-Multi adds on top of FedALC-AP:
  - Clustering feature: cumulative ΔB (stable task vector signal)
  - Adaptive warm-up via Hopkins statistic (cluster tendency test)
  - Freeze mechanism (silhouette or stable rounds)
  - Built-in Metric B layer selection (dimensionality reduction so Hopkins
    works) — the essence of FedALC-AP-LWC is absorbed as preprocessing.

Layer selection (built-in preprocessing, not an orthogonal variant):
  Hopkins statistic is unreliable when feature dimension D is very high
  (curse of dimensionality + numerical overflow from u_dist**d when D≈50K).
  To keep Hopkins meaningful, Phase 0 uses Metric B (dissim × norm) to
  pick top-K LoRA layers, then computes Hopkins / AP clustering on the
  concatenated ΔB of those layers only. The standalone FedALCAPLWCStrategy
  is preserved as an ablation baseline (layer selection without Hopkins).

Three-phase structure:
  Phase 0 (adaptive warm-up):
    1. FedSA mode (A global, B local), no per-cluster aggregation yet
    2. Accumulate cumulative ΔB = running average of (B_r - B_init)
    3. Select top-K layers via Metric B on chosen feature
    4. Compute Hopkins on top-K layer ΔB subset
    5. Trigger Phase 1 when H > hopkins_threshold or warmup_max_rounds hit
  Phase 1 (clustering):
    AP clustering on top-K ΔB, per-cluster weighted avg of current B
  Phase 2 (frozen):
    Cluster assignment frozen, no more AP, reuse per-cluster aggregation

Theoretical framing:
  LoRA ΔW = BA is a rank-r task vector (Ilharco et al., ICLR 2023).
  FedALC-* = federated task vector clustering with shared basis.
  See notes/papers/task_vector_connection.md for details.

Key differences from baseline methods:
  vs FedALC-AP (simple baseline): Adaptive warm-up + cumulative ΔB + layer
    selection + freeze (not R1 cold-start on full B)
  vs FedLEASE (NeurIPS 2025): FedSA warm-up + adaptive trigger (not fixed E);
    shared A + clustered B (not per-cluster full LoRA + MoE router)
  vs HiLoRA (CVPR 2026): Single LoRA + shared A (not 3-tier with orthogonality)
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
    EvaluateIns,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def hopkins_statistic(X: np.ndarray, m: Optional[int] = None, seed: int = 42) -> float:
    """Compute Hopkins statistic for cluster tendency.

    H ≈ 1.0 → highly clusterable (real points cluster, uniform points don't)
    H ≈ 0.5 → random (no cluster tendency)
    H ≈ 0.0 → uniform (anti-cluster)

    Standard threshold: H > 0.75 indicates strong clustering tendency.

    IMPORTANT: This function assumes X has already been reduced to a
    low-dimensional representation (e.g., top-K LoRA layers or PCA).
    Applying Hopkins directly to very high-dimensional features (D > ~50)
    suffers from two issues:
      (1) curse of dimensionality: all pairwise distances become nearly
          equal, making H collapse to ~0.5 regardless of true structure;
      (2) numerical overflow: u_dist**d and w_dist**d blow up for D≈50K.
    Caller is responsible for providing low-D input.

    Reference:
        Banerjee & Davé (2004). Validating clusters using the Hopkins statistic.

    Args:
        X: (N, D) low-dimensional feature matrix (D should be modest).
        m: number of samples (default: min(N-1, N//2))
        seed: random seed

    Returns:
        H: float in [0, 1]. Returns 0.5 (no-tendency) for degenerate inputs.
    """
    rng = np.random.RandomState(seed)
    n, d = X.shape

    # Guard: Hopkins needs at least a few points to sample from
    if n < 4:
        return 0.5

    if m is None:
        m = max(1, min(n - 1, n // 2))
    m = min(m, n - 1)

    # Bounding box of X
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)

    # Sample m uniform random points in bounding box
    U = rng.uniform(x_min, x_max, size=(m, d))

    # Sample m real points from X (without replacement)
    indices = rng.choice(n, size=m, replace=False)
    P = X[indices]

    # Fit NN on X
    nn = NearestNeighbors(n_neighbors=2).fit(X)

    # u_i: distance from uniform random U[i] to nearest X
    u_dist, _ = nn.kneighbors(U, n_neighbors=1)
    u_dist = u_dist.flatten()

    # w_i: distance from sampled X P[i] to its 2nd nearest X (1st is itself)
    w_dist, _ = nn.kneighbors(P, n_neighbors=2)
    w_dist = w_dist[:, 1]

    u_sum = np.sum(u_dist ** d)
    w_sum = np.sum(w_dist ** d)

    if u_sum + w_sum == 0 or not np.isfinite(u_sum + w_sum):
        return 0.5  # degenerate case

    return float(u_sum / (u_sum + w_sum))


class FedALCAPMultiStrategy(FedAvg):
    """FedALC with Affinity Propagation clustering and built-in layer selection.

    See module docstring for the full three-phase design.

    Args:
        lora_param_keys: Ordered list of LoRA parameter names.
        ap_damping: AP damping factor.
        ap_max_iter: AP max iterations.
        hopkins_threshold: Hopkins H > threshold → exit warm-up.
        warmup_max_rounds: Hard cap on warm-up length.
        freeze_sil_threshold: Silhouette > threshold → freeze clustering.
        freeze_stable_rounds: Consecutive unchanged rounds → freeze clustering.
        layer_selection_k: Number of top-K layers to keep for Hopkins/AP.
        layer_reselect_every: 0 = one-shot (select once in Phase 0 round 1),
            N > 0 = reselect every N rounds while in Phase 0/1.
        layer_score_feature: "cumulative_delta_b" (default) or "current_b".
            Feature used by Metric B for layer scoring.
        use_wandb: Log clustering metrics to wandb.
        log_dir: Directory for clustering.jsonl.
        **kwargs: Passed to FedAvg.
    """

    VALID_LAYER_SCORE_FEATURES = ("cumulative_delta_b", "current_b")

    def __init__(
        self,
        lora_param_keys: Optional[List[str]] = None,
        ap_damping: float = 0.5,
        ap_max_iter: int = 100,
        hopkins_threshold: float = 0.75,
        warmup_max_rounds: int = 10,
        freeze_sil_threshold: float = 0.9,
        freeze_stable_rounds: int = 3,
        layer_selection_k: int = 10,
        layer_reselect_every: int = 1,
        layer_score_feature: str = "cumulative_delta_b",
        use_wandb: bool = False,
        log_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lora_param_keys = lora_param_keys or []
        self.ap_damping = ap_damping
        self.ap_max_iter = ap_max_iter
        self.hopkins_threshold = hopkins_threshold
        self.warmup_max_rounds = warmup_max_rounds
        self.freeze_sil_threshold = freeze_sil_threshold
        self.freeze_stable_rounds = freeze_stable_rounds
        self.layer_selection_k = layer_selection_k
        self.layer_reselect_every = layer_reselect_every
        if layer_score_feature not in self.VALID_LAYER_SCORE_FEATURES:
            raise ValueError(
                f"layer_score_feature must be one of {self.VALID_LAYER_SCORE_FEATURES}, "
                f"got {layer_score_feature!r}"
            )
        self.layer_score_feature = layer_score_feature
        self.use_wandb = use_wandb
        self.log_dir = log_dir

        # Phase: 0=warmup, 1=clustering, 2=frozen
        self.phase = 0

        # Per-client state
        self.client_b_matrices: Dict[str, List[np.ndarray]] = {}
        self.client_others: Dict[str, List[np.ndarray]] = {}

        # Cumulative ΔB tracking
        # {cid: (running_avg_delta_b: List[np.ndarray], count: int)}
        self.client_delta_b_cumulative: Dict[str, Tuple[List[np.ndarray], int]] = {}
        # Initial B per client. For standard LoRA B_init = 0; populated lazily
        # with np.zeros_like on first sighting of each client.
        self.client_b_init: Dict[str, List[np.ndarray]] = {}

        # Layer selection state
        # `selected_layer_indices` = current round's fresh pick (updated every round
        #   whenever `_maybe_reselect_layers` decides to).
        # `frozen_layer_indices` = snapshot at Phase 0 → Phase 1 trigger. Used for
        #   all downstream Hopkins / AP / freeze decisions once Phase 1 begins, so
        #   that those decisions stay in a fixed low-D feature space.
        # Post-trigger we keep updating `selected_layer_indices` purely for
        #   observation (logged side-by-side with frozen for drift analysis).
        self.selected_layer_indices: Optional[List[int]] = None
        self.selected_layer_scores: Optional[List[float]] = None
        self.frozen_layer_indices: Optional[List[int]] = None
        self.frozen_layer_scores: Optional[List[float]] = None
        self._rounds_since_reselect: int = 0

        # Global state
        self.global_a_matrices: Optional[List[np.ndarray]] = None
        self.global_b_matrices: Optional[List[np.ndarray]] = None
        self.global_others: Optional[List[np.ndarray]] = None

        # Freeze state
        self.frozen_cluster_groups: Optional[Dict[int, List[str]]] = None
        self.prev_cluster_groups: Optional[set] = None
        self.rounds_stable: int = 0

        # Log file
        self._cluster_log_path: Optional[str] = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self._cluster_log_path = os.path.join(log_dir, "clustering.jsonl")

        logger.info(
            f"FedALCAPMultiStrategy initialized "
            f"(hopkins_thresh={hopkins_threshold}, warmup_max={warmup_max_rounds}, "
            f"freeze_sil={freeze_sil_threshold}, layer_k={layer_selection_k}, "
            f"score_feature={layer_score_feature})"
        )

    # ── Shared utilities ──

    def _separate_a_b_others(self, parameters):
        a_params, b_params, other_params = [], [], []
        for key, param in zip(self.lora_param_keys, parameters):
            if "lora_A" in key:
                a_params.append(param)
            elif "lora_B" in key:
                b_params.append(param)
            else:
                other_params.append(param)
        return a_params, b_params, other_params

    def _reconstruct_parameters(self, a_params, b_params, other_params):
        result = []
        a_idx, b_idx, o_idx = 0, 0, 0
        for key in self.lora_param_keys:
            if "lora_A" in key:
                result.append(a_params[a_idx]); a_idx += 1
            elif "lora_B" in key:
                result.append(b_params[b_idx]); b_idx += 1
            else:
                result.append(other_params[o_idx]); o_idx += 1
        return result

    @staticmethod
    def _weighted_average(matrix_lists, weights):
        total = float(sum(weights))
        if total == 0:
            return [mat.copy() for mat in matrix_lists[0]] if matrix_lists else []
        factors = [w / total for w in weights]
        aggregated = [mat.copy() * factors[0] for mat in matrix_lists[0]]
        for i, mats in enumerate(matrix_lists[1:], start=1):
            for j, mat in enumerate(mats):
                aggregated[j] += mat * factors[i]
        return aggregated

    # ── Cumulative ΔB tracking ──

    def _update_cumulative_delta_b(
        self, client_b_list: List[List[np.ndarray]], client_cids: List[str], server_round: int
    ):
        """Update running average of ΔB for each client.

        ΔB_i = B_i^(r) - B_i^(init)

        For standard LoRA, PEFT initializes B = 0, so we set b_init = zeros
        on first sighting of each client. This makes round-1 ΔB = B_r1
        (the trained B matrix), not zero. For LoRA variants with B_init ≠ 0
        (e.g., PiSSA), this would need pre-round-1 hook — not supported yet.
        """
        # Lazily record zero-initialised b_init on first appearance
        for cid, b in zip(client_cids, client_b_list):
            if cid not in self.client_b_init:
                self.client_b_init[cid] = [np.zeros_like(b_layer) for b_layer in b]

        # Update running avg
        for cid, b in zip(client_cids, client_b_list):
            init = self.client_b_init[cid]
            delta = [b[i] - init[i] for i in range(len(b))]
            if cid in self.client_delta_b_cumulative:
                prev, count = self.client_delta_b_cumulative[cid]
                new_avg = [(prev[i] * count + delta[i]) / (count + 1) for i in range(len(b))]
                self.client_delta_b_cumulative[cid] = (new_avg, count + 1)
            else:
                self.client_delta_b_cumulative[cid] = (delta, 1)

    def _get_layer_score_source(
        self, client_b_list: List[List[np.ndarray]], client_cids: List[str]
    ) -> List[List[np.ndarray]]:
        """Return per-client per-layer matrices for layer scoring.

        Selector for `layer_score_feature`:
          - "cumulative_delta_b": use running average of ΔB (matches Hopkins feature)
          - "current_b": use current round's B (LWC-compatible behaviour)
        """
        if self.layer_score_feature == "current_b":
            return client_b_list
        # cumulative_delta_b
        source: List[List[np.ndarray]] = []
        for cid in client_cids:
            delta_b, _ = self.client_delta_b_cumulative[cid]
            source.append(delta_b)
        return source

    # ── Layer selection (Metric B = dissim × norm) ──

    def _compute_layer_scores(
        self, per_client_layers: List[List[np.ndarray]]
    ) -> List[float]:
        """Metric B per layer: (1 - mean pairwise cosine sim) × mean Frobenius norm.

        Higher score = layer where clients disagree strongly *and* have nontrivial
        magnitude. Used to pick top-K layers for downstream Hopkins / AP.
        """
        n_layers = len(per_client_layers[0])
        scores: List[float] = []
        for l in range(n_layers):
            vecs = np.stack([c[l].flatten() for c in per_client_layers])
            sim = cosine_similarity(vecs)
            mask = np.triu(np.ones(sim.shape, dtype=bool), k=1)
            if mask.sum() == 0:
                dissim = 0.0
            else:
                dissim = float(1.0 - sim[mask].mean())
            avg_norm = float(np.mean([np.linalg.norm(c[l]) for c in per_client_layers]))
            scores.append(dissim * avg_norm)
        return scores

    def _maybe_reselect_layers(
        self,
        client_b_list: List[List[np.ndarray]],
        client_cids: List[str],
        server_round: int,
    ) -> None:
        """Refresh selected_layer_indices per `layer_reselect_every` policy.

        Policy:
          - First call always selects.
          - reselect_every == 0 → one-shot (never reselect after first call).
          - reselect_every >  0 → reselect every N rounds.
        """
        need_select = self.selected_layer_indices is None
        if not need_select and self.layer_reselect_every > 0:
            self._rounds_since_reselect += 1
            if self._rounds_since_reselect >= self.layer_reselect_every:
                need_select = True

        if not need_select:
            return

        source = self._get_layer_score_source(client_b_list, client_cids)
        scores = self._compute_layer_scores(source)

        k = min(self.layer_selection_k, len(scores))
        top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        self.selected_layer_indices = sorted(top_k)
        self.selected_layer_scores = [scores[i] for i in self.selected_layer_indices]
        self._rounds_since_reselect = 0

        b_keys = [key for key in self.lora_param_keys if "lora_B" in key]
        selected_names = [b_keys[i] for i in self.selected_layer_indices]
        logger.info(
            f"Round {server_round}: layer selection → top-{k} "
            f"(feature={self.layer_score_feature}), first 3 = {selected_names[:3]}"
        )

    # ── Low-D feature matrix (top-K layer ΔB) ──

    def _active_layer_indices(self) -> Optional[List[int]]:
        """Indices used for Hopkins/AP computation.

        After Phase 0→1 trigger `frozen_layer_indices` is populated and is used
        permanently (so Phase 1 clustering + Phase 2 freeze decisions stay in a
        fixed feature space). Before that, the fresh `selected_layer_indices`
        from the current round is used.
        """
        if self.frozen_layer_indices is not None:
            return self.frozen_layer_indices
        return self.selected_layer_indices

    def _build_feature_matrix(
        self, client_cids: List[str]
    ) -> np.ndarray:
        """Stack cumulative ΔB of active top-K layers per client → (N, D_lowK)."""
        indices = self._active_layer_indices()
        assert indices is not None, \
            "Layer selection must run before building feature matrix"
        vectors = []
        for cid in client_cids:
            delta_b, _ = self.client_delta_b_cumulative[cid]
            selected = [delta_b[i].flatten() for i in indices]
            vectors.append(np.concatenate(selected))
        return np.stack(vectors)

    @staticmethod
    def _jaccard_overlap(a: Optional[List[int]], b: Optional[List[int]]) -> float:
        """Jaccard similarity between two layer-index sets.

        1.0 → identical picks, 0.0 → no overlap, 0.5 → half overlap.
        Returns 1.0 if both are empty/None; -1.0 if only one side is populated.
        """
        if not a and not b:
            return 1.0
        if not a or not b:
            return -1.0
        sa, sb = set(a), set(b)
        union = sa | sb
        if not union:
            return 1.0
        return round(len(sa & sb) / len(union), 4)

    def _compute_hopkins(self, client_cids: List[str]) -> float:
        """Hopkins statistic on low-D feature (top-K layer cumulative ΔB)."""
        X = self._build_feature_matrix(client_cids)
        return hopkins_statistic(X)

    # ── Clustering on low-D ΔB ──

    def _cumulative_count_stats(self, client_cids: List[str]) -> Dict[str, float]:
        """Summary statistics of cumulative ΔB accumulation counts across clients."""
        counts = [
            self.client_delta_b_cumulative[cid][1]
            for cid in client_cids
            if cid in self.client_delta_b_cumulative
        ]
        if not counts:
            return {"min": 0, "max": 0, "mean": 0.0}
        return {
            "min": int(min(counts)),
            "max": int(max(counts)),
            "mean": round(float(sum(counts) / len(counts)), 2),
        }

    def _cluster_on_delta_b(
        self,
        client_b_list: List[List[np.ndarray]],
        client_weights: List[int],
        client_cids: List[str],
        server_round: int,
        cid_to_pid: Optional[Dict[str, str]] = None,
    ) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, Scalar]]:
        """AP clustering on top-K layer cumulative ΔB; aggregate current B per cluster."""
        metrics: Dict[str, Scalar] = {}

        # Detect whether this is the very first clustering call (Phase 0 → 1 trigger)
        is_first_clustering = self.prev_cluster_groups is None

        feature_matrix = self._build_feature_matrix(client_cids)

        sim_matrix = cosine_similarity(feature_matrix)
        ap = AffinityPropagation(
            affinity="precomputed",
            damping=self.ap_damping,
            max_iter=self.ap_max_iter,
            random_state=42,
        )
        labels = ap.fit_predict(sim_matrix)
        ap_converged = ap.n_iter_ < self.ap_max_iter
        if not ap_converged:
            logger.warning(f"Round {server_round}: AP did not converge")
        n_clusters = len(set(labels))

        sil_score = -1.0
        if n_clusters > 1 and n_clusters < len(client_cids):
            sil_score = float(silhouette_score(feature_matrix, labels, metric="cosine"))

        # Build cluster membership (AP labels are exemplar indices, not 0..K-1)
        cluster_members: Dict[int, List[str]] = {}
        for i, label in enumerate(labels):
            cluster_members.setdefault(int(label), []).append(client_cids[i])

        cluster_sizes = [len(m) for _, m in sorted(cluster_members.items())]
        logger.info(
            f"Round {server_round}: AP phase={self.phase}, {n_clusters} clusters, "
            f"sil={sil_score:.4f}, sizes={cluster_sizes}, ap_iter={ap.n_iter_}"
        )

        # Per-cluster weighted avg of CURRENT B (not ΔB)
        client_b_aggregated: Dict[str, List[np.ndarray]] = {}
        for _, member_cids in cluster_members.items():
            member_indices = [client_cids.index(cid) for cid in member_cids]
            cluster_b = [client_b_list[i] for i in member_indices]
            cluster_w = [client_weights[i] for i in member_indices]
            agg_b = self._weighted_average(cluster_b, cluster_w)
            for cid in member_cids:
                client_b_aggregated[cid] = agg_b

        # Update stability tracking BEFORE freeze check so log reflects new state
        current_groups = set(frozenset(v) for v in cluster_members.values())
        if self.prev_cluster_groups is not None and current_groups == self.prev_cluster_groups:
            self.rounds_stable += 1
        else:
            self.rounds_stable = 0
        self.prev_cluster_groups = current_groups

        # Freeze decision + reason
        freeze_fired = False
        freeze_trigger: Optional[str] = None
        if sil_score >= self.freeze_sil_threshold:
            freeze_fired = True
            freeze_trigger = "silhouette"
        elif self.rounds_stable >= self.freeze_stable_rounds:
            freeze_fired = True
            freeze_trigger = "stable_rounds"

        if freeze_fired:
            self.phase = 2
            self.frozen_cluster_groups = {
                int(k): list(v) for k, v in cluster_members.items()
            }
            logger.info(
                f"Round {server_round}: → Phase 2 (frozen), "
                f"trigger={freeze_trigger}, sil={sil_score:.4f}, stable={self.rounds_stable}"
            )

        # Log (after all state updates so we record the final decision)
        if self._cluster_log_path:
            b_keys = [key for key in self.lora_param_keys if "lora_B" in key]
            active_indices = self._active_layer_indices() or []
            active_b_keys = [b_keys[i] for i in active_indices]
            observation_b_keys = [b_keys[i] for i in (self.selected_layer_indices or [])]

            # Compare this-round observation pick against frozen pick (drift analysis)
            layer_drift_jaccard = self._jaccard_overlap(
                self.selected_layer_indices, self.frozen_layer_indices
            )

            log_entry = {
                "round": server_round,
                "phase": 1,  # this method runs for Phase 1 clustering (or Phase 0→1 trigger round)
                "n_clusters": n_clusters,
                "silhouette_score": round(sil_score, 6),
                "rounds_stable": self.rounds_stable,
                "is_first_clustering": is_first_clustering,
                "ap_converged": ap_converged,
                "ap_n_iter": int(ap.n_iter_),
                "freeze_fired": freeze_fired,
                "freeze_trigger": freeze_trigger,
                "cumulative_count": self._cumulative_count_stats(client_cids),
                "clustering_features": {
                    "method": "top_k_cumulative_delta_b",
                    # Active (used for clustering)
                    "active_layer_indices": active_indices,
                    "active_b_keys": active_b_keys,
                    "layer_score_feature": self.layer_score_feature,
                    "n_params_clustering": int(feature_matrix.shape[1]),
                    # Observation-only: this round's fresh pick (for drift analysis)
                    "observation_layer_indices": self.selected_layer_indices,
                    "observation_b_keys": observation_b_keys,
                    "observation_layer_scores": [
                        round(float(s), 4) for s in (self.selected_layer_scores or [])
                    ],
                    "layer_drift_jaccard_vs_frozen": layer_drift_jaccard,
                },
                "clusters": {
                    str(k): (
                        sorted([cid_to_pid.get(c, c) for c in v],
                               key=lambda x: int(x) if x.isdigit() else x)
                        if cid_to_pid else v
                    )
                    for k, v in sorted(cluster_members.items())
                },
            }
            with open(self._cluster_log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        metrics["n_clusters"] = n_clusters
        metrics["silhouette_score"] = sil_score
        metrics["cluster_sizes"] = str(cluster_sizes)
        metrics["ap_converged"] = int(ap_converged)
        metrics["ap_n_iter"] = int(ap.n_iter_)
        metrics["rounds_stable"] = self.rounds_stable
        if freeze_fired:
            metrics["freeze_trigger"] = freeze_trigger

        return client_b_aggregated, metrics

    # ── Main aggregate_fit ──

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        # Collect
        client_a_list, client_b_list, client_other_list = [], [], []
        client_weights, client_cids = [], []
        cid_to_pid: Dict[str, str] = {}
        fit_metrics_list = []

        for client, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            a_params, b_params, other_params = self._separate_a_b_others(ndarrays)
            client_a_list.append(a_params)
            client_b_list.append(b_params)
            client_other_list.append(other_params)
            client_weights.append(fit_res.num_examples)
            client_cids.append(client.cid)
            cid_to_pid[client.cid] = str(fit_res.metrics.get("partition_id", client.cid))
            fit_metrics_list.append((fit_res.num_examples, fit_res.metrics))

        # A: always global
        agg_a = self._weighted_average(client_a_list, client_weights)
        self.global_a_matrices = agg_a

        # Others: always local
        for cid, oth in zip(client_cids, client_other_list):
            self.client_others[cid] = oth
        self.global_others = self._weighted_average(client_other_list, client_weights)

        # Global B avg (for server eval fallback)
        self.global_b_matrices = self._weighted_average(client_b_list, client_weights)

        # Update cumulative ΔB in every phase (Phase 2 keeps it updated for
        # drift observation even though clustering itself is frozen)
        self._update_cumulative_delta_b(client_b_list, client_cids, server_round)

        clustering_metrics: Dict[str, Scalar] = {}

        # ── Phase 0: Adaptive warm-up ──
        if self.phase == 0:
            # B stays local (FedSA mode)
            for cid, b in zip(client_cids, client_b_list):
                self.client_b_matrices[cid] = b

            # Layer selection (fresh each round in Phase 0; frozen at trigger)
            self._maybe_reselect_layers(client_b_list, client_cids, server_round)

            # Compute Hopkins on top-K layer ΔB
            H = self._compute_hopkins(client_cids)
            clustering_metrics["hopkins"] = H
            clustering_metrics["phase"] = 0
            clustering_metrics["n_params_clustering"] = int(
                self._build_feature_matrix(client_cids).shape[1]
            )

            logger.info(
                f"Round {server_round}: Phase 0 (warm-up), Hopkins={H:.4f} "
                f"(threshold={self.hopkins_threshold}, max_rounds={self.warmup_max_rounds})"
            )

            # Determine trigger before writing the log so we can annotate the row
            trigger_fired = False
            trigger_reason: Optional[str] = None
            if H > self.hopkins_threshold:
                trigger_fired = True
                trigger_reason = "hopkins_threshold"
            elif server_round >= self.warmup_max_rounds:
                trigger_fired = True
                trigger_reason = "max_rounds"

            # Snapshot: freeze selected layers at trigger (used by Phase 1/2 clustering)
            if trigger_fired and self.frozen_layer_indices is None:
                self.frozen_layer_indices = list(self.selected_layer_indices or [])
                self.frozen_layer_scores = list(self.selected_layer_scores or [])
                logger.info(
                    f"Round {server_round}: frozen top-{len(self.frozen_layer_indices)} "
                    f"layer indices = {self.frozen_layer_indices}"
                )

            if self._cluster_log_path:
                b_keys = [key for key in self.lora_param_keys if "lora_B" in key]
                selected_b_keys = [b_keys[i] for i in (self.selected_layer_indices or [])]
                log_entry = {
                    "round": server_round,
                    "phase": 0,
                    "hopkins": round(float(H), 6),
                    "trigger_fired": trigger_fired,
                    "trigger_reason": trigger_reason,
                    "cumulative_count": self._cumulative_count_stats(client_cids),
                    "clustering_features": {
                        "method": "warmup_fedsa",
                        "selected_layer_indices": self.selected_layer_indices,
                        "selected_b_keys": selected_b_keys,
                        "layer_scores_topk": [
                            round(float(s), 4)
                            for s in (self.selected_layer_scores or [])
                        ],
                        "layer_score_feature": self.layer_score_feature,
                        "n_params_hopkins": clustering_metrics["n_params_clustering"],
                    },
                    "clusters": {},
                }
                with open(self._cluster_log_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

            if trigger_fired:
                self.phase = 1
                logger.info(
                    f"Round {server_round}: → Phase 1 (clustering), reason={trigger_reason}"
                )
                # Immediately do clustering this round using frozen layers
                clustered_b, cl_metrics = self._cluster_on_delta_b(
                    client_b_list, client_weights, client_cids, server_round,
                    cid_to_pid=cid_to_pid,
                )
                self.client_b_matrices = clustered_b
                clustering_metrics.update(cl_metrics)
                clustering_metrics["phase"] = self.phase

        # ── Phase 1: Clustering on frozen top-K cumulative ΔB ──
        elif self.phase == 1:
            # Continue observation-only reselect so we can log drift vs frozen
            self._maybe_reselect_layers(client_b_list, client_cids, server_round)

            clustered_b, cl_metrics = self._cluster_on_delta_b(
                client_b_list, client_weights, client_cids, server_round,
                cid_to_pid=cid_to_pid,
            )
            self.client_b_matrices = clustered_b
            clustering_metrics.update(cl_metrics)
            clustering_metrics["phase"] = self.phase

        # ── Phase 2: Frozen ──
        elif self.phase == 2:
            # Observation-only: keep selecting top-K each round so we can log
            # drift vs. frozen pick. Does not affect clustering decisions.
            self._maybe_reselect_layers(client_b_list, client_cids, server_round)

            cluster_groups: Dict[int, List[str]] = {}
            known_cids: set = set()
            if self.frozen_cluster_groups:
                for cl, member_cids in self.frozen_cluster_groups.items():
                    present = [c for c in member_cids if c in client_cids]
                    if present:
                        cluster_groups[cl] = present
                        known_cids.update(present)

            n_fallback = 0
            # Fallback: no frozen group survived → treat all current clients as one group
            if not cluster_groups:
                cluster_groups[0] = list(client_cids)
                known_cids.update(client_cids)
                n_fallback = len(client_cids)
                logger.warning(
                    f"Round {server_round}: Phase 2 frozen groups empty, "
                    f"falling back to single-cluster FedAvg over all clients"
                )
            else:
                # Assign unknown clients to the largest surviving cluster
                for cid in client_cids:
                    if cid not in known_cids:
                        largest = max(cluster_groups.keys(),
                                      key=lambda k: len(cluster_groups[k]))
                        cluster_groups[largest].append(cid)
                        n_fallback += 1

            # Per-cluster avg with current B
            client_b_aggregated: Dict[str, List[np.ndarray]] = {}
            for cl, member_cids in cluster_groups.items():
                if not member_cids:
                    continue
                member_indices = [client_cids.index(cid) for cid in member_cids]
                cluster_b = [client_b_list[i] for i in member_indices]
                cluster_w = [client_weights[i] for i in member_indices]
                agg_b = self._weighted_average(cluster_b, cluster_w)
                for cid in member_cids:
                    client_b_aggregated[cid] = agg_b

            self.client_b_matrices = client_b_aggregated

            n_clusters = sum(1 for m in cluster_groups.values() if m)
            cluster_sizes = [len(m) for m in cluster_groups.values() if m]
            clustering_metrics["n_clusters"] = n_clusters
            clustering_metrics["phase"] = 2
            clustering_metrics["cluster_sizes"] = str(cluster_sizes)
            clustering_metrics["n_fallback_assignments"] = n_fallback

            if self._cluster_log_path:
                b_keys = [key for key in self.lora_param_keys if "lora_B" in key]
                frozen_b_keys = [b_keys[i] for i in (self.frozen_layer_indices or [])]
                observation_b_keys = [b_keys[i] for i in (self.selected_layer_indices or [])]
                layer_drift_jaccard = self._jaccard_overlap(
                    self.selected_layer_indices, self.frozen_layer_indices
                )
                log_entry = {
                    "round": server_round,
                    "phase": 2,
                    "n_clusters": n_clusters,
                    "silhouette_score": -1.0,
                    "n_fallback_assignments": n_fallback,
                    "cumulative_count": self._cumulative_count_stats(client_cids),
                    "clustering_features": {
                        "method": "frozen",
                        "frozen_layer_indices": self.frozen_layer_indices,
                        "frozen_b_keys": frozen_b_keys,
                        "observation_layer_indices": self.selected_layer_indices,
                        "observation_b_keys": observation_b_keys,
                        "observation_layer_scores": [
                            round(float(s), 4)
                            for s in (self.selected_layer_scores or [])
                        ],
                        "layer_drift_jaccard_vs_frozen": layer_drift_jaccard,
                        "layer_score_feature": self.layer_score_feature,
                    },
                    "clusters": {
                        str(k): sorted(
                            [cid_to_pid.get(c, c) for c in v],
                            key=lambda x: int(x) if x.isdigit() else x
                        )
                        for k, v in cluster_groups.items() if v
                    },
                }
                with open(self._cluster_log_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

            logger.info(
                f"Round {server_round}: Phase 2 (frozen), {n_clusters} clusters, "
                f"sizes={cluster_sizes}, fallback={n_fallback}"
            )

        # Server eval global params
        combined = self._reconstruct_parameters(
            agg_a, self.global_b_matrices, self.global_others
        )

        # Aggregate metrics
        metrics_aggregated: Dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics_list)

        for k, v in clustering_metrics.items():
            metrics_aggregated[f"clustering/{k}"] = v

        if self.use_wandb:
            import wandb
            wandb_metrics = {f"clustering/{k}": v for k, v in clustering_metrics.items()
                            if isinstance(v, (int, float))}
            wandb_metrics["round"] = server_round
            wandb.log(wandb_metrics, step=server_round)

        return ndarrays_to_parameters(combined), metrics_aggregated

    # ── Parameter distribution ──

    def configure_fit(
        self, server_round, parameters, client_manager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)

        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        if self.global_a_matrices is None:
            return [(client, FitIns(parameters, dict(config))) for client in clients]

        fit_ins_list = []
        for client in clients:
            cid = client.cid
            client_b = self.client_b_matrices.get(cid, self.global_b_matrices)
            client_oth = self.client_others.get(cid, self.global_others)

            personalized = self._reconstruct_parameters(
                self.global_a_matrices, client_b, client_oth
            )
            personalized_params = ndarrays_to_parameters(personalized)
            fit_ins_list.append((client, FitIns(personalized_params, dict(config))))

        return fit_ins_list

    def configure_evaluate(
        self, server_round, parameters, client_manager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        config = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)

        sample_size, min_num_clients = self.num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        if self.global_a_matrices is None:
            return [(client, EvaluateIns(parameters, dict(config))) for client in clients]

        eval_ins_list = []
        for client in clients:
            cid = client.cid
            client_b = self.client_b_matrices.get(cid, self.global_b_matrices)
            client_oth = self.client_others.get(cid, self.global_others)

            personalized = self._reconstruct_parameters(
                self.global_a_matrices, client_b, client_oth
            )
            personalized_params = ndarrays_to_parameters(personalized)
            eval_ins_list.append((client, EvaluateIns(personalized_params, dict(config))))

        return eval_ins_list
