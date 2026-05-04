"""FedALC-Agglo-LWC Strategy: Agglomerative layer-wise clustering with FedLEASE-style fixed warm-up.

Three-phase strategy targeting single-task FL:
  Phase 0 (R1..R_{E-1}): FedSA warm-up — A global, B per-client local
  Phase 1 trigger (R_E): one-shot top-K layer selection + Agglomerative K sweep
                         → freeze (K*, assignment, layer mask)
  Phase 2 (R_{E+1}..): frozen reuse of cluster groups

Design differences vs AP family:
  * sklearn AgglomerativeClustering with K sweep 2..10 (silhouette-selected K*)
  * Fixed E-round warm-up (no silhouette trigger, no Hopkins)
  * Server-side evaluate_fn is intentionally skipped (filtered out at dispatch);
    aggregate_fit returns an empty Parameters placeholder to satisfy Flower's
    API. Main metric is client-side eval_metrics.tsv
    (see .claude/rules/evaluation_metric.md).
  * K sweep records per-K (sil, sizes, n_singletons) to clustering.jsonl
    (`sweep_results` field) so downstream analysis can diagnose singleton causes.

Assumptions:
  * **fraction-fit = 1.0 at R_E is required**. Phase 1 trigger clusters on the
    clients present that round; any absent client ends up routed to the largest
    cluster by Phase 2's fallback. Guard: a warning is logged if Phase 1 trigger
    runs without all clients participating.
  * **K=1 fallback** (silhouette below `agglo_min_silhouette`): strategy
    degenerates to FedSA-with-global-B — all clients share one averaged B. This
    is the intended safe fallback when no clustering structure is found.
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
# Cached empty Parameters for aggregate_fit return: Flower's API expects a
# Parameters object; we explicitly return an empty one to signal "no global
# model maintained by this strategy" rather than reusing a client's trained
# params (which would be misleading semantics).
_EMPTY_PARAMETERS = Parameters(tensors=[], tensor_type="")

from bert.lora_utils import (
    compute_layer_scores,
    reconstruct_parameters,
    separate_a_b_others,
    weighted_average,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FedALCAggloLWCStrategy(FedAvg):
    """Agglomerative LWC with fixed E-round FedSA warm-up and one-shot freeze.

    Args:
        lora_param_keys: Ordered list of LoRA parameter names from the model.
        warmup_rounds: Phase 0 length E. At round E, cluster + freeze happens.
        k_min, k_max: Agglomerative K sweep range (inclusive).
        agglo_linkage: sklearn linkage. `ward` rejected (requires Euclidean).
        agglo_min_silhouette: If best sweep silhouette < this, fallback to K=1.
        layer_selection_k: top-K layers by Metric B (dissim × norm) for clustering.
        use_wandb: whether to log to wandb.
        log_dir: directory for clustering.jsonl.
        **kwargs: forwarded to FedAvg. Server-side evaluate_fn should already be
                  filtered out by the dispatcher in server_app.py.
    """

    def __init__(
        self,
        lora_param_keys: Optional[List[str]] = None,
        warmup_rounds: int = 5,
        k_min: int = 2,
        k_max: int = 10,
        agglo_linkage: str = "average",
        agglo_min_silhouette: float = 0.0,
        layer_selection_k: int = 10,
        use_wandb: bool = False,
        log_dir: Optional[str] = None,
        **kwargs,
    ):
        if agglo_linkage == "ward":
            raise ValueError(
                "ward linkage requires Euclidean metric; use 'average', "
                "'complete', or 'single' with cosine distance."
            )
        super().__init__(**kwargs)
        self.lora_param_keys = lora_param_keys or []
        self.warmup_rounds = warmup_rounds
        self.k_min = k_min
        self.k_max = k_max
        self.agglo_linkage = agglo_linkage
        self.agglo_min_silhouette = agglo_min_silhouette
        self.layer_selection_k = layer_selection_k
        self.use_wandb = use_wandb
        self.log_dir = log_dir

        # Per-client state (populated in aggregate_fit, consumed in configure_fit)
        self.client_b_matrices: Dict[str, List[np.ndarray]] = {}
        self.client_others: Dict[str, List[np.ndarray]] = {}

        # Global state used by configure_fit to dispatch personalized params.
        # global_b_matrices / global_others are only fallbacks for clients that
        # haven't been seen yet (e.g., late-arriving under fraction-fit<1.0).
        self.global_a_matrices: Optional[List[np.ndarray]] = None
        self.global_b_matrices: Optional[List[np.ndarray]] = None
        self.global_others: Optional[List[np.ndarray]] = None

        # Frozen state (set once at R_E trigger, never updated afterward)
        self.selected_layer_indices: Optional[List[int]] = None
        self.selected_layer_scores: Optional[List[float]] = None
        self.frozen_cluster_groups: Optional[Dict[int, List[str]]] = None
        self.frozen_k: Optional[int] = None

        # Clustering log
        self._cluster_log_path: Optional[str] = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self._cluster_log_path = os.path.join(log_dir, "clustering.jsonl")

        logger.info(
            f"FedALCAggloLWCStrategy initialized "
            f"(E={warmup_rounds}, K=[{k_min}..{k_max}], "
            f"linkage={agglo_linkage}, layer_k={layer_selection_k}, "
            f"min_sil={agglo_min_silhouette})"
        )

    # ── New: K sweep with Agglomerative ─────────────────────────────────

    def _agglo_k_sweep(
        self, feature_matrix: np.ndarray
    ) -> Tuple[int, np.ndarray, float, List[Dict[str, Scalar]]]:
        """K sweep. Returns (best_k, best_labels, best_sil, sweep_log).

        sweep_log records (k, sil, cluster_sizes, n_singletons) for every K
        tried, so downstream analysis can diagnose singleton causes.
        No singleton guard — lets Agglomerative's natural behaviour through.
        """
        n = feature_matrix.shape[0]

        # Distance matrix with float-error correction
        dist = 1.0 - cosine_similarity(feature_matrix)
        np.fill_diagonal(dist, 0.0)
        np.clip(dist, 0.0, 2.0, out=dist)

        # K upper bound: silhouette requires K <= N-1
        k_upper = min(self.k_max, n - 1)

        sweep_log: List[Dict[str, Scalar]] = []
        best_k, best_labels, best_sil = 1, np.zeros(n, dtype=int), -1.0

        for k in range(self.k_min, k_upper + 1):
            model = AgglomerativeClustering(
                n_clusters=k,
                metric="precomputed",
                linkage=self.agglo_linkage,
            )
            labels = model.fit_predict(dist)

            if len(set(labels)) < 2:
                # Theoretically unreachable for k>=2 with Agglomerative, but guard anyway.
                continue

            sil = float(silhouette_score(dist, labels, metric="precomputed"))
            sizes = np.bincount(labels).tolist()
            n_singletons = sum(1 for s in sizes if s == 1)

            sweep_log.append({
                "k": int(k),
                "silhouette": round(sil, 6),
                "cluster_sizes": [int(s) for s in sizes],
                "n_singletons": int(n_singletons),
            })

            if sil > best_sil:
                best_k, best_labels, best_sil = k, labels, sil

        # Fallback: if best silhouette too low, force single cluster (no clustering benefit)
        if best_sil < self.agglo_min_silhouette:
            logger.warning(
                f"All K in sweep produced sil < {self.agglo_min_silhouette}; "
                f"falling back to K=1 (single cluster)."
            )
            return 1, np.zeros(n, dtype=int), best_sil, sweep_log

        return best_k, best_labels, best_sil, sweep_log

    # ── Main ────────────────────────────────────────────────────────────

    def _log_clustering(self, entry: Dict) -> None:
        if self._cluster_log_path:
            with open(self._cluster_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Three-phase aggregation: FedSA warm-up → R_E cluster+freeze → frozen reuse."""
        if not results:
            return None, {}

        # ── Collect per-client separated parameters ──
        client_a_list: List[List[np.ndarray]] = []
        client_b_list: List[List[np.ndarray]] = []
        client_other_list: List[List[np.ndarray]] = []
        client_weights: List[int] = []
        client_cids: List[str] = []
        cid_to_pid: Dict[str, str] = {}
        fit_metrics_list: List[Tuple[int, dict]] = []

        for client, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            a_params, b_params, other_params = separate_a_b_others(
                ndarrays, self.lora_param_keys
            )
            client_a_list.append(a_params)
            client_b_list.append(b_params)
            client_other_list.append(other_params)
            client_weights.append(fit_res.num_examples)
            client_cids.append(client.cid)
            cid_to_pid[client.cid] = str(
                fit_res.metrics.get("partition_id", client.cid)
            )
            fit_metrics_list.append((fit_res.num_examples, fit_res.metrics))

        # ── Always: global A + per-client others ──
        agg_a = weighted_average(client_a_list, client_weights)
        self.global_a_matrices = agg_a
        for cid, o in zip(client_cids, client_other_list):
            self.client_others[cid] = o
        # Fallback globals for clients seen for the first time in later rounds
        self.global_others = weighted_average(client_other_list, client_weights)
        self.global_b_matrices = weighted_average(client_b_list, client_weights)

        clustering_metrics: Dict[str, Scalar] = {}

        # ── Phase 0: FedSA warm-up (R1..R_{E-1}) ──
        if server_round < self.warmup_rounds:
            for cid, b in zip(client_cids, client_b_list):
                self.client_b_matrices[cid] = b   # per-client local B

            self._log_clustering({
                "round": server_round,
                "phase": 0,
                "event": "warmup",
                "n_clusters": 0,
                "clustering_features": {"method": "fedsa_warmup"},
                "clusters": {},
            })
            clustering_metrics = {"phase": 0, "n_clusters": 0}
            logger.info(f"Round {server_round}: Phase 0 (FedSA warm-up)")

        # ── Phase 1 trigger: one-shot cluster + freeze at R_E ──
        elif server_round == self.warmup_rounds:
            # Assumption check: Phase 1 is designed for fraction-fit=1.0.
            # Under partial sampling, frozen clusters will only cover
            # participating clients; absent ones are routed via Phase 2 fallback.
            expected_n_clients = getattr(self, "_expected_n_clients", None)
            if expected_n_clients is not None and len(client_cids) < expected_n_clients:
                logger.warning(
                    f"Round {server_round}: Phase 1 trigger with "
                    f"{len(client_cids)}/{expected_n_clients} clients. "
                    f"Absent clients will be routed via Phase 2 fallback."
                )

            # (1) Metric B layer scoring
            layer_scores = compute_layer_scores(client_b_list)
            top_k_idx = sorted(
                range(len(layer_scores)),
                key=lambda i: layer_scores[i],
                reverse=True,
            )[: self.layer_selection_k]
            self.selected_layer_indices = sorted(top_k_idx)
            self.selected_layer_scores = [
                layer_scores[i] for i in self.selected_layer_indices
            ]

            # (2) Build feature matrix from selected layers only
            feat_vecs = []
            for b_list in client_b_list:
                flat = np.concatenate(
                    [b_list[l].flatten() for l in self.selected_layer_indices]
                )
                feat_vecs.append(flat)
            feature_matrix = np.stack(feat_vecs)

            # (3) K sweep
            best_k, labels, best_sil, sweep_log = self._agglo_k_sweep(feature_matrix)
            self.frozen_k = best_k

            # (4) Build frozen cluster groups
            self.frozen_cluster_groups = {}
            for i, L in enumerate(labels):
                self.frozen_cluster_groups.setdefault(int(L), []).append(client_cids[i])

            # (5) Per-cluster aggregation on FULL B (all layers, not just top-K)
            client_b_agg: Dict[str, List[np.ndarray]] = {}
            for cluster_id, members in self.frozen_cluster_groups.items():
                idxs = [client_cids.index(c) for c in members]
                cluster_b = [client_b_list[i] for i in idxs]
                cluster_w = [client_weights[i] for i in idxs]
                agg = weighted_average(cluster_b, cluster_w)
                for cid in members:
                    client_b_agg[cid] = agg
            self.client_b_matrices = client_b_agg

            # Log trigger round
            b_keys = [k for k in self.lora_param_keys if "lora_B" in k]
            selected_b_keys = [b_keys[i] for i in self.selected_layer_indices]
            self._log_clustering({
                "round": server_round,
                "phase": 1,
                "event": "cluster_and_freeze",
                "n_clusters": best_k,
                "silhouette_score": round(best_sil, 6),
                "clustering_features": {
                    "method": "agglo_k_sweep",
                    "linkage": self.agglo_linkage,
                    "k_sweep_range": [self.k_min, self.k_max],
                    "selected_k": best_k,
                    "selected_layer_indices": self.selected_layer_indices,
                    "selected_b_keys": selected_b_keys,
                    "layer_scores_topk": [
                        round(s, 6) for s in self.selected_layer_scores
                    ],
                    "fallback_to_k1": best_k == 1,
                    "sweep_results": sweep_log,
                },
                "clusters": {
                    str(k): sorted(
                        [cid_to_pid.get(c, c) for c in v],
                        key=lambda x: int(x) if x.isdigit() else x,
                    )
                    for k, v in sorted(self.frozen_cluster_groups.items())
                },
            })
            clustering_metrics = {
                "phase": 1,
                "n_clusters": best_k,
                "silhouette_score": best_sil,
            }
            logger.info(
                f"Round {server_round}: Phase 1 TRIGGER → K*={best_k}, "
                f"sil={best_sil:.4f}, cluster_sizes={[len(v) for v in self.frozen_cluster_groups.values()]}"
            )

        # ── Phase 2: frozen reuse (R_{E+1}..) ──
        else:
            # Apply frozen_cluster_groups; route unknown cids to largest cluster
            cluster_groups: Dict[int, List[str]] = {}
            known_cids: set = set()
            for cl, members in self.frozen_cluster_groups.items():
                cluster_groups[cl] = [c for c in members if c in client_cids]
                known_cids.update(cluster_groups[cl])

            for cid in client_cids:
                if cid not in known_cids:
                    if cluster_groups:
                        largest = max(
                            cluster_groups.keys(),
                            key=lambda k: len(cluster_groups[k]),
                        )
                        cluster_groups[largest].append(cid)
                        logger.warning(
                            f"Round {server_round}: unknown cid {cid} routed to largest cluster {largest}"
                        )
                    else:
                        cluster_groups[0] = [cid]

            # Per-cluster aggregate on full B
            client_b_agg: Dict[str, List[np.ndarray]] = {}
            for cl, members in cluster_groups.items():
                idxs = [client_cids.index(cid) for cid in members]
                cluster_b = [client_b_list[i] for i in idxs]
                cluster_w = [client_weights[i] for i in idxs]
                agg = weighted_average(cluster_b, cluster_w)
                for cid in members:
                    client_b_agg[cid] = agg
            self.client_b_matrices = client_b_agg

            cluster_sizes = [len(m) for m in cluster_groups.values()]
            self._log_clustering({
                "round": server_round,
                "phase": 2,
                "event": "frozen",
                "n_clusters": len(cluster_groups),
                "silhouette_score": -1.0,
                "clustering_features": {"method": "frozen"},
                "clusters": {
                    str(k): sorted(
                        [cid_to_pid.get(c, c) for c in v],
                        key=lambda x: int(x) if x.isdigit() else x,
                    )
                    for k, v in sorted(cluster_groups.items())
                },
            })
            clustering_metrics = {
                "phase": 2,
                "n_clusters": len(cluster_groups),
                "cluster_sizes": str(cluster_sizes),
            }

        # ── Metrics aggregation + wandb ──
        metrics_aggregated: Dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics_list)
        for k, v in clustering_metrics.items():
            metrics_aggregated[f"clustering/{k}"] = v

        if self.use_wandb:
            import wandb
            wandb_metrics = {
                f"clustering/{k}": v
                for k, v in clustering_metrics.items()
                if isinstance(v, (int, float))
            }
            wandb_metrics["round"] = server_round
            wandb.log(wandb_metrics, step=server_round)

        # Server-side eval is disabled for this strategy. Return empty
        # Parameters placeholder — configure_fit / configure_evaluate use
        # self.* state instead of this argument, so the return value is never
        # consumed. Using explicit empty makes the "no global model" semantics
        # obvious vs reusing a trained client's params.
        return _EMPTY_PARAMETERS, metrics_aggregated

    # ── Personalized dispatch (copy from fedalc_ap_lwc_strategy.py:485-555) ──

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Send personalized (global A + cluster B + client others) per client."""
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)

        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        if self.global_a_matrices is None:
            return [(client, FitIns(parameters, dict(config))) for client in clients]

        fit_ins_list = []
        for client in clients:
            cid = client.cid
            client_b = self.client_b_matrices.get(cid, self.global_b_matrices)
            client_oth = self.client_others.get(cid, self.global_others)

            personalized = reconstruct_parameters(
                self.global_a_matrices, client_b, client_oth, self.lora_param_keys
            )
            personalized_params = ndarrays_to_parameters(personalized)
            fit_ins_list.append((client, FitIns(personalized_params, dict(config))))

        return fit_ins_list

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Send personalized parameters for client-side evaluation."""
        config = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)

        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        if self.global_a_matrices is None:
            return [(client, EvaluateIns(parameters, dict(config))) for client in clients]

        eval_ins_list = []
        for client in clients:
            cid = client.cid
            client_b = self.client_b_matrices.get(cid, self.global_b_matrices)
            client_oth = self.client_others.get(cid, self.global_others)

            personalized = reconstruct_parameters(
                self.global_a_matrices, client_b, client_oth, self.lora_param_keys
            )
            personalized_params = ndarrays_to_parameters(personalized)
            eval_ins_list.append(
                (client, EvaluateIns(personalized_params, dict(config)))
            )

        return eval_ins_list
