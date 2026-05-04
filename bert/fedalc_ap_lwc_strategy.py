"""FedALC-LWC Strategy: Layer-Wise Clustering for LoRA.

Three-phase strategy:
  Phase 0 (warm-up): FedSA mode (A global, B local) until silhouette > threshold
  Phase 1 (layer-selected clustering): Metric B selects top-K layers for AP clustering
  Phase 2 (frozen): Cluster assignment frozen, no more AP
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

from bert.lora_utils import (
    compute_layer_scores,
    reconstruct_parameters,
    separate_a_b_others,
    weighted_average,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FedALCAPLWCStrategy(FedAvg):
    """Layer-wise clustering: warm-up → layer selection → AP clustering → freeze.

    Args:
        lora_param_keys: Ordered list of LoRA parameter names from the model.
        ap_damping: Damping factor for Affinity Propagation (0.5-1.0).
        ap_max_iter: Maximum iterations for AP convergence.
        warmup_sil_threshold: Silhouette threshold to exit warm-up (phase 0 → 1).
        freeze_sil_threshold: Silhouette threshold to freeze clustering (phase 1 → 2).
        layer_selection_k: Number of top-K layers to select for clustering.
        use_wandb: Whether to log clustering metrics to wandb.
        log_dir: Directory for clustering.jsonl log.
        **kwargs: Additional arguments passed to FedAvg.
    """

    def __init__(
        self,
        lora_param_keys: Optional[List[str]] = None,
        ap_damping: float = 0.5,
        ap_max_iter: int = 100,
        warmup_sil_threshold: float = 0.5,
        freeze_sil_threshold: float = 0.9,
        layer_selection_k: int = 10,
        layer_reselect_every: int = 0,
        use_wandb: bool = False,
        log_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lora_param_keys = lora_param_keys or []
        self.ap_damping = ap_damping
        self.ap_max_iter = ap_max_iter
        self.warmup_sil_threshold = warmup_sil_threshold
        self.freeze_sil_threshold = freeze_sil_threshold
        self.layer_selection_k = layer_selection_k
        self.layer_reselect_every = layer_reselect_every  # 0=one-shot, N=every N rounds
        self._rounds_since_reselect = 0
        self.use_wandb = use_wandb
        self.log_dir = log_dir

        # Phase: 1=layer_selected_clustering, 2=frozen (no warm-up)
        self.phase = 1

        # Per-client state
        self.client_b_matrices: Dict[str, List[np.ndarray]] = {}
        self.client_others: Dict[str, List[np.ndarray]] = {}

        # Global state
        self.global_a_matrices: Optional[List[np.ndarray]] = None
        self.global_b_matrices: Optional[List[np.ndarray]] = None
        self.global_others: Optional[List[np.ndarray]] = None

        # Layer selection state
        self.selected_layer_indices: Optional[List[int]] = None
        self.selected_layer_scores: Optional[List[float]] = None

        # Clustering freeze state
        self.frozen_cluster_groups: Optional[Dict[int, List[str]]] = None  # cluster_id → [cids]
        self.prev_cluster_groups: Optional[set] = None
        self.rounds_stable: int = 0

        # Freeze snapshot (echoed into Phase 2 logs for self-contained reading)
        self._frozen_layer_indices: Optional[List[int]] = None
        self._frozen_at_round: Optional[int] = None

        # Init clustering log file
        self._cluster_log_path: Optional[str] = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self._cluster_log_path = os.path.join(log_dir, "clustering.jsonl")

        logger.info(
            f"FedALCAPLWCStrategy initialized (warmup_sil={warmup_sil_threshold}, "
            f"freeze_sil={freeze_sil_threshold}, layer_k={layer_selection_k})"
        )


    # ── Trial clustering (for silhouette check without aggregation) ────

    def _trial_ap_silhouette(
        self, client_b_list: List[List[np.ndarray]]
    ) -> Tuple[float, int, np.ndarray]:
        """Run AP on full B (trial only), return (silhouette, n_clusters, labels)."""
        client_vectors = []
        for b_list in client_b_list:
            flat = np.concatenate([b.flatten() for b in b_list])
            client_vectors.append(flat)
        feature_matrix = np.stack(client_vectors)

        sim_matrix = cosine_similarity(feature_matrix)
        ap = AffinityPropagation(
            affinity="precomputed",
            damping=self.ap_damping,
            max_iter=self.ap_max_iter,
            random_state=42,
        )
        labels = ap.fit_predict(sim_matrix)
        n_clusters = len(set(labels))

        sil = -1.0
        if n_clusters > 1 and n_clusters < len(client_b_list):
            sil = float(silhouette_score(feature_matrix, labels, metric="cosine"))

        return sil, n_clusters, labels

    # ── Layer-selected AP clustering ───────────────────────────────────

    def _cluster_with_selected_layers(
        self,
        client_b_list: List[List[np.ndarray]],
        client_weights: List[int],
        client_cids: List[str],
        server_round: int,
        cid_to_pid: Optional[Dict[str, str]] = None,
    ) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, Scalar]]:
        """AP clustering using only selected layers' B, aggregate ALL B per cluster."""
        metrics: Dict[str, Scalar] = {}

        # Build feature matrix from selected layers only
        client_vectors = []
        for b_list in client_b_list:
            selected = [b_list[i].flatten() for i in self.selected_layer_indices]
            client_vectors.append(np.concatenate(selected))
        feature_matrix = np.stack(client_vectors)

        # AP clustering
        sim_matrix = cosine_similarity(feature_matrix)
        ap = AffinityPropagation(
            affinity="precomputed",
            damping=self.ap_damping,
            max_iter=self.ap_max_iter,
            random_state=42,
        )
        labels = ap.fit_predict(sim_matrix)

        if ap.n_iter_ == self.ap_max_iter:
            logger.warning(f"Round {server_round}: AP did not converge")
        n_clusters = len(set(labels))

        # Silhouette on selected-layer features
        sil_score = -1.0
        if n_clusters > 1 and n_clusters < len(client_b_list):
            sil_score = float(silhouette_score(feature_matrix, labels, metric="cosine"))

        # Build cluster membership
        cluster_members: Dict[int, List[str]] = {}
        for i, label in enumerate(labels):
            cluster_members.setdefault(int(label), []).append(client_cids[i])

        cluster_sizes = [len(m) for _, m in sorted(cluster_members.items())]
        logger.info(
            f"Round {server_round}: LWC phase={self.phase}, {n_clusters} clusters, "
            f"sil={sil_score:.4f}, sizes={cluster_sizes}"
        )

        # Log
        b_keys = [k for k in self.lora_param_keys if "lora_B" in k]
        selected_b_keys = [b_keys[i] for i in self.selected_layer_indices]
        if self._cluster_log_path:
            log_entry = {
                "round": server_round,
                "phase": self.phase,
                "n_clusters": n_clusters,
                "silhouette_score": round(sil_score, 6),
                "clustering_features": {
                    "method": "layer_selected",
                    "selected_layer_indices": self.selected_layer_indices,
                    "selected_b_keys": selected_b_keys,
                    "layer_scores_topk": [round(float(s), 4) for s in self.selected_layer_scores]
                            if self.selected_layer_scores else [],
                    "metric": "dissim_norm",
                    "n_params_clustering": int(feature_matrix.shape[1]),
                },
                "clusters": {
                    str(k): (
                        sorted([cid_to_pid.get(c, c) for c in v], key=lambda x: int(x) if x.isdigit() else x)
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

        # Per-cluster weighted average of ALL B (not just selected)
        client_b_aggregated: Dict[str, List[np.ndarray]] = {}
        for cluster_label, member_cids in cluster_members.items():
            member_indices = [client_cids.index(cid) for cid in member_cids]
            cluster_b = [client_b_list[i] for i in member_indices]
            cluster_w = [client_weights[i] for i in member_indices]
            agg_b = weighted_average(cluster_b, cluster_w)
            for cid in member_cids:
                client_b_aggregated[cid] = agg_b

        # Check phase transition: 1 → 2
        current_groups = set(frozenset(v) for v in cluster_members.values())
        if self.prev_cluster_groups is not None and current_groups == self.prev_cluster_groups:
            self.rounds_stable += 1
        else:
            self.rounds_stable = 0
        self.prev_cluster_groups = current_groups

        if sil_score >= self.freeze_sil_threshold or self.rounds_stable >= 3:
            self.phase = 2
            self.frozen_cluster_groups = {
                int(label): list(member_cids)
                for label, member_cids in cluster_members.items()
            }
            # Snapshot layer selection at freeze time so Phase 2 logs are self-contained
            self._frozen_layer_indices = (
                list(self.selected_layer_indices) if self.selected_layer_indices else None
            )
            self._frozen_at_round = int(server_round)
            logger.info(
                f"Round {server_round}: → Phase 2 (frozen), "
                f"sil={sil_score:.4f}, stable_rounds={self.rounds_stable}"
            )

        return client_b_aggregated, metrics

    # ── Main aggregation ───────────────────────────────────────────────

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Three-phase aggregation: warm-up → layer-selected clustering → frozen."""
        if not results:
            return None, {}

        # Collect per-client separated parameters
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
            cid_to_pid[client.cid] = str(fit_res.metrics.get("partition_id", client.cid))
            fit_metrics_list.append((fit_res.num_examples, fit_res.metrics))

        # A: always global FedAvg
        agg_a = weighted_average(client_a_list, client_weights)
        self.global_a_matrices = agg_a

        # Others: always per-client local
        for cid, oth in zip(client_cids, client_other_list):
            self.client_others[cid] = oth
        self.global_others = weighted_average(client_other_list, client_weights)

        # Global B average (for server-side eval fallback, all phases)
        self.global_b_matrices = weighted_average(client_b_list, client_weights)

        clustering_metrics: Dict[str, Scalar] = {}

        # ── Phase 1: Layer-selected clustering ──
        if self.phase == 1:
            # Compute layer scores: first round always, then based on reselect_every
            need_select = self.selected_layer_indices is None  # first round
            if not need_select and self.layer_reselect_every > 0:
                self._rounds_since_reselect += 1
                if self._rounds_since_reselect >= self.layer_reselect_every:
                    need_select = True

            if need_select:
                scores = compute_layer_scores(client_b_list)
                top_k_indices = sorted(
                    range(len(scores)), key=lambda i: scores[i], reverse=True
                )[:self.layer_selection_k]
                self.selected_layer_indices = sorted(top_k_indices)
                self.selected_layer_scores = [scores[i] for i in self.selected_layer_indices]
                self._rounds_since_reselect = 0

                b_keys = [k for k in self.lora_param_keys if "lora_B" in k]
                selected_names = [b_keys[i] for i in self.selected_layer_indices]
                logger.info(
                    f"Round {server_round}: Layer selection → {selected_names[:3]}... "
                    f"(top-{self.layer_selection_k})"
                )

            clustered_b, cl_metrics = self._cluster_with_selected_layers(
                client_b_list, client_weights, client_cids, server_round, cid_to_pid=cid_to_pid
            )
            self.client_b_matrices = clustered_b
            clustering_metrics.update(cl_metrics)
            clustering_metrics["phase"] = self.phase  # may have changed to 2

        # ── Phase 2: Frozen clustering ──
        elif self.phase == 2:
            # Use frozen_cluster_groups directly (no AP, no id() tricks)
            cluster_groups = {}
            known_cids = set()
            for cl, member_cids in self.frozen_cluster_groups.items():
                cluster_groups[cl] = [c for c in member_cids if c in client_cids]
                known_cids.update(cluster_groups[cl])

            # Assign unknown clients to largest cluster
            for cid in client_cids:
                if cid not in known_cids:
                    if cluster_groups:
                        largest = max(cluster_groups.keys(), key=lambda k: len(cluster_groups[k]))
                        cluster_groups[largest].append(cid)
                    else:
                        cluster_groups[0] = [cid]

            # Per-cluster avg with current round's B
            client_b_aggregated: Dict[str, List[np.ndarray]] = {}
            for cl, member_cids in cluster_groups.items():
                member_indices = [client_cids.index(cid) for cid in member_cids]
                cluster_b = [client_b_list[i] for i in member_indices]
                cluster_w = [client_weights[i] for i in member_indices]
                agg_b = weighted_average(cluster_b, cluster_w)
                for cid in member_cids:
                    client_b_aggregated[cid] = agg_b

            self.client_b_matrices = client_b_aggregated

            n_clusters = len(cluster_groups)
            cluster_sizes = [len(m) for m in cluster_groups.values()]
            clustering_metrics["n_clusters"] = n_clusters
            clustering_metrics["phase"] = 2
            clustering_metrics["cluster_sizes"] = str(cluster_sizes)

            # Log frozen round (echo layer snapshot so each entry is self-contained)
            if self._cluster_log_path:
                b_keys = [k for k in self.lora_param_keys if "lora_B" in k]
                frozen_b_keys = (
                    [b_keys[i] for i in self._frozen_layer_indices]
                    if self._frozen_layer_indices else []
                )
                log_entry = {
                    "round": server_round,
                    "phase": 2,
                    "n_clusters": n_clusters,
                    "silhouette_score": -1.0,
                    "clustering_features": {
                        "method": "frozen",
                        "frozen_layer_indices": self._frozen_layer_indices,
                        "frozen_b_keys": frozen_b_keys,
                        "frozen_at_round": self._frozen_at_round,
                    },
                    "clusters": {
                        str(k): sorted(
                            [cid_to_pid.get(c, c) for c in v],
                            key=lambda x: int(x) if x.isdigit() else x
                        )
                        for k, v in cluster_groups.items()
                    },
                }
                with open(self._cluster_log_path, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

            logger.info(
                f"Round {server_round}: Phase 2 (frozen), {n_clusters} clusters, "
                f"sizes={cluster_sizes}"
            )

        # Reconstruct global parameters for server-side evaluate_fn
        combined = reconstruct_parameters(
            agg_a, self.global_b_matrices, self.global_others, self.lora_param_keys
        )

        # Aggregate fit metrics
        metrics_aggregated: Dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics_list)

        for k, v in clustering_metrics.items():
            metrics_aggregated[f"clustering/{k}"] = v

        # Log to wandb
        if self.use_wandb:
            import wandb
            wandb_metrics = {f"clustering/{k}": v for k, v in clustering_metrics.items()
                            if isinstance(v, (int, float))}
            wandb_metrics["round"] = server_round
            wandb.log(wandb_metrics, step=server_round)

        return ndarrays_to_parameters(combined), metrics_aggregated

    # ── Parameter distribution ─────────────────────────────────────────

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Send personalized parameters based on current phase."""
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
        """Send personalized parameters for evaluation."""
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
