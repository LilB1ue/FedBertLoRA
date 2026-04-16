"""FedALC-LoRA Strategy: Adaptive Layer-selective Clustering for LoRA.

Phase 1: AP clustering on full B matrices.
  - A matrices: global FedAvg
  - B matrices: AP clustering → per-cluster FedAvg
  - Others (classifier, etc.): stays local per-client (not aggregated)
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
from flwr.common.typing import NDArrays
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FedALCAPStrategy(FedAvg):
    """Strategy with AP clustering on B matrices, global A, local others.

    Args:
        lora_param_keys: Ordered list of LoRA parameter names from the model.
        ap_damping: Damping factor for Affinity Propagation (0.5-1.0).
        ap_max_iter: Maximum iterations for AP convergence.
        **kwargs: Additional arguments passed to FedAvg.
    """

    def __init__(
        self,
        lora_param_keys: Optional[List[str]] = None,
        ap_damping: float = 0.5,
        ap_max_iter: int = 100,
        use_wandb: bool = False,
        log_dir: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lora_param_keys = lora_param_keys or []
        self.ap_damping = ap_damping
        self.ap_max_iter = ap_max_iter
        self.use_wandb = use_wandb
        self.log_dir = log_dir

        # Per-client state
        self.client_b_matrices: Dict[str, List[np.ndarray]] = {}
        self.client_others: Dict[str, List[np.ndarray]] = {}

        # Global state
        self.global_a_matrices: Optional[List[np.ndarray]] = None
        self.global_b_matrices: Optional[List[np.ndarray]] = None
        self.global_others: Optional[List[np.ndarray]] = None

        # Init clustering log file
        self._cluster_log_path: Optional[str] = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self._cluster_log_path = os.path.join(log_dir, "clustering.jsonl")

        logger.info(
            f"FedALCAPStrategy initialized (AP damping={ap_damping}, max_iter={ap_max_iter})"
        )

    def _separate_a_b_others(
        self, parameters: NDArrays
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Separate parameters into A, B, and others (classifier, etc.).

        All non-lora_A and non-lora_B parameters are grouped as "others"
        and stay local per-client (not aggregated).

        Returns:
            (a_params, b_params, other_params)
        """
        a_params = []
        b_params = []
        other_params = []
        for key, param in zip(self.lora_param_keys, parameters):
            if "lora_A" in key:
                a_params.append(param)
            elif "lora_B" in key:
                b_params.append(param)
            else:
                other_params.append(param)
        return a_params, b_params, other_params

    def _reconstruct_parameters(
        self,
        a_params: List[np.ndarray],
        b_params: List[np.ndarray],
        other_params: List[np.ndarray],
    ) -> NDArrays:
        """Reconstruct flat parameter array from separated A, B, others."""
        result = []
        a_idx, b_idx, o_idx = 0, 0, 0
        for key in self.lora_param_keys:
            if "lora_A" in key:
                result.append(a_params[a_idx])
                a_idx += 1
            elif "lora_B" in key:
                result.append(b_params[b_idx])
                b_idx += 1
            else:
                result.append(other_params[o_idx])
                o_idx += 1
        return result

    @staticmethod
    def _weighted_average(
        matrix_lists: List[List[np.ndarray]], weights: List[int]
    ) -> List[np.ndarray]:
        """Compute weighted average of matrix lists."""
        total = float(sum(weights))
        if total == 0:
            return [mat.copy() for mat in matrix_lists[0]] if matrix_lists else []

        factors = [w / total for w in weights]
        aggregated = [mat.copy() * factors[0] for mat in matrix_lists[0]]
        for i, mats in enumerate(matrix_lists[1:], start=1):
            for j, mat in enumerate(mats):
                aggregated[j] += mat * factors[i]
        return aggregated

    def _cluster_b_matrices(
        self,
        client_b_list: List[List[np.ndarray]],
        client_weights: List[int],
        client_cids: List[str],
        server_round: int,
        cid_to_pid: Optional[Dict[str, str]] = None,
    ) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, Scalar]]:
        """Run AP clustering on B matrices and aggregate per-cluster.

        Returns:
            (client_b_aggregated, clustering_metrics)
        """
        n_clients = len(client_b_list)
        metrics: Dict[str, Scalar] = {}

        # Flatten each client's B into a single vector
        client_vectors = []
        for b_list in client_b_list:
            flat = np.concatenate([b.flatten() for b in b_list])
            client_vectors.append(flat)
        feature_matrix = np.stack(client_vectors)  # (N_clients, D)

        # Cosine similarity matrix
        sim_matrix = cosine_similarity(feature_matrix)

        # AP clustering
        ap = AffinityPropagation(
            affinity="precomputed",
            damping=self.ap_damping,
            max_iter=self.ap_max_iter,
            random_state=42,
        )
        labels = ap.fit_predict(sim_matrix)

        # Fallback: if AP didn't converge or produced single cluster, treat as all-one-cluster
        if ap.n_iter_ == self.ap_max_iter:
            logger.warning(f"Round {server_round}: AP did not converge in {self.ap_max_iter} iters")
        n_clusters = len(set(labels))

        # Silhouette score
        sil_score = -1.0
        if n_clusters > 1 and n_clusters < n_clients:
            sil_score = float(
                silhouette_score(feature_matrix, labels, metric="cosine")
            )

        # Build cluster membership: {cluster_id: [cid, ...]}
        cluster_members: Dict[int, List[str]] = {}
        for i, label in enumerate(labels):
            cluster_members.setdefault(int(label), []).append(client_cids[i])

        cluster_sizes = [len(members) for _, members in sorted(cluster_members.items())]
        logger.info(
            f"Round {server_round}: AP clustering → {n_clusters} clusters, "
            f"sizes={cluster_sizes}, silhouette={sil_score:.4f}"
        )

        # Identify which B keys were used for clustering
        b_keys_used = [k for k in self.lora_param_keys if "lora_B" in k]

        # Write clustering log (JSONL: one JSON object per round)
        if self._cluster_log_path:
            log_entry = {
                "round": server_round,
                "n_clusters": n_clusters,
                "silhouette_score": round(sil_score, 6),
                "clustering_features": {
                    "method": "full_b",  # Phase 3: "layer_selected"
                    "b_keys": b_keys_used,
                    "n_params": int(feature_matrix.shape[1]),
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

        # Per-cluster weighted average of B (use cluster_members, not range(n_clusters))
        client_b_aggregated: Dict[str, List[np.ndarray]] = {}
        for cluster_label, member_cids in cluster_members.items():
            member_indices = [client_cids.index(cid) for cid in member_cids]
            cluster_b = [client_b_list[i] for i in member_indices]
            cluster_w = [client_weights[i] for i in member_indices]
            agg_b = self._weighted_average(cluster_b, cluster_w)

            for cid in member_cids:
                client_b_aggregated[cid] = agg_b

        return client_b_aggregated, metrics

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate: global A, clustered B, local others."""
        if not results:
            return None, {}

        # Collect per-client separated parameters
        client_a_list: List[List[np.ndarray]] = []
        client_b_list: List[List[np.ndarray]] = []
        client_other_list: List[List[np.ndarray]] = []
        client_weights: List[int] = []
        client_cids: List[str] = []
        cid_to_pid: Dict[str, str] = {}  # for readable logging
        fit_metrics_list: List[Tuple[int, dict]] = []

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

        # 1. A matrices: global FedAvg
        agg_a = self._weighted_average(client_a_list, client_weights)
        self.global_a_matrices = agg_a

        # 2. B matrices: AP clustering → per-cluster FedAvg
        clustered_b, clustering_metrics = self._cluster_b_matrices(
            client_b_list, client_weights, client_cids, server_round, cid_to_pid=cid_to_pid
        )
        self.client_b_matrices = clustered_b

        # Global B average (for server-side eval fallback)
        self.global_b_matrices = self._weighted_average(client_b_list, client_weights)

        # 3. Others (classifier, etc.): stays local per-client
        for cid, oth in zip(client_cids, client_other_list):
            self.client_others[cid] = oth

        # Global others average (for server-side eval fallback)
        self.global_others = self._weighted_average(client_other_list, client_weights)

        # Reconstruct global parameters for server-side evaluate_fn
        combined = self._reconstruct_parameters(
            agg_a, self.global_b_matrices, self.global_others
        )

        logger.info(
            f"Round {server_round}: FedALC — A global avg, "
            f"B clustered ({clustering_metrics.get('n_clusters', '?')} clusters), "
            f"others local ({len(self.client_others)} clients)"
        )

        # Aggregate fit metrics
        metrics_aggregated: Dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics_list)

        # Add clustering metrics
        for k, v in clustering_metrics.items():
            metrics_aggregated[f"clustering/{k}"] = v

        # Log clustering metrics to wandb (fit_metrics_aggregation_fn already logged client metrics)
        if self.use_wandb:
            import wandb
            wandb_clustering = {f"clustering/{k}": v for k, v in clustering_metrics.items()
                                if isinstance(v, (int, float))}
            wandb_clustering["round"] = server_round
            wandb.log(wandb_clustering, step=server_round)

        return ndarrays_to_parameters(combined), metrics_aggregated

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Send personalized parameters: global A + cluster B + own others."""
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
            # First round: send global parameters as-is
            return [(client, FitIns(parameters, dict(config))) for client in clients]

        fit_ins_list = []
        for client in clients:
            cid = client.cid

            # B: use client's cluster B, fallback to global avg
            client_b = self.client_b_matrices.get(cid, self.global_b_matrices)

            # Classifier: use client's own, fallback to global avg
            client_oth = self.client_others.get(cid, self.global_others)

            personalized = self._reconstruct_parameters(
                self.global_a_matrices, client_b, client_oth
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

            personalized = self._reconstruct_parameters(
                self.global_a_matrices, client_b, client_oth
            )
            personalized_params = ndarrays_to_parameters(personalized)
            eval_ins_list.append(
                (client, EvaluateIns(personalized_params, dict(config)))
            )

        return eval_ins_list
