"""FedALC-Random Strategy: control baseline that swaps AP for random cluster assignment.

Behaviour matches FedALC-AP exactly except for the clustering step:
  - A matrices       : global FedAvg (same as FedALC-AP)
  - B matrices       : random partition into K clusters → per-cluster FedAvg
  - Others (clf, …)  : stays per-client local (same as FedALC-AP)

Purpose
-------
Isolates the contribution of *AP-based* B-similarity clustering from the
"per-cluster B aggregation" mechanism itself. If FedALC-AP and FedALC-Random
score similarly, the clustering signal is not contributing — the win comes from
the cluster-sized aggregation buckets (or just from keeping B local-ish).

Modes supported
---------------
1. **Fixed-K random partition** (implemented):
   Round 1: shuffle clients → split evenly into K groups.
   By default the assignment is *frozen* across rounds (`fixed_assignment=True`)
   so the baseline measures a stable random groupage rather than per-round
   noise. Set `fixed_assignment=False` to re-draw every round.

2. **Match AP cluster-size distribution** (NOT implemented):
   Would require either an online AP run for size targets or a config-fed
   histogram. Skipped intentionally — fixed-K is sufficient for the control,
   and matching distributions adds another knob without changing the core
   ablation question. Revisit if reviewers ask for tighter matching.

Seed
----
Controlled by `random_seed` (defaults to fall back on a passed-in seed).
Re-using the same seed across runs reproduces the exact same partition.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from flwr.common import Scalar
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from bert.fedalc_ap_strategy import FedALCAPStrategy
from bert.lora_utils import weighted_average

logger = logging.getLogger(__name__)


class FedALCRandomStrategy(FedALCAPStrategy):
    """FedALC-AP with random cluster assignment instead of AP.

    Args:
        random_cluster_k: Number of clusters K (fixed). Required.
        fixed_assignment: If True (default), round-1 assignment is reused for
            all subsequent rounds. If False, re-draw every round.
        random_seed: Seed for the np.random.Generator used for assignment.
        **kwargs: Forwarded to FedALCAPStrategy (lora_param_keys, log_dir,
            use_wandb, FedAvg kwargs, ...). The AP-specific kwargs
            (ap_damping, ap_max_iter) are accepted but ignored.
    """

    def __init__(
        self,
        random_cluster_k: int = 3,
        fixed_assignment: bool = True,
        random_seed: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if random_cluster_k < 1:
            raise ValueError(f"random_cluster_k must be >= 1, got {random_cluster_k}")
        self.random_cluster_k = random_cluster_k
        self.fixed_assignment = fixed_assignment
        self.random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)

        # Frozen pid → cluster_id map (populated round 1 if fixed_assignment).
        # Keyed on partition_id (stable across rounds) rather than client cid
        # which Flower may re-issue between rounds.
        self._frozen_pid_to_cluster: Optional[Dict[str, int]] = None

        logger.info(
            f"FedALCRandomStrategy initialized "
            f"(K={random_cluster_k}, fixed_assignment={fixed_assignment}, "
            f"seed={random_seed})"
        )

    def _assign_random_clusters(
        self, n_clients: int, k: int
    ) -> np.ndarray:
        """Return labels of length n_clients with K balanced groups."""
        k = min(k, n_clients)  # K cannot exceed number of clients
        # Even split with leftover going to first clusters
        base, rem = divmod(n_clients, k)
        sizes = [base + (1 if i < rem else 0) for i in range(k)]
        labels = np.concatenate([np.full(s, i, dtype=int) for i, s in enumerate(sizes)])
        self._rng.shuffle(labels)
        return labels

    def _cluster_b_matrices(
        self,
        client_b_list: List[List[np.ndarray]],
        client_weights: List[int],
        client_cids: List[str],
        server_round: int,
        cid_to_pid: Optional[Dict[str, str]] = None,
    ) -> Tuple[Dict[str, List[np.ndarray]], Dict[str, Scalar]]:
        """Assign clients to K clusters at random, then per-cluster FedAvg of B.

        Mirrors FedALCAPStrategy._cluster_b_matrices output contract so the
        rest of the aggregate_fit pipeline works unchanged.
        """
        n_clients = len(client_b_list)
        metrics: Dict[str, Scalar] = {}
        cid_to_pid = cid_to_pid or {}

        # Resolve current-round cid → pid map (pid is stable across rounds; cid
        # is what the FedAvg pipeline keys on).
        cid_to_pid_round = {cid: cid_to_pid.get(cid, cid) for cid in client_cids}

        # 1) Determine per-client labels
        if self.fixed_assignment and self._frozen_pid_to_cluster is not None:
            # Reuse frozen assignment, indexed by partition_id
            labels = np.array(
                [
                    self._frozen_pid_to_cluster.get(cid_to_pid_round[cid], 0)
                    for cid in client_cids
                ],
                dtype=int,
            )
        else:
            labels = self._assign_random_clusters(n_clients, self.random_cluster_k)
            if self.fixed_assignment:
                # First round under fixed mode: persist mapping by pid
                self._frozen_pid_to_cluster = {
                    cid_to_pid_round[cid]: int(lbl)
                    for cid, lbl in zip(client_cids, labels)
                }

        n_clusters = len(set(labels.tolist()))

        # 2) Compute silhouette on cosine sim of B vectors (sanity / comparability with AP log)
        client_vectors = [np.concatenate([b.flatten() for b in bl]) for bl in client_b_list]
        feature_matrix = np.stack(client_vectors)
        sil_score = -1.0
        if n_clusters > 1 and n_clusters < n_clients:
            try:
                sil_score = float(
                    silhouette_score(feature_matrix, labels, metric="cosine")
                )
            except Exception as e:
                logger.warning(f"Silhouette failed: {e}")

        # 3) Build cluster membership
        cluster_members: Dict[int, List[str]] = {}
        for cid, lbl in zip(client_cids, labels.tolist()):
            cluster_members.setdefault(int(lbl), []).append(cid)
        cluster_sizes = [len(v) for _, v in sorted(cluster_members.items())]

        logger.info(
            f"Round {server_round}: random clustering → K={n_clusters}, "
            f"sizes={cluster_sizes}, silhouette={sil_score:.4f} "
            f"(fixed={self.fixed_assignment}, seed={self.random_seed})"
        )

        # 4) Identify B keys (for log compatibility with AP)
        b_keys_used = [k for k in self.lora_param_keys if "lora_B" in k]

        # 5) Write clustering log (JSONL — same shape as AP, plus mode/seed)
        if self._cluster_log_path:
            log_entry = {
                "round": server_round,
                "mode": "random",
                "n_clusters": n_clusters,
                "silhouette_score": round(sil_score, 6),
                "clustering_features": {
                    "method": "random",
                    "b_keys": b_keys_used,
                    "n_params": int(feature_matrix.shape[1]),
                    "fixed_assignment": self.fixed_assignment,
                    "seed": self.random_seed,
                    "k_requested": self.random_cluster_k,
                },
                "clusters": {
                    str(k): sorted(
                        [cid_to_pid.get(c, c) for c in v],
                        key=lambda x: int(x) if str(x).isdigit() else x,
                    )
                    for k, v in sorted(cluster_members.items())
                },
            }
            with open(self._cluster_log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        metrics["n_clusters"] = n_clusters
        metrics["silhouette_score"] = sil_score
        metrics["cluster_sizes"] = str(cluster_sizes)

        # 6) Per-cluster weighted average of B
        client_b_aggregated: Dict[str, List[np.ndarray]] = {}
        for cluster_label, member_cids in cluster_members.items():
            member_indices = [client_cids.index(cid) for cid in member_cids]
            cluster_b = [client_b_list[i] for i in member_indices]
            cluster_w = [client_weights[i] for i in member_indices]
            agg_b = weighted_average(cluster_b, cluster_w)
            for cid in member_cids:
                client_b_aggregated[cid] = agg_b

        return client_b_aggregated, metrics
