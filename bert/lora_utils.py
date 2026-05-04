"""Shared pure utilities for LoRA federated strategies.

All functions are deterministic and side-effect free. Extracted from the
five strategy files (fedsa, fedalc_ap, fedalc_ap_lwc, fedalc_ap_multi,
fedalc_agglo_lwc) to reduce review burden — no behavioural change.
"""

from typing import List, Sequence, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from flwr.common.typing import NDArrays


def separate_a_b_others(
    parameters: NDArrays,
    lora_param_keys: Sequence[str],
    b_extra_keys: Tuple[str, ...] = (),
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Split flat ndarray list into (lora_A, lora_B + extras, others) by key name.

    `b_extra_keys` defaults to empty → FedALC behaviour (only `lora_B` keys go
    into the b list). FedSA passes `("classifier", "score")` to bundle the
    classifier head into the b list (B and head both stay client-local).
    """
    a_params: List[np.ndarray] = []
    b_params: List[np.ndarray] = []
    other_params: List[np.ndarray] = []
    for key, param in zip(lora_param_keys, parameters):
        if "lora_A" in key:
            a_params.append(param)
        elif "lora_B" in key or any(k in key for k in b_extra_keys):
            b_params.append(param)
        else:
            other_params.append(param)
    return a_params, b_params, other_params


def reconstruct_parameters(
    a_params: List[np.ndarray],
    b_params: List[np.ndarray],
    other_params: List[np.ndarray],
    lora_param_keys: Sequence[str],
    b_extra_keys: Tuple[str, ...] = (),
) -> NDArrays:
    """Reverse of `separate_a_b_others` — interleave back per key order."""
    result: NDArrays = []
    a_idx, b_idx, o_idx = 0, 0, 0
    for key in lora_param_keys:
        if "lora_A" in key:
            result.append(a_params[a_idx])
            a_idx += 1
        elif "lora_B" in key or any(k in key for k in b_extra_keys):
            result.append(b_params[b_idx])
            b_idx += 1
        else:
            result.append(other_params[o_idx])
            o_idx += 1
    return result


def weighted_average(
    matrix_lists: List[List[np.ndarray]],
    weights: List[int],
) -> List[np.ndarray]:
    """Weighted mean of a list-of-lists of ndarrays. Empty/zero weights safe."""
    total = float(sum(weights))
    if total == 0:
        return [mat.copy() for mat in matrix_lists[0]] if matrix_lists else []
    factors = [w / total for w in weights]
    aggregated = [mat.copy() * factors[0] for mat in matrix_lists[0]]
    for i, mats in enumerate(matrix_lists[1:], start=1):
        for j, mat in enumerate(mats):
            aggregated[j] += mat * factors[i]
    return aggregated


def compute_layer_scores(client_b_list: List[List[np.ndarray]]) -> List[float]:
    """Metric B per layer: (1 - mean pairwise cosine sim) × mean Frobenius norm.

    Higher score = layer where clients disagree strongly *and* have nontrivial
    magnitude. Used by FedALC-AP-LWC, FedALC-AP-Multi, FedALC-Agglo-LWC for
    top-K layer selection.

    For n=1 client (no pairs), returns 0.0 dissimilarity instead of NaN.
    """
    n_layers = len(client_b_list[0])
    scores: List[float] = []
    for l in range(n_layers):
        vecs = np.stack([c[l].flatten() for c in client_b_list])
        sim = cosine_similarity(vecs)
        mask = np.triu(np.ones(sim.shape, dtype=bool), k=1)
        if mask.sum() == 0:
            dissim = 0.0
        else:
            dissim = float(1.0 - sim[mask].mean())
        avg_norm = float(np.mean([np.linalg.norm(c[l]) for c in client_b_list]))
        scores.append(dissim * avg_norm)
    return scores
