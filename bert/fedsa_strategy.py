"""FedSA-LoRA Strategy: Selective aggregation of LoRA A/B matrices.

Supports multiple aggregation modes:
  - "fedsa":  FedSA-LoRA — aggregate A only, B stays local (default)
  - "ffa":    FFA-LoRA — freeze A (not aggregated), aggregate B only
  - "cluster": (future) Clustering-based B aggregation + global A

Note: "fedavg" (standard FedAvg) is handled by FedAvgStrategy in strategy.py,
not by this class.
"""

import logging
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

from bert.lora_utils import (
    reconstruct_parameters,
    separate_a_b_others,
    weighted_average,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In FedSA-LoRA the classifier head (and `score` for some heads) stays
# client-local alongside lora_B, so it is bundled into the "b" list when
# splitting parameters.
B_EXTRA_KEYS: Tuple[str, ...] = ("classifier", "score")


class FedSALoRAStrategy(FedAvg):
    """Strategy that selectively aggregates LoRA A and B matrices.

    Args:
        aggregation_mode: One of "fedsa", "ffa", "full".
        lora_param_keys: Ordered list of LoRA parameter names from the model.
        **kwargs: Additional arguments passed to FedAvg.
    """

    def __init__(
        self,
        aggregation_mode: str = "fedsa",
        lora_param_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.aggregation_mode = aggregation_mode
        self.lora_param_keys = lora_param_keys or []

        # Track per-client B matrices (for fedsa mode: B stays local)
        # Key: client cid, Value: list of B numpy arrays
        self.client_b_matrices: Dict[str, List[np.ndarray]] = {}

        # Global aggregated A matrices
        self.global_a_matrices: Optional[List[np.ndarray]] = None

        # Global aggregated B matrices (for ffa/full modes)
        self.global_b_matrices: Optional[List[np.ndarray]] = None

        logger.info(f"FedSALoRAStrategy initialized with mode: {aggregation_mode}")

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate client results with selective A/B handling."""
        if not results:
            return None, {}

        # Collect per-client separated parameters
        client_a_list: List[List[np.ndarray]] = []
        client_b_list: List[List[np.ndarray]] = []
        client_other_list: List[List[np.ndarray]] = []
        client_weights: List[int] = []
        client_cids: List[str] = []

        fit_metrics_list: List[Tuple[int, dict]] = []

        for client, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            a_params, b_params, other_params = separate_a_b_others(
                ndarrays, self.lora_param_keys, b_extra_keys=B_EXTRA_KEYS
            )

            client_a_list.append(a_params)
            client_b_list.append(b_params)
            client_other_list.append(other_params)
            client_weights.append(fit_res.num_examples)
            client_cids.append(client.cid)
            fit_metrics_list.append((fit_res.num_examples, fit_res.metrics))

        # Aggregate "other" params via FedAvg (if any remain after A/B/classifier separation)
        if client_other_list and client_other_list[0]:
            agg_other = weighted_average(client_other_list, client_weights)
        else:
            agg_other = []

        if self.aggregation_mode == "fedsa":
            # FedSA-LoRA: aggregate A globally, B stays per-client
            agg_a = weighted_average(client_a_list, client_weights)
            self.global_a_matrices = agg_a

            # Store each client's B matrices for next round
            for cid, b_params in zip(client_cids, client_b_list):
                self.client_b_matrices[cid] = b_params

            # For the returned "global" parameters, use aggregated A + averaged B as fallback
            # (This is used for evaluate_fn; actual client params are personalized in configure_fit)
            agg_b = weighted_average(client_b_list, client_weights)
            self.global_b_matrices = agg_b
            combined = reconstruct_parameters(
                agg_a, agg_b, agg_other, self.lora_param_keys, b_extra_keys=B_EXTRA_KEYS
            )

            logger.info(f"Round {server_round}: FedSA-LoRA — aggregated A ({len(agg_a)} matrices), "
                        f"tracking B for {len(self.client_b_matrices)} clients")

        elif self.aggregation_mode == "ffa":
            # FFA-LoRA: aggregate B only, A stays as-is (frozen from init)
            agg_b = weighted_average(client_b_list, client_weights)
            self.global_b_matrices = agg_b

            # A matrices: keep the first client's A (should all be same if frozen)
            if self.global_a_matrices is None:
                self.global_a_matrices = client_a_list[0]
            agg_a = self.global_a_matrices

            combined = reconstruct_parameters(
                agg_a, agg_b, agg_other, self.lora_param_keys, b_extra_keys=B_EXTRA_KEYS
            )
            logger.info(f"Round {server_round}: FFA-LoRA — aggregated B, A frozen")

        else:
            raise ValueError(f"Unknown aggregation_mode: {self.aggregation_mode}")

        # Pass through client fit metrics for aggregation (fit_metrics_aggregation_fn)
        metrics_aggregated: Dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics_list)

        return ndarrays_to_parameters(combined), metrics_aggregated

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure next round: send personalized parameters per client."""
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)

        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        if self.aggregation_mode == "fedsa" and self.global_a_matrices is not None:
            # Personalized: global A + each client's own B
            fit_ins_list = []
            for client in clients:
                cid = client.cid
                if cid in self.client_b_matrices:
                    client_b = self.client_b_matrices[cid]
                else:
                    # New client or first round: use global B as fallback
                    if self.global_b_matrices is not None:
                        client_b = self.global_b_matrices
                    else:
                        # Very first round: just send global parameters
                        fit_ins = FitIns(parameters, dict(config))
                        fit_ins_list.append((client, fit_ins))
                        continue

                # Reconstruct: need "other" params from current global
                global_arrays = parameters_to_ndarrays(parameters)
                _, _, other_params = separate_a_b_others(
                    global_arrays, self.lora_param_keys, b_extra_keys=B_EXTRA_KEYS
                )

                personalized = reconstruct_parameters(
                    self.global_a_matrices,
                    client_b,
                    other_params,
                    self.lora_param_keys,
                    b_extra_keys=B_EXTRA_KEYS,
                )
                personalized_params = ndarrays_to_parameters(personalized)
                fit_ins = FitIns(personalized_params, dict(config))
                fit_ins_list.append((client, fit_ins))

            return fit_ins_list
        else:
            # FFA/Full: same parameters for all clients
            return [(client, FitIns(parameters, dict(config))) for client in clients]

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Send personalized parameters to each client for evaluation."""
        config = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)

        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        if self.aggregation_mode == "fedsa" and self.global_a_matrices is not None:
            eval_ins_list = []
            for client in clients:
                cid = client.cid
                if cid in self.client_b_matrices:
                    client_b = self.client_b_matrices[cid]
                else:
                    client_b = self.global_b_matrices if self.global_b_matrices is not None else None

                if client_b is None:
                    eval_ins_list.append((client, EvaluateIns(parameters, dict(config))))
                    continue

                global_arrays = parameters_to_ndarrays(parameters)
                _, _, other_params = separate_a_b_others(
                    global_arrays, self.lora_param_keys, b_extra_keys=B_EXTRA_KEYS
                )

                personalized = reconstruct_parameters(
                    self.global_a_matrices,
                    client_b,
                    other_params,
                    self.lora_param_keys,
                    b_extra_keys=B_EXTRA_KEYS,
                )
                personalized_params = ndarrays_to_parameters(personalized)
                eval_ins_list.append((client, EvaluateIns(personalized_params, dict(config))))

            return eval_ins_list
        else:
            return [(client, EvaluateIns(parameters, dict(config))) for client in clients]
