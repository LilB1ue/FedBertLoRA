"""FedSA-LoRA Strategy: Selective aggregation of LoRA A/B matrices.

Supports multiple aggregation modes:
  - "fedsa":  FedSA-LoRA — aggregate A only, B stays local (default)
  - "ffa":    FFA-LoRA — freeze A (not aggregated), aggregate B only
  - "full":   Standard FedAvg — aggregate both A and B
  - "cluster": (future) Clustering-based B aggregation + global A
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

    def _separate_a_b(self, parameters: NDArrays) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Separate parameters into A matrices, B matrices, and other params.

        Uses lora_param_keys for name-based separation.
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
        """Reconstruct flat parameter array from separated A, B, and other params."""
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

        for client, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            a_params, b_params, other_params = self._separate_a_b(ndarrays)

            client_a_list.append(a_params)
            client_b_list.append(b_params)
            client_other_list.append(other_params)
            client_weights.append(fit_res.num_examples)
            client_cids.append(client.cid)

        # Always aggregate "other" params (e.g., classifier head) via FedAvg
        if client_other_list and client_other_list[0]:
            agg_other = self._weighted_average(client_other_list, client_weights)
        else:
            agg_other = []

        if self.aggregation_mode == "fedsa":
            # FedSA-LoRA: aggregate A globally, B stays per-client
            agg_a = self._weighted_average(client_a_list, client_weights)
            self.global_a_matrices = agg_a

            # Store each client's B matrices for next round
            for cid, b_params in zip(client_cids, client_b_list):
                self.client_b_matrices[cid] = b_params

            # For the returned "global" parameters, use aggregated A + averaged B as fallback
            # (This is used for evaluate_fn; actual client params are personalized in configure_fit)
            agg_b = self._weighted_average(client_b_list, client_weights)
            self.global_b_matrices = agg_b
            combined = self._reconstruct_parameters(agg_a, agg_b, agg_other)

            logger.info(f"Round {server_round}: FedSA-LoRA — aggregated A ({len(agg_a)} matrices), "
                        f"tracking B for {len(self.client_b_matrices)} clients")

        elif self.aggregation_mode == "ffa":
            # FFA-LoRA: aggregate B only, A stays as-is (frozen from init)
            agg_b = self._weighted_average(client_b_list, client_weights)
            self.global_b_matrices = agg_b

            # A matrices: keep the first client's A (should all be same if frozen)
            if self.global_a_matrices is None:
                self.global_a_matrices = client_a_list[0]
            agg_a = self.global_a_matrices

            combined = self._reconstruct_parameters(agg_a, agg_b, agg_other)
            logger.info(f"Round {server_round}: FFA-LoRA — aggregated B, A frozen")

        elif self.aggregation_mode == "full":
            # Standard FedAvg: aggregate both A and B
            agg_a = self._weighted_average(client_a_list, client_weights)
            agg_b = self._weighted_average(client_b_list, client_weights)
            self.global_a_matrices = agg_a
            self.global_b_matrices = agg_b

            combined = self._reconstruct_parameters(agg_a, agg_b, agg_other)
            logger.info(f"Round {server_round}: Full FedAvg — aggregated A+B")

        else:
            raise ValueError(f"Unknown aggregation_mode: {self.aggregation_mode}")

        return ndarrays_to_parameters(combined), {}

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
                _, _, other_params = self._separate_a_b(global_arrays)

                personalized = self._reconstruct_parameters(
                    self.global_a_matrices, client_b, other_params
                )
                personalized_params = ndarrays_to_parameters(personalized)
                fit_ins = FitIns(personalized_params, dict(config))
                fit_ins_list.append((client, fit_ins))

            return fit_ins_list
        else:
            # FFA/Full: same parameters for all clients
            return [(client, FitIns(parameters, dict(config))) for client in clients]
