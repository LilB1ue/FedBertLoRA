"""FedAvg Strategy: Standard federated averaging of all LoRA parameters.

Aggregates both A and B matrices using weighted average (FedAvg).
For selective aggregation strategies, see fedsa_strategy.py.
"""

from flwr.server.strategy import FedAvg

# Re-export FedAvg directly — no custom logic needed for standard aggregation.
FedAvgStrategy = FedAvg
