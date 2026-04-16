"""Model loading, LoRA configuration, and A/B matrix separation utilities."""

import math
import os
import random
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch
from flwr.common.typing import NDArrays
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModelForSequenceClassification


def set_seed(seed: int = 42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 3e-4,
    lrate_min: float = 1e-6,
) -> float:
    """Implement cosine annealing learning rate schedule."""
    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


def get_model(model_name: str, num_labels: int, lora_r: int, lora_alpha: int,
              lora_target_modules: List[str], lora_dropout: float = 0.1):
    """Load pre-trained model with LoRA adapters for sequence classification."""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        torch_dtype=torch.float32,
    )

    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        task_type="SEQ_CLS",
        target_modules=lora_target_modules,
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def freeze_lora_a(model) -> None:
    """Freeze all lora_A parameters (for FFA-LoRA mode)."""
    frozen_count = 0
    for name, param in model.named_parameters():
        if "lora_A" in name and param.requires_grad:
            param.requires_grad = False
            frozen_count += 1
    print(f"[FFA-LoRA] Froze {frozen_count} lora_A parameters")


def get_parameters(model) -> NDArrays:
    """Return all LoRA parameters as a flat list of numpy arrays."""
    state_dict = get_peft_model_state_dict(model)
    return [val.cpu().numpy() for _, val in state_dict.items()]


def set_parameters(model, parameters: NDArrays) -> None:
    """Set all LoRA parameters from a flat list of numpy arrays."""
    peft_state_dict_keys = get_peft_model_state_dict(model).keys()
    params_dict = zip(peft_state_dict_keys, parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    set_peft_model_state_dict(model, state_dict)


def get_parameter_keys(model) -> List[str]:
    """Return ordered list of LoRA parameter names."""
    return list(get_peft_model_state_dict(model).keys())


def separate_a_b_parameters(model) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Separate LoRA-A and LoRA-B parameters by name.

    Returns:
        (a_params, b_params): Lists of numpy arrays for A and B matrices.
    """
    a_params = []
    b_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad or "lora_" not in name:
            continue
        if "lora_A" in name:
            a_params.append(param.data.cpu().numpy())
        elif "lora_B" in name:
            b_params.append(param.data.cpu().numpy())
    return a_params, b_params


def separate_a_b_from_arrays(model, parameters: NDArrays) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Separate A and B matrices from a flat parameter array using model keys.

    Args:
        model: The PEFT model (used for key inspection).
        parameters: Flat list of numpy arrays (same order as get_parameters).

    Returns:
        (a_params, b_params): Separated lists.
    """
    keys = get_parameter_keys(model)
    a_params = []
    b_params = []
    for key, param in zip(keys, parameters):
        if "lora_A" in key:
            a_params.append(param)
        elif "lora_B" in key:
            b_params.append(param)
    return a_params, b_params


def construct_parameters_from_a_b(model, a_params: List[np.ndarray],
                                   b_params: List[np.ndarray]) -> NDArrays:
    """Reconstruct flat parameter array from separate A and B lists.

    Maintains the original ordering from get_peft_model_state_dict.
    """
    keys = get_parameter_keys(model)
    result = []
    a_idx = 0
    b_idx = 0
    for key in keys:
        if "lora_A" in key:
            result.append(a_params[a_idx])
            a_idx += 1
        elif "lora_B" in key:
            result.append(b_params[b_idx])
            b_idx += 1
        else:
            # Non-A/B parameters (e.g., classifier head if trainable)
            # Get from current model state
            state_dict = get_peft_model_state_dict(model)
            result.append(state_dict[key].cpu().numpy())
    return result
