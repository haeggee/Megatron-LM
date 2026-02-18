# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Metric computation functions for model internals logging."""

import math
from typing import Dict
import torch
from torch import Tensor


def compute_activation_stats(tensor: Tensor) -> Dict[str, float]:
    """Compute activation statistics: mean, std, min, max, kurtosis.

    Args:
        tensor: Activation tensor of any shape.

    Returns:
        Dictionary with activation statistics.
    """
    with torch.no_grad():
        flat = tensor.flatten().float()

        mean = flat.mean().item()
        std = flat.std().item()
        min_val = flat.min().item()
        max_val = flat.max().item()

        # Kurtosis: E[(X-mu)^4] / sigma^4 - 3 (excess kurtosis)
        # Normal distribution has kurtosis = 0
        if std > 1e-8:
            centered = flat - mean
            fourth_moment = centered.pow(4).mean()
            kurtosis = (fourth_moment / (std ** 4) - 3).item()
        else:
            kurtosis = 0.0

    return {
        'mean': mean,
        'std': std,
        'min': min_val,
        'max': max_val,
        'kurtosis': kurtosis,
    }


def get_param_grad(param: Tensor) -> Optional[Tensor]:
    """Get gradient tensor for a parameter.

    Megatron-LM uses main_grad for distributed training instead of standard grad.
    This helper checks both.

    Args:
        param: Parameter tensor.

    Returns:
        Gradient tensor or None if no gradient is available.
    """
    # Check main_grad first (used by Megatron's DistributedDataParallel)
    if hasattr(param, 'main_grad') and param.main_grad is not None:
        return param.main_grad
    # Fall back to standard grad
    if param.grad is not None:
        return param.grad
    return None


def compute_gradient_norm(param: Tensor) -> float:
    """Compute L2 norm of gradient for a parameter.

    Args:
        param: Parameter tensor with .grad or .main_grad attribute.

    Returns:
        L2 norm of the gradient, or 0.0 if no gradient.
    """
    grad = get_param_grad(param)
    if grad is None:
        return 0.0
    with torch.no_grad():
        return grad.float().norm().item()


def compute_weight_delta(
    current: Tensor,
    previous: Tensor,
) -> float:
    """Compute relative weight change: ||W_t - W_{t-1}|| / ||W_{t-1}||.

    Args:
        current: Current weight tensor.
        previous: Previous weight tensor (same shape as current).

    Returns:
        Relative L2 norm of the weight change, or 0.0 if previous norm is zero.
    """
    with torch.no_grad():
        # Move to same device if needed
        if previous.device != current.device:
            previous = previous.to(current.device)

        delta_norm = (current.float() - previous.float()).norm().item()
        prev_norm = previous.float().norm().item()

        if prev_norm > 1e-10:
            return delta_norm / prev_norm
        return 0.0


def compute_weight_delta_per_neuron(
    current: Tensor,
    previous: Tensor,
) -> Dict[str, float]:
    """Compute per-neuron (per-row) relative weight changes.

    For a weight matrix W of shape [out_features, in_features], computes
    the relative change for each output neuron (row) and returns statistics.

    Args:
        current: Current weight tensor (at least 2D).
        previous: Previous weight tensor (same shape as current).

    Returns:
        Dictionary with per-neuron statistics:
        - 'full': Relative change of full matrix (Frobenius norm)
        - 'per_neuron_mean': Mean of per-neuron relative changes
        - 'per_neuron_std': Std of per-neuron relative changes
        - 'per_neuron_max': Max per-neuron relative change
        - 'per_neuron_min': Min per-neuron relative change
    """
    with torch.no_grad():
        # Move to same device if needed
        if previous.device != current.device:
            previous = previous.to(current.device)

        current_f = current.float()
        previous_f = previous.float()

        # Full matrix relative change
        delta_norm = (current_f - previous_f).norm().item()
        prev_norm = previous_f.norm().item()
        full_delta = delta_norm / prev_norm if prev_norm > 1e-10 else 0.0

        result = {'full': full_delta}

        # Per-neuron stats only make sense for 2D+ tensors
        if current.dim() >= 2:
            # Reshape to [num_neurons, -1] for per-row computation
            num_neurons = current.size(0)
            current_2d = current_f.view(num_neurons, -1)
            previous_2d = previous_f.view(num_neurons, -1)

            # Per-row (per-neuron) norms
            delta_per_neuron = (current_2d - previous_2d).norm(dim=1)  # [num_neurons]
            prev_per_neuron = previous_2d.norm(dim=1)  # [num_neurons]

            # Relative change per neuron (avoid division by zero)
            mask = prev_per_neuron > 1e-10
            relative_delta = torch.zeros_like(delta_per_neuron)
            relative_delta[mask] = delta_per_neuron[mask] / prev_per_neuron[mask]

            result['per_neuron_mean'] = relative_delta.mean().item()
            result['per_neuron_std'] = relative_delta.std().item() if num_neurons > 1 else 0.0
            result['per_neuron_max'] = relative_delta.max().item()
            result['per_neuron_min'] = relative_delta.min().item()
        else:
            # For 1D tensors (biases), just use the full delta
            result['per_neuron_mean'] = full_delta
            result['per_neuron_std'] = 0.0
            result['per_neuron_max'] = full_delta
            result['per_neuron_min'] = full_delta

        return result


def compute_angular_update(
    current: Tensor,
    previous: Tensor,
) -> Dict[str, float]:
    """Compute angular change between weight tensors.

    Measures how much the *direction* of weights changed, independent of magnitude.
    Useful for analyzing spherical/normalized training dynamics (nGPT, EDM2).

    Args:
        current: Current weight tensor.
        previous: Previous weight tensor (same shape as current).

    Returns:
        Dictionary with:
        - cos_similarity: cosine similarity between W_t and W_{t-1} (-1 to 1)
        - angular_change: angle in radians (0 to pi)
        - angular_change_degrees: angle in degrees (0 to 180)
    """
    with torch.no_grad():
        # Move to same device if needed
        if previous.device != current.device:
            previous = previous.to(current.device)

        curr_flat = current.flatten().float()
        prev_flat = previous.flatten().float()

        curr_norm = curr_flat.norm()
        prev_norm = prev_flat.norm()

        if curr_norm < 1e-10 or prev_norm < 1e-10:
            return {
                'cos_similarity': 0.0,
                'angular_change': 0.0,
                'angular_change_degrees': 0.0,
            }

        cos_sim = (curr_flat @ prev_flat) / (curr_norm * prev_norm)
        cos_sim = cos_sim.clamp(-1.0, 1.0)  # numerical stability

        angular_change = torch.acos(cos_sim).item()

    return {
        'cos_similarity': cos_sim.item(),
        'angular_change': angular_change,
        'angular_change_degrees': angular_change * 180.0 / math.pi,
    }


def compute_gradient_weight_alignment(param: Tensor) -> Dict[str, float]:
    """Compute alignment between gradient and weight vectors.

    Measures whether gradients point along weights (radial) or orthogonal (tangential).
    For spherical training, tangential updates (cos ~ 0) preserve weight norms.

    Args:
        param: Parameter tensor with .grad or .main_grad attribute.

    Returns:
        Dictionary with:
        - cos_alignment: cosine of angle between gradient and weight (-1 to 1)
          - cos ~ 1: gradient increases weight norm (radial, norm-increasing)
          - cos ~ 0: gradient changes direction only (tangential)
          - cos ~ -1: gradient decreases weight norm (radial, norm-decreasing)
        - radial_component: ||grad|| * cos(grad, W) - component along weight direction
        - tangential_component: ||grad|| * sin(grad, W) - component orthogonal to weight
    """
    grad = get_param_grad(param)
    if grad is None:
        return {
            'cos_alignment': 0.0,
            'radial_component': 0.0,
            'tangential_component': 0.0,
        }

    with torch.no_grad():
        grad_flat = grad.flatten().float()
        weight_flat = param.data.flatten().float()

        grad_norm = grad_flat.norm()
        weight_norm = weight_flat.norm()

        if grad_norm < 1e-10 or weight_norm < 1e-10:
            return {
                'cos_alignment': 0.0,
                'radial_component': 0.0,
                'tangential_component': 0.0,
            }

        cos_align = (grad_flat @ weight_flat) / (grad_norm * weight_norm)
        cos_align = cos_align.clamp(-1.0, 1.0)

        # Decompose gradient into radial and tangential components
        radial = (grad_norm * cos_align).item()
        tangential = (grad_norm * torch.sqrt(1 - cos_align ** 2)).item()

    return {
        'cos_alignment': cos_align.item(),
        'radial_component': radial,
        'tangential_component': tangential,
    }
