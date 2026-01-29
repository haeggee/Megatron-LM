# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""State management for tracking previous values across iterations."""

from typing import Dict, Optional
import torch
from torch import Tensor
import torch.nn as nn

from .metrics import (
    compute_weight_delta,
    compute_weight_delta_per_neuron,
    compute_activation_delta,
    compute_angular_update,
)


class InternalsStateManager:
    """Manages state for computing delta metrics across training iterations.

    This class caches previous weights and activations to compute relative
    changes (delta_W, delta_Y) between iterations. Weights are stored on CPU
    to conserve GPU memory.

    Attributes:
        previous_weights: Dictionary mapping parameter names to their previous values.
        previous_activations: Dictionary mapping layer numbers to previous activation samples.
        initialized: Whether the first iteration's state has been captured.
        weights_on_gpu: Whether to keep previous weights on GPU (faster but uses more memory).
    """

    def __init__(self, weights_on_gpu: bool = False):
        """Initialize the state manager with empty caches.

        Args:
            weights_on_gpu: If True, keep previous weights on GPU to avoid
                GPU-CPU transfer overhead. Uses more GPU memory but eliminates
                performance variance from PCIe transfers.
        """
        self.previous_weights: Dict[str, Tensor] = {}
        self.previous_activations: Dict[int, Tensor] = {}
        self.initialized = False
        self.weights_on_gpu = weights_on_gpu

    def update_weights(self, model: nn.Module, subset_params: Optional[int] = None) -> None:
        """Cache current weights for next iteration's delta computation.

        Args:
            model: The model whose weights to cache.
            subset_params: If set, only cache this many parameters (for memory efficiency).
        """
        self.previous_weights.clear()
        count = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Clone weights - keep on GPU if configured, otherwise move to CPU
                cloned = param.data.detach().clone()
                if not self.weights_on_gpu:
                    cloned = cloned.cpu()
                self.previous_weights[name] = cloned
                count += 1
                if subset_params is not None and count >= subset_params:
                    break
        self.initialized = True

    def update_activations(self, activations: Dict[int, Tensor]) -> None:
        """Cache activation samples for delta_Y computation.

        Args:
            activations: Dictionary mapping layer numbers to activation tensors.
        """
        self.previous_activations.clear()
        for layer_num, activation in activations.items():
            # Sample first element and move to CPU to save GPU memory
            sampled = activation[:1].detach().clone().cpu() if activation.dim() >= 1 else activation.detach().clone().cpu()
            self.previous_activations[layer_num] = sampled

    def compute_weight_deltas(self, model: nn.Module) -> Dict[str, float]:
        """Compute relative weight updates for all cached parameters.

        Args:
            model: The model with current weights.

        Returns:
            Dictionary mapping metric names to relative weight changes.
        """
        if not self.initialized:
            return {}

        metrics = {}
        layer_deltas = {}  # For aggregating per-layer metrics
        layer_per_neuron_stats = {}  # For per-neuron stats aggregation

        for name, param in model.named_parameters():
            if name in self.previous_weights:
                # Compute per-neuron statistics
                delta_stats = compute_weight_delta_per_neuron(param.data, self.previous_weights[name])

                # Create a clean metric name (replace dots with slashes for W&B grouping)
                clean_name = name.replace('.', '/')

                # Log full matrix delta
                metrics[f'delta_W/{clean_name}'] = delta_stats['full']

                # Log per-neuron stats for weight matrices (2D+)
                if param.dim() >= 2:
                    metrics[f'delta_W_per_neuron/{clean_name}/mean'] = delta_stats['per_neuron_mean']
                    metrics[f'delta_W_per_neuron/{clean_name}/std'] = delta_stats['per_neuron_std']
                    metrics[f'delta_W_per_neuron/{clean_name}/max'] = delta_stats['per_neuron_max']
                    metrics[f'delta_W_per_neuron/{clean_name}/min'] = delta_stats['per_neuron_min']

                # Aggregate by layer for summary metrics
                if 'layers' in name:
                    # Extract layer index from name like 'decoder.layers.0.self_attention.linear_qkv.weight'
                    parts = name.split('.')
                    for i, part in enumerate(parts):
                        if part == 'layers' and i + 1 < len(parts):
                            try:
                                layer_idx = int(parts[i + 1])
                                if layer_idx not in layer_deltas:
                                    layer_deltas[layer_idx] = []
                                    layer_per_neuron_stats[layer_idx] = {
                                        'means': [], 'stds': [], 'maxs': [], 'mins': []
                                    }
                                layer_deltas[layer_idx].append(delta_stats['full'])
                                if param.dim() >= 2:
                                    layer_per_neuron_stats[layer_idx]['means'].append(delta_stats['per_neuron_mean'])
                                    layer_per_neuron_stats[layer_idx]['stds'].append(delta_stats['per_neuron_std'])
                                    layer_per_neuron_stats[layer_idx]['maxs'].append(delta_stats['per_neuron_max'])
                                    layer_per_neuron_stats[layer_idx]['mins'].append(delta_stats['per_neuron_min'])
                            except ValueError:
                                pass
                            break

        # Compute per-layer aggregate deltas
        for layer_idx, deltas in layer_deltas.items():
            if deltas:
                avg_delta = sum(deltas) / len(deltas)
                max_delta = max(deltas)
                metrics[f'delta_W_avg/layer_{layer_idx}'] = avg_delta
                metrics[f'delta_W_max/layer_{layer_idx}'] = max_delta

                # Per-neuron aggregates at layer level
                pn_stats = layer_per_neuron_stats.get(layer_idx, {})
                if pn_stats.get('means'):
                    metrics[f'delta_W_per_neuron_avg/layer_{layer_idx}/mean'] = sum(pn_stats['means']) / len(pn_stats['means'])
                    metrics[f'delta_W_per_neuron_avg/layer_{layer_idx}/max'] = max(pn_stats['maxs'])
                    metrics[f'delta_W_per_neuron_avg/layer_{layer_idx}/std'] = sum(pn_stats['stds']) / len(pn_stats['stds'])

        return metrics

    def compute_activation_deltas(
        self,
        current_activations: Dict[int, Tensor],
    ) -> Dict[str, float]:
        """Compute relative activation changes for all cached layers.

        Args:
            current_activations: Dictionary mapping layer numbers to current activations.

        Returns:
            Dictionary mapping metric names to relative activation changes.
        """
        if not self.initialized:
            return {}

        metrics = {}
        for layer_num, curr_act in current_activations.items():
            if layer_num in self.previous_activations:
                delta = compute_activation_delta(
                    curr_act,
                    self.previous_activations[layer_num]
                )
                metrics[f'delta_Y/layer_{layer_num}'] = delta

        return metrics

    def compute_angular_updates(self, model: nn.Module) -> Dict[str, float]:
        """Compute angular changes (direction changes) for all cached parameters.

        Useful for spherical/normalized training analysis (nGPT, EDM2).

        Args:
            model: The model with current weights.

        Returns:
            Dictionary mapping metric names to angular change values.
        """
        if not self.initialized:
            return {}

        metrics = {}
        layer_angular = {}  # For aggregating per-layer metrics

        for name, param in model.named_parameters():
            if name in self.previous_weights:
                angular_stats = compute_angular_update(param.data, self.previous_weights[name])

                # Create a clean metric name
                clean_name = name.replace('.', '/')
                metrics[f'angular/{clean_name}/cos_similarity'] = angular_stats['cos_similarity']
                metrics[f'angular/{clean_name}/degrees'] = angular_stats['angular_change_degrees']

                # Aggregate by layer for summary metrics
                if 'layers' in name:
                    parts = name.split('.')
                    for i, part in enumerate(parts):
                        if part == 'layers' and i + 1 < len(parts):
                            try:
                                layer_idx = int(parts[i + 1])
                                if layer_idx not in layer_angular:
                                    layer_angular[layer_idx] = {
                                        'cos_sims': [],
                                        'degrees': [],
                                    }
                                layer_angular[layer_idx]['cos_sims'].append(angular_stats['cos_similarity'])
                                layer_angular[layer_idx]['degrees'].append(angular_stats['angular_change_degrees'])
                            except ValueError:
                                pass
                            break

        # Compute per-layer aggregate angular metrics
        for layer_idx, stats in layer_angular.items():
            if stats['degrees']:
                avg_degrees = sum(stats['degrees']) / len(stats['degrees'])
                max_degrees = max(stats['degrees'])
                avg_cos = sum(stats['cos_sims']) / len(stats['cos_sims'])
                metrics[f'angular_avg/layer_{layer_idx}/degrees'] = avg_degrees
                metrics[f'angular_avg/layer_{layer_idx}/max_degrees'] = max_degrees
                metrics[f'angular_avg/layer_{layer_idx}/cos_similarity'] = avg_cos

        return metrics

    def clear(self) -> None:
        """Clear all cached state."""
        self.previous_weights.clear()
        self.previous_activations.clear()
        self.initialized = False
