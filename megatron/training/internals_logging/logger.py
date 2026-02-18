# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Main logger class for model internals logging to W&B."""

from collections import defaultdict
from typing import Any, Dict, List, Optional
import torch
from torch import Tensor
import torch.nn as nn

from .config import InternalsLoggingConfig
from .hooks import InternalsHookManager
from .state_manager import InternalsStateManager
from .metrics import (
    compute_activation_stats,
    compute_gradient_norm,
    compute_gradient_weight_alignment,
    get_param_grad,
)


class InternalsLogger:
    """Main logger class that orchestrates model internals logging to W&B.

    This class coordinates the hook manager, state manager, and metric computation
    to log comprehensive model internals to Weights & Biases.

    Attributes:
        config: Configuration for internals logging.
        hook_manager: Manages forward hooks for capturing activations/attention.
        state_manager: Manages state for delta computations.
    """

    def __init__(
        self,
        config: InternalsLoggingConfig,
        hook_manager: InternalsHookManager,
        state_manager: InternalsStateManager,
    ):
        """Initialize the internals logger.

        Args:
            config: Configuration specifying what to log.
            hook_manager: Hook manager for capturing forward pass data.
            state_manager: State manager for delta computations.
        """
        self.config = config
        self.hook_manager = hook_manager
        self.state_manager = state_manager

    def log_internals(
        self,
        model: nn.Module,
        iteration: int,
        wandb_writer: Any,
    ) -> None:
        """Main entry point for logging model internals.

        Computes all requested metrics and logs them to W&B.

        Args:
            model: The model to analyze (should have hooks registered).
            iteration: Current training iteration.
            wandb_writer: W&B writer instance (wandb module).
        """
        if wandb_writer is None:
            return

        metrics: Dict[str, float] = {}

        # 1. Log activation statistics
        if self.config.log_activation_stats:
            activation_metrics = self._compute_activation_metrics()
            metrics.update(activation_metrics)

        # 2. Log gradient statistics
        if self.config.log_gradient_stats:
            gradient_metrics = self._compute_gradient_metrics(model)
            metrics.update(gradient_metrics)

        # 3. Log relative weight updates (delta_W)
        if self.config.log_relative_updates:
            weight_delta_metrics = self.state_manager.compute_weight_deltas(model)
            metrics.update(weight_delta_metrics)

        # 4. Log angular metrics (direction changes, gradient-weight alignment)
        if self.config.log_angular_metrics:
            # Angular updates (direction changes between iterations)
            angular_metrics = self.state_manager.compute_angular_updates(model)
            metrics.update(angular_metrics)

            # Gradient-weight alignment (radial vs tangential components)
            alignment_metrics = self._compute_gradient_weight_alignment(model)
            metrics.update(alignment_metrics)

        # Log all metrics to W&B
        if metrics:
            wandb_writer.log(metrics, step=iteration)

        # Update state for next iteration
        if self.config.log_relative_updates or self.config.log_angular_metrics:
            self.state_manager.update_weights(model)

        # Clear captured data to free memory
        self.hook_manager.clear_captured_data()

    def _compute_activation_metrics(self) -> Dict[str, float]:
        """Compute activation statistics for all captured layers.

        Returns:
            Dictionary mapping metric names to values.
        """
        metrics = {}

        for layer_num, activation in self.hook_manager.captured_activations.items():
            stats = compute_activation_stats(activation)
            for stat_name, value in stats.items():
                metrics[f'activations/layer_{layer_num}/{stat_name}'] = value

        return metrics

    def _compute_gradient_metrics(self, model: nn.Module) -> Dict[str, float]:
        """Compute per-layer gradient statistics.

        Args:
            model: The model with gradients computed.

        Returns:
            Dictionary mapping metric names to gradient statistics.
        """
        metrics = {}
        layer_grad_norms: Dict[int, List[float]] = defaultdict(list)

        # Debug: count parameters with gradients
        total_params = 0
        params_with_grad = 0
        params_with_main_grad = 0
        for name, param in model.named_parameters():
            total_params += 1
            if param.grad is not None:
                params_with_grad += 1
            if hasattr(param, 'main_grad') and param.main_grad is not None:
                params_with_main_grad += 1

        if total_params > 0 and params_with_grad == 0 and params_with_main_grad == 0:
            print(f"[InternalsLogger] WARNING: No gradients found! "
                  f"total_params={total_params}, params_with_grad={params_with_grad}, "
                  f"params_with_main_grad={params_with_main_grad}")

        for name, param in model.named_parameters():
            if get_param_grad(param) is not None:
                grad_norm = compute_gradient_norm(param)

                # Clean name for metric key
                clean_name = name.replace('.', '/')
                metrics[f'gradients/{clean_name}/norm'] = grad_norm

                # Aggregate by layer
                if 'layers' in name:
                    parts = name.split('.')
                    for i, part in enumerate(parts):
                        if part == 'layers' and i + 1 < len(parts):
                            try:
                                layer_idx = int(parts[i + 1])
                                layer_grad_norms[layer_idx].append(grad_norm)
                            except ValueError:
                                pass
                            break

        # Compute per-layer aggregate metrics
        for layer_idx, norms in layer_grad_norms.items():
            if norms:
                # Total L2 norm for layer: sqrt(sum of squared norms)
                total_norm = (sum(n ** 2 for n in norms)) ** 0.5
                avg_norm = sum(norms) / len(norms)
                max_norm = max(norms)

                metrics[f'gradients_per_layer/layer_{layer_idx}/total_norm'] = total_norm
                metrics[f'gradients_per_layer/layer_{layer_idx}/avg_norm'] = avg_norm
                metrics[f'gradients_per_layer/layer_{layer_idx}/max_norm'] = max_norm

        # Compute gradient flow metrics (ratio between consecutive layers)
        sorted_layers = sorted(layer_grad_norms.keys())
        for i in range(len(sorted_layers) - 1):
            l1, l2 = sorted_layers[i], sorted_layers[i + 1]
            norms_l1 = layer_grad_norms[l1]
            norms_l2 = layer_grad_norms[l2]

            if norms_l1 and norms_l2:
                total_l1 = (sum(n ** 2 for n in norms_l1)) ** 0.5
                total_l2 = (sum(n ** 2 for n in norms_l2)) ** 0.5

                if total_l1 > 1e-10:
                    ratio = total_l2 / total_l1
                    metrics[f'gradient_flow/layer_{l1}_to_{l2}'] = ratio

        return metrics

    def _compute_gradient_weight_alignment(self, model: nn.Module) -> Dict[str, float]:
        """Compute gradient-weight alignment for all parameters.

        Measures whether gradients point along weights (radial) or orthogonal (tangential).
        For spherical training, tangential updates (cos ~ 0) preserve weight norms.

        Args:
            model: The model with gradients computed.

        Returns:
            Dictionary mapping metric names to alignment statistics.
        """
        metrics = {}
        layer_alignments: Dict[int, Dict[str, List[float]]] = defaultdict(
            lambda: {'cos': [], 'radial': [], 'tangential': []}
        )

        for name, param in model.named_parameters():
            if get_param_grad(param) is not None:
                align_stats = compute_gradient_weight_alignment(param)

                # Clean name for metric key
                clean_name = name.replace('.', '/')
                metrics[f'grad_weight_align/{clean_name}/cos'] = align_stats['cos_alignment']
                metrics[f'grad_weight_align/{clean_name}/radial'] = align_stats['radial_component']
                metrics[f'grad_weight_align/{clean_name}/tangential'] = align_stats['tangential_component']

                # Aggregate by layer
                if 'layers' in name:
                    parts = name.split('.')
                    for i, part in enumerate(parts):
                        if part == 'layers' and i + 1 < len(parts):
                            try:
                                layer_idx = int(parts[i + 1])
                                layer_alignments[layer_idx]['cos'].append(align_stats['cos_alignment'])
                                layer_alignments[layer_idx]['radial'].append(align_stats['radial_component'])
                                layer_alignments[layer_idx]['tangential'].append(align_stats['tangential_component'])
                            except ValueError:
                                pass
                            break

        # Compute per-layer aggregate alignment metrics
        for layer_idx, stats in layer_alignments.items():
            if stats['cos']:
                avg_cos = sum(stats['cos']) / len(stats['cos'])
                avg_radial = sum(stats['radial']) / len(stats['radial'])
                avg_tangential = sum(stats['tangential']) / len(stats['tangential'])

                metrics[f'grad_weight_align_avg/layer_{layer_idx}/cos'] = avg_cos
                metrics[f'grad_weight_align_avg/layer_{layer_idx}/radial'] = avg_radial
                metrics[f'grad_weight_align_avg/layer_{layer_idx}/tangential'] = avg_tangential

        return metrics
