# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Hook management for capturing model internals during forward pass."""

from typing import Any, Callable, Dict, List, Tuple
import torch
from torch import Tensor
import torch.nn as nn

from .config import InternalsLoggingConfig


class InternalsHookManager:
    """Manages forward hooks for capturing model internals.

    This class registers PyTorch forward hooks on transformer layers to capture
    activations for logging purposes. Hooks are only active during designated
    capture iterations to minimize performance impact.

    Attributes:
        config: Configuration for internals logging.
        hooks: List of registered hook handles.
        captured_activations: Dictionary mapping layer numbers to captured activations.
        should_capture: Flag indicating whether capture is currently enabled.
    """

    def __init__(self, config: InternalsLoggingConfig):
        """Initialize the hook manager.

        Args:
            config: Configuration specifying which layers/metrics to log.
        """
        self.config = config
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.captured_activations: Dict[int, Tensor] = {}
        self.should_capture = False

    def register_hooks(self, model: nn.Module) -> None:
        """Register forward hooks on transformer layers.

        Args:
            model: The model to register hooks on. Should contain TransformerLayer modules.
        """
        for name, module in model.named_modules():
            # Check if this is a TransformerLayer by looking for layer_number attribute
            if hasattr(module, 'layer_number') and hasattr(module, 'self_attention'):
                layer_num = module.layer_number

                if self.config.should_log_layer(layer_num):
                    # Register hook for activation capture
                    hook = module.register_forward_hook(
                        self._make_activation_hook(layer_num)
                    )
                    self.hooks.append(hook)

    def _make_activation_hook(self, layer_num: int) -> Callable:
        """Create a forward hook for capturing activations.

        Args:
            layer_num: The layer number this hook is attached to.

        Returns:
            Hook function that captures the layer output.
        """
        def hook(
            module: nn.Module,
            input: Tuple[Tensor, ...],
            output: Any,
        ) -> None:
            if not self.should_capture:
                return

            # TransformerLayer.forward returns (output, context)
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Detach and sample to save memory (take first sample from batch)
            if isinstance(hidden_states, Tensor):
                # Shape is [seq, batch, hidden]
                sampled = hidden_states
                # TODO: Potentially only keep a small sample to reduce memory
                # if sampled.dim() >= 2:
                #     sampled = sampled[:min(8, sampled.size(0))]

                # Only store if we haven't captured this layer yet
                if self.captured_activations.get(layer_num, None) is None:
                    self.captured_activations[layer_num] = sampled

        return hook

    def enable_capture(self) -> None:
        """Enable capture for the current iteration.

        Clears previous captured data and sets the capture flag.
        """
        self.should_capture = True
        self.captured_activations.clear()

    def disable_capture(self) -> None:
        """Disable capture after logging."""
        self.should_capture = False

    def clear_captured_data(self) -> None:
        """Clear captured data to free memory."""
        self.captured_activations.clear()

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_hooks()
