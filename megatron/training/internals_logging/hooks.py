# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Hook management for capturing model internals during forward pass."""

from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
from torch import Tensor
import torch.nn as nn

from .config import InternalsLoggingConfig


class InternalsHookManager:
    """Manages forward hooks for capturing model internals.

    This class registers PyTorch forward hooks on transformer layers to capture
    activations and attention patterns for logging purposes. Hooks are only
    active during designated capture iterations to minimize performance impact.

    Attributes:
        config: Configuration for internals logging.
        hooks: List of registered hook handles.
        captured_activations: Dictionary mapping layer numbers to captured activations.
        captured_attention: Dictionary mapping layer numbers to captured attention weights.
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
        self.captured_attention: Dict[int, Tensor] = {}
        self.should_capture = False
        # Keep track of attention modules for enabling/disabling capture
        self._attention_modules: List[nn.Module] = []

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

                    # Register hook on self_attention for attention capture
                    if self.config.log_attention_patterns and hasattr(module, 'self_attention'):
                        attn_module = module.self_attention
                        # Try to find the core attention module for attention weights
                        if hasattr(attn_module, 'core_attention'):
                            core_attn = attn_module.core_attention
                            attn_hook = core_attn.register_forward_hook(
                                self._make_attention_hook(layer_num)
                            )
                            self.hooks.append(attn_hook)
                            # Track attention module for enabling/disabling capture
                            if hasattr(core_attn, 'capture_attention_probs'):
                                self._attention_modules.append(core_attn)

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
                # Sample: take first token position and first batch element
                # Shape is typically [seq, batch, hidden] or [batch, seq, hidden]
                sampled = hidden_states.detach()
                # Only keep a small sample to reduce memory
                if sampled.dim() >= 2:
                    sampled = sampled[:min(8, sampled.size(0))]
                self.captured_activations[layer_num] = sampled

        return hook

    def _make_attention_hook(self, layer_num: int) -> Callable:
        """Create a forward hook for capturing attention weights.

        Args:
            layer_num: The layer number this hook is attached to.

        Returns:
            Hook function that captures attention weights.
        """
        def hook(
            module: nn.Module,
            input: Tuple[Tensor, ...],
            output: Any,
        ) -> None:
            if not self.should_capture:
                return

            # Check if module has captured attention probs
            if hasattr(module, 'last_attention_probs') and module.last_attention_probs is not None:
                attn_probs = module.last_attention_probs.detach()
                # Sample to reduce memory: take first batch element, all heads
                if attn_probs.dim() >= 3:
                    attn_probs = attn_probs[:1]  # First batch element
                self.captured_attention[layer_num] = attn_probs

        return hook

    def enable_capture(self) -> None:
        """Enable capture for the current iteration.

        Clears previous captured data and sets the capture flag.
        Also enables attention capture on DotProductAttention modules.
        """
        self.should_capture = True
        self.captured_activations.clear()
        self.captured_attention.clear()
        # Enable attention capture on attention modules
        for attn_module in self._attention_modules:
            attn_module.capture_attention_probs = True

    def disable_capture(self) -> None:
        """Disable capture after logging.

        Also disables attention capture and clears stored attention probs.
        """
        self.should_capture = False
        # Disable attention capture on attention modules
        for attn_module in self._attention_modules:
            attn_module.capture_attention_probs = False
            attn_module.last_attention_probs = None

    def clear_captured_data(self) -> None:
        """Clear captured data to free memory."""
        self.captured_activations.clear()
        self.captured_attention.clear()

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_hooks()
