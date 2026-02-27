# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from .base_context import BaseInferenceContext
from .dynamic_block_allocator import BlockAllocator
from .static_context import StaticInferenceContext

# Lazy-load deprecated dynamic_context imports (triggers expensive flashinfer load).
# These will be removed in megatron-core 0.14.
_DYNAMIC_CONTEXT_ATTRS = {
    "ActiveRequestCountOverflowError",
    "BlockOverflowError",
    "ContextOverflowError",
    "DynamicInferenceContext",
    "RequestOverflowError",
    "TokenOverflowError",
}


def __getattr__(name):
    if name in _DYNAMIC_CONTEXT_ATTRS:
        import warnings

        warnings.warn(
            f"Importing {name} from `megatron.core.inference.contexts` is deprecated "
            "and will be removed in `megatron-core` 0.14. "
            "Import directly from `megatron.core.inference.contexts.dynamic_context` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from .dynamic_context import (
            ActiveRequestCountOverflowError,
            BlockOverflowError,
            ContextOverflowError,
            DynamicInferenceContext,
            RequestOverflowError,
            TokenOverflowError,
        )

        _attrs = {
            "ActiveRequestCountOverflowError": ActiveRequestCountOverflowError,
            "BlockOverflowError": BlockOverflowError,
            "ContextOverflowError": ContextOverflowError,
            "DynamicInferenceContext": DynamicInferenceContext,
            "RequestOverflowError": RequestOverflowError,
            "TokenOverflowError": TokenOverflowError,
        }
        # Cache to avoid repeated warnings
        for k, v in _attrs.items():
            globals()[k] = v
        return _attrs[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
