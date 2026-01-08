#!/usr/bin/env python3
'''
This script converts a Megatron torch checkpoint to a Megatron torch_dist checkpoint.
'''
import sys
import torch
from megatron.core.enums import ModelType
from megatron.training.training import setup_model_and_optimizer
from megatron.training.initialize import initialize_megatron
from megatron.training.global_vars import get_args
from pretrain_gpt import model_provider


def validate_gpu_availability(args):
    """Validate that sufficient GPUs are available for the conversion."""
    tp = args.tensor_model_parallel_size
    pp = args.pipeline_model_parallel_size
    required_gpus = tp * pp

    # If TP=1 and PP=1, no GPUs needed (CPU mode)
    if required_gpus == 1:
        print(f"INFO: Running with TP={tp}, PP={pp} (single process, CPU mode)")
        return

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print(f"ERROR: Conversion with TP={tp}, PP={pp} requires {required_gpus} GPUs, "
              f"but CUDA is not available on this system.")
        print("SOLUTION: Either run on a GPU-enabled machine, or set TP=1 and PP=1 for CPU-only conversion.")
        sys.exit(1)

    # Check number of available GPUs
    available_gpus = torch.cuda.device_count()
    if available_gpus < required_gpus:
        print(f"ERROR: Conversion requires {required_gpus} GPUs (TP={tp} × PP={pp}), "
              f"but only {available_gpus} GPU(s) detected.")
        print(f"SOLUTION: Either:")
        print(f"  1. Run on a machine with {required_gpus}+ GPUs")
        print(f"  2. Reduce parallelism (e.g., set TP=1, PP=1 in the config)")
        print(f"  3. Skip Stage 2 conversion (set CONVERT_TO_TORCH_DIST=false)")
        sys.exit(1)

    # Success
    print(f"✓ GPU requirement met: {required_gpus} GPUs needed, {available_gpus} available")


def main():
    args_defaults = {
        "transformer_impl": "transformer_engine",
        "use_checkpoint_args": True,
        "ckpt_format": "torch",           # Source format
        "ckpt_convert_format": "torch_dist",  # Target format
        "no_load_rng": True,
        "no_load_optim": True,
        "no_save_optim": True,
        "untie_embeddings_and_output_weights": True,
        "exit_on_missing_checkpoint": True,
        # Fake args required by Megatron initialization
        "micro_batch_size": 1,
        "train_iters": 1,
        "lr": 0.0,
    }
    initialize_megatron(
        args_defaults=args_defaults,
    )
    args = get_args()
    assert args.load is not None, "You must specify --load"
    assert args.ckpt_convert_save is not None, "You must specify --ckpt-convert-save"

    # Validate GPU availability before proceeding
    validate_gpu_availability(args)

    setup_model_and_optimizer(model_provider, ModelType.encoder_or_decoder)


if __name__ == "__main__":
    main()
