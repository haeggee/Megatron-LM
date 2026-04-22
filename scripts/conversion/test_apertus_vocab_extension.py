#!/usr/bin/env python3
# Copyright (c) 2026, Swiss AI. All rights reserved.

"""
Validate that the Apertus loader's vocab-extension logic preserves the
original model's behavior on token IDs that exist in the unextended vocab.

The test loads two HuggingFace copies of the same Apertus model:
  1. The original (unmodified).
  2. A copy whose ``embed_tokens`` and ``lm_head`` tables are extended via
     ``loader_apertus.extend_embedding_table`` to ``--extend-vocab-to``.

It then runs both forward on the *same* random token IDs (all strictly less
than the original vocab size) and verifies that:

  * The first ``old_vocab_size`` rows of the extended embedding/lm_head tables
    are exactly equal to the originals (bitwise).
  * The logits of the extended model, sliced to ``[:, :, :old_vocab_size]``,
    are exactly equal to the original logits (bitwise).

Both properties are *guaranteed* by construction if the extension is correct,
so any deviation indicates a bug.

Usage:
    python scripts/conversion/test_apertus_vocab_extension.py \
        --model-dir /path/to/Apertus-8B \
        --extend-vocab-to 131072 \
        [--batch-size 2 --seq-len 32 --bf16]

This script does NOT exercise the Megatron round-trip (loader -> saver ->
trained checkpoint). For that, convert the model with ``convert.py`` and
run a forward pass through Megatron separately. The mathematical invariant
checked here is, however, the necessary and sufficient condition for the
round-trip to be correct on in-vocab tokens.
"""

import argparse
import os
import sys

import torch

# Make the loader importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "tools", "checkpoint"))
from loader_apertus import extend_embedding_table  # noqa: E402

import transformers  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True, help="Path to the HuggingFace Apertus checkpoint.")
    p.add_argument("--extend-vocab-to", type=int, required=True, help="Target vocab size for the extension.")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--bf16", action="store_true", help="Run in bfloat16 (otherwise float32).")
    p.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Target device for the model. Use 'auto' to shard with device_map='auto' "
             "across all visible GPUs (needed for 70B+ models that don't fit on one GPU).",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_model(model_dir: str, dtype: torch.dtype, device: str):
    kwargs = dict(
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if device == "auto":
        kwargs["device_map"] = "auto"
        model = transformers.AutoModelForCausalLM.from_pretrained(model_dir, **kwargs)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_dir, **kwargs).to(device)
    model.eval()
    return model


def extend_model_in_place(model, new_vocab_size: int):
    """Apply the loader's extension to embed_tokens and lm_head of an HF model.

    We mutate the existing modules in place (swap their ``weight`` Parameter and
    update ``num_embeddings`` / ``out_features``) rather than replacing the
    modules wholesale. This preserves any ``accelerate`` device-placement hooks
    attached by ``device_map="auto"``, which would otherwise be lost and cause
    cross-device tensor errors at forward time.
    """
    embed = model.model.embed_tokens
    lm_head = model.lm_head

    new_embed_w = extend_embedding_table(embed.weight.data.cpu(), new_vocab_size)
    new_head_w = extend_embedding_table(lm_head.weight.data.cpu(), new_vocab_size)

    # Swap embed_tokens.weight in place (keep device/dtype of original).
    e_device, e_dtype = embed.weight.device, embed.weight.dtype
    embed.weight = torch.nn.Parameter(
        new_embed_w.to(device=e_device, dtype=e_dtype),
        requires_grad=embed.weight.requires_grad,
    )
    embed.num_embeddings = new_vocab_size

    # Swap lm_head.weight in place (use the lm_head's own device, which under
    # device_map="auto" can differ from the embedding's device).
    h_device, h_dtype = lm_head.weight.device, lm_head.weight.dtype
    lm_head.weight = torch.nn.Parameter(
        new_head_w.to(device=h_device, dtype=h_dtype),
        requires_grad=lm_head.weight.requires_grad,
    )
    lm_head.out_features = new_vocab_size

    # Update config so the model's internal sanity checks don't trip.
    model.config.vocab_size = new_vocab_size


@torch.no_grad()
def forward_logits(model, input_ids: torch.Tensor) -> torch.Tensor:
    out = model(input_ids=input_ids, use_cache=False)
    return out.logits


def main():
    args = parse_args()
    dtype = torch.bfloat16 if args.bf16 else torch.float32

    print(f"Loading original model from {args.model_dir} (dtype={dtype}, device={args.device})...")
    model_orig = load_model(args.model_dir, dtype, args.device)
    old_vocab_size = model_orig.model.embed_tokens.weight.shape[0]
    hidden = model_orig.model.embed_tokens.weight.shape[1]
    print(f"  original vocab_size={old_vocab_size}, hidden={hidden}")

    if args.extend_vocab_to <= old_vocab_size:
        raise SystemExit(
            f"--extend-vocab-to ({args.extend_vocab_to}) must be > original vocab size ({old_vocab_size})."
        )

    print(f"Loading extended copy and applying extend_embedding_table -> {args.extend_vocab_to}...")
    model_ext = load_model(args.model_dir, dtype, args.device)
    # Snapshot the original tables before extension, on CPU, for the row-equality check.
    orig_embed_cpu = model_ext.model.embed_tokens.weight.data.detach().cpu().clone()
    orig_head_cpu = model_ext.lm_head.weight.data.detach().cpu().clone()
    pre_embed_shape = tuple(model_ext.model.embed_tokens.weight.shape)
    pre_head_shape = tuple(model_ext.lm_head.weight.shape)
    pre_vocab_cfg = model_ext.config.vocab_size
    print(f"  before extension: embed_tokens={pre_embed_shape}, lm_head={pre_head_shape}, "
          f"config.vocab_size={pre_vocab_cfg}")
    # Use the same RNG seed for reproducibility of the new (random) rows.
    torch.manual_seed(args.seed)
    extend_model_in_place(model_ext, args.extend_vocab_to)

    post_embed_shape = tuple(model_ext.model.embed_tokens.weight.shape)
    post_head_shape = tuple(model_ext.lm_head.weight.shape)
    post_vocab_cfg = model_ext.config.vocab_size
    print(f"  after  extension: embed_tokens={post_embed_shape}, lm_head={post_head_shape}, "
          f"config.vocab_size={post_vocab_cfg}")
    assert post_embed_shape == (args.extend_vocab_to, hidden), (
        f"embed_tokens shape {post_embed_shape} does not match expected "
        f"({args.extend_vocab_to}, {hidden})"
    )
    assert post_head_shape == (args.extend_vocab_to, hidden), (
        f"lm_head shape {post_head_shape} does not match expected "
        f"({args.extend_vocab_to}, {hidden})"
    )
    assert post_vocab_cfg == args.extend_vocab_to, (
        f"config.vocab_size={post_vocab_cfg}, expected {args.extend_vocab_to}"
    )
    print(f"OK  extended tables grew by {args.extend_vocab_to - old_vocab_size} rows "
          f"({old_vocab_size} -> {args.extend_vocab_to}).")

    # ---- Check 1: first old_vocab_size rows are bitwise unchanged. ----
    ext_embed_first = model_ext.model.embed_tokens.weight.data[:old_vocab_size].detach().cpu()
    ext_head_first = model_ext.lm_head.weight.data[:old_vocab_size].detach().cpu()
    assert torch.equal(ext_embed_first, orig_embed_cpu), (
        "Extended embed_tokens differs from the original in the [0:old_vocab_size) range!"
    )
    assert torch.equal(ext_head_first, orig_head_cpu), (
        "Extended lm_head differs from the original in the [0:old_vocab_size) range!"
    )
    print("OK  first old_vocab_size rows of embed_tokens and lm_head are bitwise identical.")

    # ---- Check 2: forward outputs match on in-vocab tokens. ----
    torch.manual_seed(args.seed)
    # When using device_map='auto', the embedding's device is the right place
    # to put input_ids; HF dispatches the rest internally.
    input_device = model_orig.model.embed_tokens.weight.device
    input_ids = torch.randint(
        low=0, high=old_vocab_size,
        size=(args.batch_size, args.seq_len),
        dtype=torch.long, device=input_device,
    )

    logits_orig = forward_logits(model_orig, input_ids)
    logits_ext = forward_logits(model_ext, input_ids.to(model_ext.model.embed_tokens.weight.device))
    logits_ext_sliced = logits_ext[..., :old_vocab_size]
    # Bring both to CPU so device_map='auto' shards (which can place the lm_head on
    # different ranks for the two models) don't break the comparison.
    logits_orig = logits_orig.detach().cpu()
    logits_ext_sliced = logits_ext_sliced.detach().cpu()

    assert logits_orig.shape == logits_ext_sliced.shape, (
        f"Shape mismatch after slicing: {logits_orig.shape} vs {logits_ext_sliced.shape}"
    )

    # In bf16 the matmul reductions are non-deterministic across kernel choices, so
    # bit-equality is not always achievable; require exact equality only in fp32.
    if dtype == torch.float32:
        if not torch.equal(logits_orig, logits_ext_sliced):
            max_abs_diff = (logits_orig - logits_ext_sliced).abs().max().item()
            raise AssertionError(
                f"FP32 logits not bitwise identical, max abs diff = {max_abs_diff}"
            )
        print("OK  fp32 logits are bitwise identical on in-vocab tokens.")
    else:
        diff = (logits_orig.float() - logits_ext_sliced.float()).abs()
        max_abs = diff.max().item()
        max_rel = (diff / logits_orig.float().abs().clamp(min=1e-6)).max().item()
        # bf16 forward through the same weights should agree to within a few ulps.
        tol = 1e-2
        print(f"     bf16 logits max abs diff = {max_abs:.3e}, max rel diff = {max_rel:.3e}")
        if max_abs > tol:
            raise AssertionError(f"bf16 logits diverge beyond tolerance ({max_abs} > {tol})")
        print("OK  bf16 logits agree within tolerance on in-vocab tokens.")

    print("\nAll checks passed: vocab extension preserves the original model's behavior "
          "on tokens within the original vocabulary.")


if __name__ == "__main__":
    main()
