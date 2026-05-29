# Merge: swiss-ai/multimodality/main → optimization fork

**Branch:** `feat/multimodal-optim` (created off `feat/wsdmc`, which is untouched)
**Merge commit:** `4f42dcadb` — true merge, parents `3ca8eefd1` (ours/optim) + `d184151e5` (theirs/multimodality)
**Remote added:** `swiss-ai` → `https://github.com/swiss-ai/Megatron-LM.git`
**Date:** 2026-05-29

## Goal

Combine the multimodality changes from `swiss-ai/Megatron-LM` (vision/audio
modalities, per-modality loss weighting, SeeDNorm, AdEMAMix optimizer, tokenizer
metadata) with this fork's optimization work (muon/master/aurora/rmnp/muown
optimizers, hypersphere/gains, DyT/Derf norms, SFT packing / hybrid-CP / MTP),
to run optimization experiments in a multimodal context.

## Status

- Histories diverged from NVIDIA Megatron base `7fe7f48e1` (Jan 2026): ours +1223
  commits, theirs +396 commits.
- **28 conflicting files**, all resolved. 0 conflict markers; every `.py` parses
  (verified with Python 3.11 `ast.parse`).
- Pre-merge uncommitted optimizer WIP was stashed, then re-applied cleanly after
  the merge (no re-conflicts) — still unstaged in the working tree.
- **Not yet runtime-tested** (no torch in the resolution environment) — needs a
  smoke test before launching jobs (see below).

## How conflicts were resolved

| Area | Resolution |
|---|---|
| Optimizers (`optimizer/__init__`, `optimizer_config`, `ademamix`) | Kept ours (unified `OptimizerConfig` + `ParamKey` overrides; muon/master/etc.) **+** added theirs' standalone `AdEMAMix` class and `--optimizer ademamix` dispatch/config |
| Transformer/model (`attention`, `mlp`, `moe/experts`, `extensions/transformer_engine`, `torch_norm`, `transformer_config`, `gpt_layer_specs`, `gpt_builders`, `ssm/mamba_block`, `model_provider`) | Combined both sides' independent additions: ours' DyT/Derf + fused-residual RMSNorm + offload/`apply_module` + Protocol types; theirs' SeeDNorm + `subln`/differential-attention + XIELU/XSSSLUR2 |
| `arguments.py` | Merged `--optimizer` choices (ours + `ademamix`); widened `--normalization` choices to `LayerNorm/RMSNorm/SeeDNorm/DyT/Derf`; added theirs' standard args; kept ours' `--data-parallel-sharding-strategy` default (`optim_grads_params`) |
| Datasets / dist-ckpt / tests / README | Combined both sides' additions |
| Tokenizer | Kept ours' migration to `megatron/core/tokenizers/`; deleted old `megatron/training/tokenizer/tokenizer.py` + `tools/checkpoint/loader_legacy.py`; re-added package `__init__.py` and repointed two converter imports to `tokenizer_omni_metadata` |
| **Driver files** (`pretrain_gpt.py`, `training.py`, `checkpointing.py`) | **Ours as base** + ported theirs' multimodal data/loss hooks (`modality_weights`, vision/audio loss, `loss_mask_token_ids`, goldfish) and per-modality loss logging in `loss_func` |

## ⚠️ Flagged items (judgment calls — review before relying on them)

1. **`tokens_so_far` progress-log tracking NOT ported.** Kept ours' 2-tuple
   `load_checkpoint` / `get_start_time_from_progress_log`. Ours' FLOP-based
   progress logging is intact; theirs' token-count column is dropped. Porting
   would require threading a 3rd return value through the refactored loop.
2. **Theirs' phase-based wandb logging** (`consumed-tokens`, `phase` /
   `phase-iteration` / `phase-consumed-samples`) and **`params_norm_per_param`**
   logging were dropped — they reference variables not present in ours'
   refactored `training_log`/`training` scope. Ours' `get_canonical_lr_for_logging`
   (multi-group LR logging) was kept.
3. **Theirs' `try/except AssertionError` wrapper** around `pretrain()` (touched
   `exit_trigger` on error) was removed — ours' base has none. Also dropped:
   theirs' `use_legacy_models` branch in `forward_step`, and theirs' per-optimizer
   config-class construction + **xielu-alpha weight-decay exclusion**
   (`--weight-decay-on-xielu-alphas`) — incompatible with ours' unified
   `OptimizerConfig` + `ParamKey` construction, so that WD exclusion is not applied.
4. **`mamba_block.py`**: theirs' SeeDNorm final-norm branch was dropped (kept ours'
   re-export to `hybrid_block`). To use SeeDNorm in the hybrid stack, port it into
   `megatron/core/models/hybrid/hybrid_block.py` (`final_norm` construction).
5. **`moe/experts.py`**: legacy `GroupedMLP` (theirs) removed in favor of ours'
   `TEGroupedMLP` + Protocol interfaces; theirs' XIELU/XSSSLUR2 activation support
   lived only on that legacy class (not on `TEGroupedMLP`).
6. **FSDP** (`arguments.py`, validation block): kept ours' auto-enable of
   `fsdp_manual_registration` when `nccl_ub` + `use_megatron_fsdp`; dropped theirs'
   assertion-based validation of the same flags.
7. **`tools/checkpoint/loader_base.py` / `saver_base.py`**: datasets/tools resolution
   took theirs' converter-fake-groups approach; saver setup-block was a manual
   combine (flagged as the one non-pick-a-side spot). `_vocab_size_with_padding`
   import repointed from the deleted `tokenizer.py` to `tokenizer_omni_metadata`
   in `saver_base.py` and `saver_swissai_hf.py` (offline HF tools, lazy imports —
   not in the training path).
8. **SeeDNorm** is supported only on the torch (`--transformer-impl local`) backend;
   the TE backend raises `NotImplementedError` (matches theirs).

## Next step: smoke test (not yet done — no torch in resolution env)

```bash
python -c "import pretrain_gpt"          # catches import-time breakage
# or a 1-step dry run: pretrain_gpt with --train-iters 1 on a tiny config
```

## Housekeeping

- ~50 untracked `core_nid*` crash dumps in the repo root (not part of the merge) —
  safe to `rm` when convenient.
- Pre-merge optimizer WIP (`emerging_optimizers.py`, `master.py`, `optimizer_config.py`,
  `arguments.py`, launch scripts) is restored and **unstaged** — commit when ready.
