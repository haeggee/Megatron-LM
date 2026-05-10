# WSDMC: warmup-stable-decay multi-cooldown protocol

A single main run + N cooldowns producing N decayed-loss endpoints. Default
for ablation runs at Stage 2+ of the SCALING.md ladder.

## Files

| file | purpose |
| --- | --- |
| `transformer-pp-350m-wsdmc-aurora-qkn-moe-32e-tk3-sh1.sbatch` | main run (15B, MoE leader, no decay, save every 3B) |
| `wsdmc-cooldown-aurora-qkn-moe-32e-tk3-sh1.sbatch` | paired cooldown (architecture must match the checkpoint; one cooldown per main) |
| `submit-wsdmc.sh` | submits 1 main + 5 cooldowns with `--dependency=afterok` |
| `wsdmc-smoke-main.sbatch` | smoke: real MoE config, 30 iters, ckpts at 10/20/30 |
| `wsdmc-smoke-cooldown.sbatch` | smoke: load iter 10, run 5-iter cooldown |
| `wsdmc-smoke-verify.py` | structural + log checks on the smoke output |

## Mechanics

Main run: WSD with warmup + extended stable phase, **no decay phase** (the
WSD `wsd_anneal_start` is pushed past `train_iters` by setting
`--lr-decay-samples = 2 * train_samples`). Saves checkpoints every
`train_iters / N_CKPTS` iters. Invariant: `train_iters` must equal
`N_CKPTS * save_interval` (final save coincides with end of training).

Each cooldown loads its specific checkpoint via `--ckpt-step`, imposes a
fresh WSD schedule via `--override-opt_param-scheduler`, and trains for
`COOLDOWN_ITERS = round(ckpt_iter * COOLDOWN_FRAC)` more iters with the
WSD decay phase active over those cooldown iters. Result: each cooldown
is a real WSD endpoint at total `D = (1 + COOLDOWN_FRAC) * ckpt_pos`.

Defaults: `N_CKPTS=5`, `COOLDOWN_FRAC=0.2` -> 5 endpoints at
`{1.2, 2.4, 3.6, 4.8, 6.0} * (train_tokens / 5)`. For 350M/15B that is
`{3.6, 7.2, 10.8, 14.4, 18.0}B`.

## Usage

```bash
# Submit the MoE leader's wsdmc run (1 main + 5 cooldown deps)
./_research/launch/submit-wsdmc.sh \
    _research/launch/transformer-pp-350m-wsdmc-aurora-qkn-moe-32e-tk3-sh1.sbatch
```

Override defaults via env:

```bash
WSDMC_N_CKPTS=5 WSDMC_COOLDOWN_FRAC=0.1 \
    ./_research/launch/submit-wsdmc.sh path/to/main.sbatch
```

## Smoke test

Validate save/load + scheduler-override on the actual MoE config before
launching the real 15B run.

```bash
sbatch _research/launch/wsdmc-smoke-main.sbatch
# wait for completion (~5-7 min wall, ~25 min queue + train)
sbatch _research/launch/wsdmc-smoke-cooldown.sbatch
# wait for completion
python3 _research/launch/wsdmc-smoke-verify.py
```

The verifier checks:
- 3 main ckpts at iters 10/20/30 with consistent shard layout
- expert weight shards present (grouped-GEMM 3D tensors)
- cooldown produced its iter_15 ckpt with the same shard layout as main's
  iter_10 (proves load round-trip works for MoE)
- cooldown log shows successful load, monotonically non-increasing LR
  across cooldown iters, no NaN losses

## Generalizing to larger models

Cooldown lengths scale automatically with checkpoint position. For
760M/30B at the same `N_CKPTS=5`, set the main sbatch's
`--train-samples` to 30B-equivalent and `--save-interval` to
`train_iters / 5`; cooldown endpoints become `{7.2, 14.4, 21.6, 28.8, 36.0}B`.
The submit script enforces the `train_iters == N_CKPTS * save_interval`
invariant.
