# Megatron-LM Research Baseline

A fork of [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) for
reproducible research comparisons of dense and MoE language models at various
scales. Configurations are tuned for the Swiss AI Alps supercomputer (Clariden
cluster, GH200 Grace-Hopper nodes with 4 GPUs per node, Slingshot-11
interconnect).

Training data is [ClimbMix](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix)
in Megatron binary format with the GPT-2 BPE tokenizer (50257 vocab).

## Leaderboards

Ranked run lists per model size; each entry is a self-contained sbatch + W&B link.

- [`350m-ablation`](_research/leaderboards/350m-ablation/README.md) — 1B-token optimizer ablations (NorMuon / Muon / AdamW)
- [`350m`](_research/leaderboards/350m/README.md) — 15B-token full baseline (placeholder)
- [`760m`](_research/leaderboards/760m/README.md) — 30B-token full baseline (placeholder)
- [`1.3b`](_research/leaderboards/1.3b/README.md) — 100B-token full baseline (placeholder)
- [`2.7b`](_research/leaderboards/2.7b/README.md) — 300B-token full baseline (placeholder)

## Quick start

Intended environment: Clariden (Swiss AI Alps, GH200 nodes, SLURM +
enroot container runtime). Everything below assumes that cluster; to
adapt to a different site, swap the EDF and dataset path and rewrite
the `#SBATCH` headers to fit your scheduler.

**First-time setup** — add to your `~/.bashrc` (or source once per
shell) so `sbatch` picks up sane defaults and the sbatches can find
your dataset:

```bash
export SBATCH_ACCOUNT=<your-slurm-account>        # required
export SBATCH_RESERVATION=<your-reservation>      # optional; omit for normal queue
export WANDB_API_KEY=<your-wandb-key>             # optional; omit to log locally only
export WANDB_PROJECT=<your-wandb-project>         # optional; defaults to megatron-lm-research-baseline
export MEGATRON_DATA_PATH=/path/to/climbmix_small # optional on Clariden (framework defaults to the shared path below); Megatron-binary prefix without .bin/.idx
```

On Clariden, a tokenized ClimbMix copy is available at a shared
read-only path:

```bash
export MEGATRON_DATA_PATH=/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small_megatron/climbmix_small
```

`sbatch` reads `SBATCH_ACCOUNT` / `SBATCH_RESERVATION` natively — no wrapper
needed. The launch framework defaults `MEGATRON_DATA_PATH` to the Clariden
shared ClimbMix path shown above, so on Clariden you can skip it; set it to use
another dataset or off-Clariden. (The older `_research/legacy/` sbatches still
require it and abort at submit time if unset.)

**Running a job.** Runs are composed as **one model size × one recipe** by the
launch framework. Clone into the expected path
(jobs write the package dir, caches, and SLURM logs here), then submit from the framework dir:

```bash
cd /iopsstor/scratch/cscs/$USER
git clone https://github.com/ischlag/megatron-lm-research-baseline.git
cd megatron-lm-research-baseline/_research/launch/framework

# the main baseline — 350M-active MoE, ~15B tokens, 2 nodes:
bash submit.sh --size 350m-moe --recipe normuon            # NorMuon
bash submit.sh --size 350m-moe --recipe adamw              # AdamW baseline
bash submit.sh --size 350m-moe --recipe normuon --dry-run  # inspect args, don't submit
```

The alps3 enroot container + `_research/launch/install_python_deps.sh` set up
the Python environment inside the job; no local install needed. To add your own
idea as a ~15-line recipe, sweep LR or token budget, debug interactively, or
chain long runs with auto-resume, see the **[launch framework
README](_research/launch/framework/README.md)** — the day-to-day entry point for
experiments. For a fast interactive smoke test inside an allocation:

```bash
bash _research/launch/framework/debug.sh --size 175m-moe --recipe adamw --nproc 2 --iters 10
```
Python dependencies (`wandb`, `emerging-optimizers`) are
installed into `_research/packages/` inside the container on first run
via `_research/launch/install_python_deps.sh`; no `pip install` on the
login node is needed.

See [README.nvidia.md](README.nvidia.md) for the original NVIDIA Megatron-LM
documentation.

**AI assistants**: read [AGENTS.md](AGENTS.md) before making changes — it
documents the repo layout, experiment flow, and conventions.

## Configurations

Runs are composed by the launch framework as **one size × one recipe**
([`_research/launch/framework/`](_research/launch/framework/README.md)): a
*size* fixes model dims, batch, token budget and node count; a *recipe* is the
optimizer/architecture idea. Available sizes:

| size | active params | type | tokens | nodes | role |
| --- | ---: | --- | ---: | ---: | --- |
| `175m-moe` | 175M | MoE 16e-tk1-sh1 | ~12B | 1 | cheap LR probe |
| `350m-moe` | 350M | MoE 16e-tk1-sh1 | 15B | 2 | **main baseline** |
| `760m-moe` | 760M | MoE 16e-tk1-sh1 | 30B | 4 | scale-up rung |
| `1.5b-moe` | 1.5B | MoE 16e-tk1-sh1 | 30B | 4 | scale-up rung |
| `1.3b` | 1.3B | dense | 100B | 8 | dense baseline |
| `2.7b` | 2.7B | dense | 300B | 16 | dense baseline |

The MoE ladder shares one shape (16 routed experts top-1 + 1 shared, iso-active
to the dense FFN), so a recipe that wins at 350m transfers up the family. All
configs use GBS=128 (524K tokens/step), bf16 + FP8 e4m3 blockwise, and a linear
LR decay to `MIN_LR` by default (WSD variants in `recipes/wsd/`). Per-size
detail, the recipe list, knob sweeps and interactive debugging are in the
[framework README](_research/launch/framework/README.md);
[`_research/SCALING.md`](_research/SCALING.md) covers which rungs and token
budgets are worth running. Rough dense-baseline wall-clock for reference:
350M/15B ≈8 h (1 node), 760M/30B ≈7 h (4), 1.3B/100B ≈19 h (8), 2.7B/300B
≈60 h (16).

Shared dense Transformer++ dimensions (SwiGLU, RMSNorm, RoPE, GQA):

| | 350M | 760M | 1.3B | 2.7B |
| --- | ---: | ---: | ---: | ---: |
| hidden | 1024 | 1536 | 2048 | 2560 |
| layers | 24 | 24 | 24 | 32 |
| heads / kv_heads | 16 / 4 | 24 / 8 | 32 / 8 | 32 / 8 |
| ffn (SwiGLU) | 2560 | 4096 | 5632 | 7680 |

### Leaderboards

Ranked per-size run lists live under
[`_research/leaderboards/`](_research/leaderboards/README.md). Each entry is a
**frozen, self-contained** sbatch (all hparams pinned, no framework dependency)
plus a W&B link and the git sha it ran at, so `git checkout <sha> && sbatch
<file>` reproduces it bitwise. Boards rank by **min loss at the board's fixed
token budget**; the `entry`/`parent`/`change` columns record lineage.

- [`350m-ablation`](_research/leaderboards/350m-ablation/README.md) — 1B-token
  optimizer ablations, the canonical worked example. Top entry: Aurora (Tilde)
  @ 1e-2, final loss 2.200 (beats the prior NorMuon @ 3.6e-4 leader by 0.024
  nats and best AdamW @ 1e-3 by ~0.12 nats).
- [`350m`](_research/leaderboards/350m/README.md) ·
  [`760m`](_research/leaderboards/760m/README.md) ·
  [`1.3b`](_research/leaderboards/1.3b/README.md) ·
  [`2.7b`](_research/leaderboards/2.7b/README.md) — full-baseline boards, no
  entries yet.

When a framework run is worth keeping, `freeze.sh` bakes it into a board entry
(`bash _research/launch/framework/freeze.sh --size <size> --recipe <recipe>
--board <board> --out auto`); fill the generated header and add a table row per
the promotion steps in the
[leaderboards README](_research/leaderboards/README.md).

## Changes from upstream

### Logging patch (`_research/logging_patch/`)

A monkey-patch layer that hooks into Megatron's `training_log` and
`setup_model_and_optimizer` without modifying upstream source files. All
configuration is via environment variables; the patch is activated by a
two-line import in `pretrain_gpt.py`.

Per-step metrics written to an append-only JSONL log (`<run>.jsonl`) plus a
metadata sidecar (`<run>.meta.json`):

| metric | env var | default |
| --- | --- | --- |
| `train_loss`, `lr`, `grad_norm`, `params_norm`, `tput` | always on | -- |
| MFU (model FLOPs utilization) | always on | -- |
| Per-block FP8-stability activation stats (`amax`, `l2`, `frac_outlier`, `rms`) | `APERTUS_LOG_ACT_STATS` | on |
| Threshold for the `frac_outlier` stat (E4M3 input range proxy) | `APERTUS_LOG_ACT_THRESHOLD` | `240` |
| Top-1 next-token accuracy (TP-aware) | `APERTUS_LOG_TOP1_ACC` | on |
| Per-parameter row-norm CV (Aurora-style neuron utilization proxy) | `APERTUS_LOG_ROW_CV` | on |
| Per-parameter gradient norms | `APERTUS_LOG_PER_LAYER_GRADS` | off |
| MLP per-neuron pre-activation stats (every N steps via `APERTUS_LOG_NEURON_INTERVAL`) | `APERTUS_LOG_NEURON_STATS` | off |
| Loss spike detection (rolling z-score) | `APERTUS_LOG_LOSS_SPIKES` | off |
| Startup phase timeline (sbatch, srun, container, dist init, model build, first iters) | always on | -- |

The JSONL writer scales O(1) per step (no full-file rewrite). An analysis
loader at `_research/analyse/load_runs.py` reads both the new JSONL format and
legacy single-file JSON for backward compatibility.

**Aurora-style neuron utilization** (`APERTUS_LOG_ROW_CV`, `APERTUS_LOG_NEURON_STATS`).
Two complementary probes for the dead-neuron / row-imbalance pathology described
in Tilde's [Aurora optimizer post](https://blog.tilderesearch.com/blog/aurora):
Muon's polar update on tall matrices does not balance row norms, so over training
some MLP neurons receive persistently small updates and stop contributing.

- `row_cv` (free, every step): per-2D-parameter `std/mean` of row L2 norms.
  Cumulative effect on the weight matrix; expected to be near-flat for AdamW
  and Aurora, drift upward for plain Muon. Logged for every matrix param so
  per-MLP / per-attention trends are visible.
- `neuron_stats` (opt-in, every `APERTUS_LOG_NEURON_INTERVAL` steps): forward
  pre-hook on each `*.mlp.linear_fc2` taps the post-gate-mul activation
  (`[s, b, d_ff]`); per-neuron mean\|x\| accumulates each microbatch and is
  summarized per layer (`mean`, `cv`, `dead_frac`, `p10`/`p50`/`p90`). The
  activation-side counterpart of `row_cv`.

Both metrics are rank-local (per TP/EP shard); see `_research/logging_patch/README.md`.

### Muon / NorMuon (`--optimizer muon`, `--optimizer adaptive_muon`)

Muon uses Newton-Schulz orthogonalization on the momentum matrix to take
spectrally-normalized steps. Routing: 2D matrix params use Muon; scalar
params (embeddings, norms, biases) use AdamW via the upstream
`_is_nonlinear_or_embedding` predicate. Relevant flags (we added three
on top of upstream):

- `--muon-scalar-lr` (decouples the AdamW-group LR from the Muon-group LR;
  NorMuon recipe uses matrix LR 3.6e-4 and scalar LR 1.5e-3).
- `--muon-scalar-weight-decay` (set to 0 to keep WD on the Muon matrix
  group only).
- `--adaptive-muon-moment2-method` (`adamuon` default, or `normuon` for
  per-row second-moment rescale, arXiv 2510.05491).

**Must omit `--overlap-param-gather` when training with Muon.** This flag
is safe for AdamW because Adam's update is elementwise: each weight moves
by `lr * m / (sqrt(v) + eps)` regardless of when other weights are
gathered, so pipelining the per-bucket all-gather with the next forward
pass is correctness-neutral. Muon's update is spectral: Newton-Schulz
orthogonalizes the full momentum matrix, and every row's update
direction depends on every other row's magnitude (via the `X Xᵀ X` terms
inside NS5). If `--overlap-param-gather` assembles the matrix from shards
gathered at inconsistent async states, NS5 sees a corrupted matrix, its
output isn't an orthogonal projection of the true momentum, and the step
has the wrong spectrum. Empirically this manifests as monotonically
growing `params_norm` and `grad_norm` from step 1, not just slower
convergence. Keep `--overlap-grad-reduce` and `--use-distributed-optimizer`
(both verified safe for Muon). Throughput cost of dropping
`--overlap-param-gather` at 350M / 1 node is ~2-3%; may grow at larger
scale where param-gather overlap matters more.

### AdEMAMix optimizer (`--optimizer ademamix`)

Ported from the [swiss-ai/Megatron-LM](https://github.com/swiss-ai/Megatron-LM)
fork. AdEMAMix adds a slow EMA on top of Adam's fast EMA for better
convergence at scale (Pagliardini et al., 2025). Relevant flags:

- `--ademamix-alpha` (default 2.0)
- `--ademamix-beta3` (default 0.9999)
- `--ademamix-beta3-warmup` (warmup steps for beta3 in half-life space)
- `--ademamix-alpha-warmup` (warmup steps for alpha)

### xIELU activation (`--xielu`)

Learnable per-layer activation from the Apertus architecture (two parameters
per layer: `alpha_p`, `alpha_n`). Non-gated 2-matrix MLP (up projection,
activation, down projection). Use with `--ffn-hidden-size` set to the full
intermediate dimension (no 2/3 scaling needed unlike SwiGLU).

### Goldfish loss (`--goldfish-loss`)

Token-level memorization suppression (Hans et al., 2024). Masks ~1/k of
target labels using a deterministic hash of the preceding h tokens. Relevant
flags:

- `--goldfish-k` (drop fraction, default 50)
- `--goldfish-h` (context width for hashing, default 50)

### Determinism (`APERTUS_DETERMINISTIC=1`)

Optional flag that enables `torch.use_deterministic_algorithms`,
deterministic cuDNN, and `CUBLAS_WORKSPACE_CONFIG=:4096:8` for bitwise
reproducibility (at a compute cost).

## Syncing with upstream

```bash
git fetch upstream
git merge upstream/main
```

The `upstream` remote points at `NVIDIA/Megatron-LM`. Our changes are
confined to `_research/`, `megatron/core/optimizer/ademamix.py`,
`megatron/core/activations.py` (XIELU class), and small edits to
`arguments.py`, `optimizer/__init__.py`, `optimizer_config.py`, `mlp.py`,
`gpt_dataset.py`, and `pretrain_gpt.py`. Upstream merges should be
low-conflict.
