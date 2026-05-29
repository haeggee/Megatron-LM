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
export MEGATRON_DATA_PATH=/path/to/climbmix_small # required; Megatron-binary prefix without .bin/.idx
```

On Clariden, a tokenized ClimbMix copy is available at a shared
read-only path:

```bash
export MEGATRON_DATA_PATH=/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small_megatron/climbmix_small
```

`sbatch` reads `SBATCH_ACCOUNT` / `SBATCH_RESERVATION` natively — no
wrapper needed. The sbatches abort with a clear error at submit time
if `MEGATRON_DATA_PATH` is unset.

**Running a job:**

```bash
# Clone into the expected path. The sbatches write the Python package
# dir, caches, and SLURM logs to this exact location; cloning elsewhere
# will scatter outputs across two directories.
cd /iopsstor/scratch/cscs/$USER
git clone https://github.com/ischlag/megatron-lm-research-baseline.git
cd megatron-lm-research-baseline

# Submit. The alps3 enroot container +
# _research/launch/install_python_deps.sh handle the Python environment
# inside the job; no local install required.
sbatch _research/launch/transformer-pp-350m-adamw.sbatch
# or the NorMuon variant:
sbatch _research/launch/transformer-pp-350m-muon.sbatch
# or a 1B-token quick reference (AdamW, ~30 min — good first smoke test):
sbatch _research/launch/transformer-pp-350m-ablation.sbatch
```

To try a variant (different optimizer, LR, schedule, etc.), copy an
existing sbatch and edit it. Frozen ablation runs live under
`_research/leaderboards/<size>/runs/`.

Python dependencies (`transformers`, `wandb`, `emerging-optimizers`) are
installed into `_research/packages/` inside the container on first run
via `_research/launch/install_python_deps.sh`; no `pip install` on the
login node is needed.

See [README.nvidia.md](README.nvidia.md) for the original NVIDIA Megatron-LM
documentation.

**AI assistants**: read [AGENTS.md](AGENTS.md) before making changes — it
documents the repo layout, experiment flow, and conventions.

## Changes from upstream

## Configurations

Transformer++ baselines (SwiGLU, RMSNorm, RoPE, GQA, AdamW, WSD schedule,
bf16) on ClimbMix with GPT-2 BPE tokenizer. All configs use GBS=128
sequences (524K tokens/step) and are tuned for GH200 nodes with 4 GPUs each.

| config | params | tokens | nodes | GPUs | DP | MBS | est. wall | GPU-h |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `transformer-pp-350m-ablation` | 350M | 1B | 1 | 4 | 4 | 16 | ~30 min | 2 |
| `transformer-pp-350m-{adamw,muon}` | 350M | 15B | 1 | 4 | 4 | 16 | ~8 h | 31 |
| `transformer-pp-760m-adamw` | 760M | 30B | 4 | 16 | 16 | 4 | ~7 h | 115 |
| `transformer-pp-1.3b-adamw` | 1.3B | 100B | 8 | 32 | 32 | 2 | ~19 h | 596 |
| `transformer-pp-2.7b-adamw` | 2.7B | 300B | 16 | 64 | 64 | 1 | ~60 h | 3,834 |

The `-muon` variant at 350M uses NorMuon (adaptive_muon + normuon) with
matrix LR 3.6e-4 and scalar LR 1.5e-3; it differs from `-adamw` only in
the optimizer block (validated on `-ablation`: NorMuon beats AdamW by
~0.10 nats final loss).

### Architectures

| | 350M | 760M | 1.3B | 2.7B |
| --- | ---: | ---: | ---: | ---: |
| hidden | 1024 | 1536 | 2048 | 2560 |
| layers | 24 | 24 | 24 | 32 |
| heads / kv_heads | 16 / 4 | 24 / 8 | 32 / 8 | 32 / 8 |
| ffn (SwiGLU) | 2560 | 4096 | 5632 | 7680 |
| peak LR | 3e-4 | 2.5e-4 | 2e-4 | 1.6e-4 |

### Leaderboards

Ranked per-size run lists live under
[`_research/leaderboards/`](_research/leaderboards/README.md). Each entry
has a self-contained sbatch (no env-var branching, all hparams pinned)
plus a W&B link and the git sha it was executed at. Current leaderboards:

- [`350m-ablation`](_research/leaderboards/350m-ablation/README.md) —
  1B-token optimizer ablations. Top entry: Aurora (Tilde) @ 1e-2, final
  loss 2.200 (beats prior NorMuon @ 3.6e-4 leader by 0.024 nats and best
  AdamW @ 1e-3 by ~0.12 nats).
- `350m`, `760m`, `1.3b`, `2.7b` — placeholder dirs; no full runs yet.

### Launch vs leaderboards

| folder | purpose |
| --- | --- |
| `_research/launch/` | launchable sbatches: baseline full-run per size (`-adamw`, `-muon`) plus a short `-ablation` 1B-token reference. All hparams pinned. |
| `_research/leaderboards/` | historical ranked runs. Each entry is a frozen, reproducible sbatch + W&B link. |

Workflow for a new ablation: copy an existing sbatch, edit the optimizer
/ LR / schedule, run it, and if it wins snapshot the file into
`leaderboards/<size>/runs/` and add a row to that `README.md`.

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

---

# Swiss AI Megatron-LM launcher (multimodality)

<h1 align="center">
<p>Megatron-LM
</h1>

<h3 align="center">
    <p>Enhanced launcher for Megatron-LM tailored to train LLMs at scale in Slurm-based clusters</p>
</h3>

<!-- TOC -->

- [Introduction](#introduction)
- [Setup](#setup)
    - [Submit a Run](#submit-a-run)
- [Data](#data)
    - [Tokenization](#tokenization)
    - [Set the Datasets in Megatron](#set-the-datasets-in-megatron)
    - [Data mixtures](#data-mixtures)
- [Checkpointing](#checkpointing)
    - [Resuming from a checkpoint](#resuming-from-a-checkpoint)
    - [Converting checkpoints to huggingface](#converting-checkpoints-to-huggingface)
- [Fault Tolerance](#fault-tolerance)
- [QOL Improvements](#qol-improvements)
    - [Exit & Save triggers](#exit--save-triggers)
    - [Debugging](#debugging)
    - [Backup codebase](#backup-codebase)
- [Contribute](#contribute)
    - [Copy Your Wandb Logs to a New Project](#copy-your-wandb-logs-to-a-new-project)

<!-- /TOC -->
# Introduction
This repository contains scripts to train large language models (LLMs) at scale using Megatron-LM, specifically tailored for Slurm clusters. It's tailored for large-scale training runs and include mechanism for improved fault tolerance, automatic resumption after crashes or interruptions, automated evaluation submission, and enhanced WANDB logging. Additionally, it provides configurations optimized for peak performance on the Alps Supercomputer.

We also provide scripts to tokenize data using datatrove at scale and tools to convert model weights to the HuggingFace format.

# Setup
1. Clone the Megatron-LM repository to your `iopsstor`.
  ```
  cd /iopsstor/scratch/cscs/$USER/
  git clone https://github.com/swiss-ai/Megatron-LM.git
  ```
2. You need to set the weights & biases environment variable with your key. Ideally you add it `~/.bashrc`.
  ```
  export WANDB_API_KEY=57b215a80f....
  ```
  Now re-login or run `source ~/.bashrc` to make sure it is set. 

## Submit a Run
1. Start the llama 3 8B baseline run by submitting the `submit-llama3-8B.sh` sbatch script. 
  ```
  sbatch submit-llama3-8B.sh
  ```
  You can find all available arguments here: [https://github.com/swiss-ai/Megatron-LM/blob/main/megatron/training/arguments.py](https://github.com/swiss-ai/Megatron-LM/blob/main/megatron/training/arguments.py)
  (We will add <1B, 3B, and 70B baseline runs using the llama3 architecture very soon.)
2. The main logs will be visible ~2min after launching the script in the `Megatron-Clariden` Project on your personal weights and biases page.
3. The local logs will be at `/iopsstor/scratch/cscs/$USER/Megatron-LM/logs` with the slurm outputs and error files at `/iopsstor/scratch/cscs/$USER/Megatron-LM/logs/slurm/training`.

# Data
>[!NOTE]
> On the Alps supercomputer, you can find tokenized datasets in `/capstor/store/cscs/swissai/a06/datasets_tokenized/megatron/sai`

## Tokenization
While Megatron provides data tokenization tools, we use `datatrove`, which enables us reading data in multiple formats (`json`, `parquet`, `csv`...), easy parallelization across multiple nodes, and efficient filtering and deduplication our data.

We have extended `datatrove` ([PR](https://github.com/huggingface/datatrove/pull/304)) to include the [`MegatronDocumentTokenizer`](https://github.com/huggingface/datatrove/blob/22606036e92c8d83268f313f462ee98eceb3fa0b/src/datatrove/pipeline/tokens/megatron_tokenizer.py#L144) pipeline stage, allowing the generation of files containing tokenized documents compatible with NeMo/Megatron.

All Megatron tokenized files, also referred to as *file prefixes*, consist of two file types:
- The `.bin` files contain raw tokens, where each token is either 4 bytes (for vocabularies > 65535) or 2 bytes.
- The `.idx` files contain metadata about the corresponding `.bin` files. For more details on creating these files, refer to [this example](https://github.com/huggingface/datatrove/blob/22606036e92c8d83268f313f462ee98eceb3fa0b/src/datatrove/pipeline/tokens/megatron_tokenizer.py#L59-L72).
>[!CAUTION]
> 🚨 On the Alps Supercomputer, ensure you **DO NOT WRITE**  under **`/capstor/store`**. Always use `/iopsstor/scratch`! 🚨

We include a launcher to tokenize data at scale using multiple nodes in `scripts/tokenization`. Start by preparing your workspace using the `scripts/tokenization/prepare_dumps.py` script, which identifies parquet files in the specified directory, filters them based on criteria (check `--filter-in` and `--filter-out`), and splits the workload evenly across `--n-dumps`.  This process generates *.txt* files for each dump, specifying the files to process.

Once the workspace is ready, configure the tokenization job by setting the tokenizer, the number of datatrove parallel workers per node, and the directory we prepared with `scripts/tokenization/prepare_dumps.py` in `scripts/tokenization/submit_tokenization.sh`. Running this script will submit multiple Slurm jobs, with each job responsible for processing one dump on a single node.
>[!CAUTION]
> 🚨 Ensure the `SBATCH --environment` flag in `scripts/tokenization/tokenize.sh` is correctly configured for your environment 🚨

Before running large-scale jobs, it is recommended to optimize the number of datatrove parallel workers  (`datatrove`'s [`LocalPipelineExecutor`](https://github.com/huggingface/datatrove/blob/22606036e92c8d83268f313f462ee98eceb3fa0b/src/datatrove/executor/local.py#L15) `tasks` & `workers`) and the input file size. You can easily modify the later with `datatrove` using the [`max_file_size`](https://github.com/huggingface/datatrove/blob/22606036e92c8d83268f313f462ee98eceb3fa0b/src/datatrove/pipeline/writers/parquet.py#L21) configuration of the `ParquetWriter`.

For example, on the Alps supercomputer, the best configuration involved processing parquet files of 500 MB with Snappy compression and using 28 datatrove workers per node, achieving a throughput of ~70 million tokens per second per node. More details [here](https://docs.google.com/presentation/d/1t12axPhvjpuxGQWr1xJIioazKeVZ212ewuyil5uWMnQ/edit#slide=id.p).

## Set the Datasets in Megatron
In Megatron, we will specify datasets using the `--data-path` configuration. This field expects a list of tokenized file prefixes, and optionally a weight for each file. If the weights are not specified, Megatron will automatically compute them based on the number of sequences it can extract from each file prefix. To simplify this process, we will use the `scripts/create_data_config.py` script to automatically generate the file prefix list by recursively searching for all file prefixes in the comma-separated list of directories specified in the `DATASETS` variable of the launcher script.

## Data mixtures
As we just mentioned in the previous section, we only need to specify a directory with file prefixes, and Megatron will assign a weight to each prefix based on the number of samples it contains.  

For data mixtures, we will follow the same workflow but add an additional step where we create a folder for our data mixture containing symlinks to the original datasets.  

To achieve this, we have developed the `scripts/tools/create_data_mixture.py` script, where we only need to specify the `--folders` for the mixture, the `--weights` of each folder, and an `--output` directory to store the created data mixture.

```
python3 scripts/tools/create_data_mixture.py --folders datasets/fineweb-edu fineweb-2 --weights 0.8 0.2 --output datasets/fw-edu-fw-2-mixture
```

Upon successfully creating a mixture, we will see its statistics, such as the number of tokens, the number of file prefixes per dataset, and the total size of the mixture.  

Keep in mind that the mixture will be created **without repetition**. This means that we will construct the mixture while respecting the weights until a dataset is exhausted.

# Checkpointing
>[!CAUTION]
> 🚨 On the Alps Supercomputer, ensure you **DO NOT WRITE**  under **`/capstor/store`**. Use `/iopsstor/scratch` instead 🚨

Checkpointing is a critical component of LLM training. It must be fast to minimize disruption to training and complete, meaning it captures not only the model weights but also the optimizer states, DataLoader states and RNG states.

We use the PyTorch distributed checkpointing backend (`--ckpt-format torch_dist`) leveraging the asynchronous checkpointing option and parallelizing both storing and loading checkpoints within all the devices. The checkpoints are topology-agnostic, allowing them to be loaded with a different topology from the one used to store them. Each process will store 2 files (Default value for `thread_count` [[1](https://github.com/NVIDIA/Megatron-LM/blob/55cdfc1e8bfe116f54dbe6e48ff70cc92c9f4a91/megatron/core/dist_checkpointing/strategies/torch.py#L614)], [[2](https://pytorch.org/docs/stable/distributed.checkpoint.html#torch.distributed.checkpoint.FileSystemWriter)]) containing the state of the run.

In Alps, writing a Llama3-70B checkpoint to `/iopsstor/scratch` blocks training for approximately 40 seconds. Be aware that these checkpoints are huge: A Llama3-70B model checkpoint takes **920 GB** and the Llama3-8B version takes **105 GB** (model weights in bf16 and optimizer states in fp32).

In the launcher set `CHECKPOINT_STEPS` as the frequency every how many steps you want to store a checkpoint.

## Resuming from a checkpoint
In Megatron, we use `--save` and `--load` to specify where checkpoints are read from and saved to.  

With each checkpoint save, the file `latest_checkpointed_iteration.txt` will be updated in the `--save` directory, containing a reference to the last saved checkpoint. There is currently no feature to just keep the last `N` checkpoints.

When resuming a run, the application will attempt to load the checkpoint referenced in the `latest_checkpointed_iteration.txt` file from the `--load` directory. To load a checkpoint from a different iteration, you will need to manually modify the reference inside `latest_checkpointed_iteration.txt`.

## Converting checkpoints to huggingface
You can use `tools/checkpoint/convert.py` to convert megatron checkpoints to huggingface.
Minimal example:
```
python tools/checkpoint/convert.py \
	--model-type GPT \
	--loader core \
	--saver swissai_hf \
	--load-dir CHECKPOINT_PATH \
	--save-dir SAVE_DIR \
	--hf-tokenizer HF_TOKENIZER_NAME  # Optional, set it to save the tokenizer config in `SAVE_DIR`.
```
Note that you will need to convert checkpoints to `torch` format if the megatron checkpoints are using `torch_dist` via the `scripts/conversion/torchdist_2_torch.py`.
Minimal example:
```
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun scripts/conversion/torchdist_2_torch.py \
	--bf16 \
	--load CHECKPOINT_PATH \
	--ckpt-convert-save INTERMEDIATE_CHECKPOINT_PATH \
	--ckpt-step ITERATION_STEP  # Optional, if not specified you will load the latest checkpoint available.
```
See `scripts/conversion/README.md` for more information.


# Fault Tolerance
Training of LLMs often spans several weeks or even months. However, compute clusters typically enforce job time limits of 12 to 24 hours. This results in frequent training interruptions, either due to these time limits or hardware crashes.

To avoid constantly monitoring the training process, we have designed a fault-tolerance system that ensures seamless continuity of training. The system handles:

- Maintaining Continuous Job Execution. The launcher ensures there is always one active job running and another queued. This is achieved using the `sbatch --dependency=singleton` flag, which guarantees that only one job with the specified name (owned by the user) can be running or suspended at any given time.
- Graceful Exit Before Time Limit. The launcher saves a checkpoint and exits gracefully a few minutes before the job time limit is reached. This allows checkpoint frequency to remain independent of the Slurm time limit while utilizing the full time window for training. This is achieved with `#SBATCH --signal=SIGUSR2@600`, which sends a `SIGUSR2` signal 600 seconds before the time limit. The signal is captured during the run to trigger the checkpoint save and exit.
- Automatic Recovery: Automatically resumes training from the most recent checkpoint, recovering model weights, optimizer states, Dataloader states and WANDB logging.

>[!NOTE]
> To avoid wasting resources, we have disabled by default the first feature that keeps one job running while another remains in the queue. Set `AUTO_JOB_REQUEUE=true` to enable it once you are sure of the cost involved in running this experiment

Additionally, to prevent a job from getting stuck failing continuously, we have added some guardrails at critical points in the application to ensure that the job can be executed correctly; otherwise, we will cancel all jobs.

# QOL Improvements
## Exit & Save triggers
By default, Slurm only allows the job owner to interact with a running job, which creates challenges during long runs when multiple team members are responsible for monitoring the process. To address this limitation, we have implemented a mechanism based on the presence of specific files (triggers):
 - **Save Trigger**: When the save trigger file is detected, the system will schedule a checkpoint save.
 - **Exit Trigger**: When the exit trigger file is detected, the system will gracefully exit the run and cancel all remaining jobs.
This allows team members (not just the job owner) to intervene when necessary. The verification of these files will be carried out every `args.log_interval` steps.

## Debugging
We have incorporated two very useful debugging utilities into the launcher:  
- **NCCL Debug**: Set `LOG_NCCL=true` to store all `NCCL_DEBUG=INFO` logs in the job’s `$DEBUG_DIR` folder, with separate files per GPU and per node (check `NCCL_DEBUG_FILE`).  
- **NSYS Profiler**: Set `NSYS_PROFILER=true` to extract traces of a job using the NSYS profiler. Check the `--profile-*` arguments available in [`megatron/training/arguments.py`](megatron/training/arguments.py).
## Backup codebase
If we set `BACKUP_CODEBASE` to `true` in the launcher scripts we will copy the Megatron-LM codebase specified in `MEGATRON_LM_DIR` to the experiment folder so we can use it across multiple runs

# Contribute
You can submit issues and create branches on `https://github.com/swiss-ai/Megatron-LM`. The main branch is protected so you won't be able to directly commit to it.
1. Pull the latest version
  ```
  git pull
  ```
2. Create a branch with a descriptive name
  ```
  git checkout -b my-contribution-branch
  ```
3. Make your changes locally and commit them. Make sure your latest commit works as expected. Do not commit unnecessary files. 
4. sync with the latest main
  ```
  git pull origin main
  ```
5. Push your commit to your branch
  ```
  git push origin my-contribution-branch
  ```
6. Go to `https://github.com/swiss-ai/Megatron-LM. 
7. GitHub will have a pop-up proposing to you to create a new PR. Click on that green button. 
8. IMPORTANT: Change the base repository to `swiss-ai/Megatron-LM` or your PR will be submitted to the official `NVIDIA/Megatron-LM` repo!
9. In your PR explain what you did and why. 
10. If you make a contribution with a run, either use the wandb web UI to move your run into a new folder (select the run in the runs table and a move button will appear) or use the `scripts/copy_wandb_project.py` to create a copy in a new fresh wandb project that contains only the relevant run and the respective baseline (but in this case all runs will start from step 0). See instructions in the next section on how to use the copy script.
11. If you want to archive your logs, move your log folder `~/Megatron-LM/logs/Meg-Runs/your_project_name/your_experiment_name` to `/capstor/store/cscs/swissai/a06/megatron_runs` and share the path in your PR.
12. Add `ischlag` or `TJ-Solergibert` to review your PR.
13. Use this command to return to main branch.
  ```
  git checkout main
  ```

## Copy Your Wandb Logs to a New Project 
To submit your contribution you have to provide a fresh wandb project with the relevant logs. 
1. Go to your `wandb.ai` page and create a new public project (e.g. `contrib_linear_attention`)
2. Use the `copy_wandb_project.py` to create a copy of your run from your personal wandb project. Make sure it contains the relevant ablations and baselines.
  ```
  python scripts/copy_wandb_project.py -se myusername -de myusername -sp Megatron-Clariden -dp contrib_linear_attention -r f5v94x1q  
  ```

## Convert your models to huggingface

To release your models or evalaute them, you can use the megatron to huggingface utility.
See `tools/checkpoint/do-convert.sh` for an end-to-end megatron->hf conversion example.
See `scripts/conversion/README.md` for more information.
