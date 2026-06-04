# Launch framework

> Detail reference for the authoring system. New here? Start at the repo
> [README](../../../README.md) for the big picture; this page is the day-to-day
> how-to. Result tracking lives in the [leaderboards README](../../leaderboards/README.md).

A run is **one model size × one recipe**. You compose them instead of copying a
300-line sbatch. You only ever write the recipe file.

### TL;DR — add a new optimizer/baseline

```bash
cd _research/launch/framework
cp recipes/_template.sh recipes/my-idea.sh   # set OPTIMIZER, EXP_TAG, RECIPE_ARGS
bash submit.sh --size 420m-moe --recipe my-idea --dry-run   # inspect composed args
bash submit.sh --size 420m-moe --recipe my-idea             # launch
# sweep a knob with no new file (flows into the flag AND the run name):
MLR=1e-2 bash submit.sh --size 420m-moe --recipe my-idea
# scale the same recipe up later:
bash submit.sh --size 1.5b-moe --recipe my-idea --nodes 4
```

### TL;DR — debug a recipe interactively

```bash
# inside an alps3 interactive allocation (data defaults to the swissai blend):
bash debug.sh --size 270m-moe --recipe my-idea --nproc 2 --iters 10   # fast inline run
bash debug.sh --size 420m-moe --recipe my-idea -- --lr 1e-4           # flags after -- override
bash debug.sh --size 420m-moe --recipe my-idea --dry-run              # show args, don't launch
# inside an mi300/ROCm interactive allocation (see ../interactive-run.sh header):
bash debug.sh --cluster mi300 --size 420m-moe --recipe my-idea
```

## Why this exists

The old `_research/launch/` had **258 sbatch files, ~56k lines**, each a near-copy
of the last. Two deltanet ablations differed by *3 lines*; everything else —
SLURM header, env exports, network args, logging, the `srun` invocation — was
duplicated 258 times. Adding an idea meant copying 300 lines to change 5, and
nobody could review the diff or trust that the "shared" part hadn't drifted.

Here the shared part lives in exactly one place (`lib/common.sh`) and an idea is
one small file.

## The three layers

| file | what it owns | who edits it |
| --- | --- | --- |
| `lib/common.sh` | SLURM env, paths, logging, all invariant args, the `srun` launch | ~never |
| `sizes/<size>.sh` | model dims, batch, token budget, MoE shape, default nodes/time | when adding a scale rung |
| `recipes/<name>.sh` | `--optimizer` + its flags, LR, regularization — *the idea* | **every experiment** |
| `clusters/<cluster>.sh` | container EDF, mpi flavor, precision, MoE dispatcher, partition/cpus/mem | when adding a machine |

`train.sbatch` just sources the four in order (`size` → `recipe` → `cluster` →
`common`) and hands off. `common.sh` fills in every default the fragments
didn't set.

## Add a new optimizer / baseline

```bash
cp recipes/_template.sh recipes/my-idea.sh
$EDITOR recipes/my-idea.sh          # set OPTIMIZER, EXP_TAG, RECIPE_ARGS, LR...
bash submit.sh --size 420m-moe --recipe my-idea --dry-run   # inspect composed args
bash submit.sh --size 420m-moe --recipe my-idea             # launch
```

A recipe is usually 10–20 lines. The only required fields are `OPTIMIZER`,
`EXP_TAG`, and `RECIPE_ARGS=(...)`. Everything else (`LR`, `WEIGHT_DECAY`,
`ADAM_BETA1/2`, `CLIP_GRAD`, the LR schedule, init std, embedding multiplier)
has a default in `common.sh` you override only when your idea needs it.

The default LR schedule is **linear** decay to `MIN_LR` over all of
`TRAIN_SAMPLES` — it auto-scales with run length and has no cooldown to tune.
For the warmup-stable-decay schedule instead, use the WSD variants in
`recipes/wsd/` (`wsd/muon`, `wsd/muown`, `wsd/master`, `wsd/normuon`) — each just
sources its base recipe and flips `LR_DECAY_STYLE=WSD`; the cooldown/warmup fall
to `common.sh`'s WSD defaults (`minus_sqrt`, ~20% of the run, warmup 0):

```bash
bash submit.sh --size 420m-moe --recipe wsd/muon   # -> 420m-moe-muon-wsd-lr1e-2
```

(Ad-hoc, no recipe: `LR_DECAY_STYLE=WSD bash submit.sh --recipe muon`.)

### Inherit when your idea is "X but with one tweak"

Arch ablations are usually a base optimizer plus a couple of flags. Source the
base and append — see `recipes/deltanet-neg.sh`:

```bash
source "$(dirname "${BASH_SOURCE[0]}")/adaptive-muon.sh"
EXP_TAG=deltanet-neg
RECIPE_ARGS+=(--linear-attention-allow-neg-eigval --linear-attention-freq 4)
```

## Sizes

| size | params | MoE | tokens (default) | nodes | role |
| --- | --- | --- | ---: | ---: | --- |
| `270m-moe` | 270m active | 64e-tk2-sh1 | 15B | 2 | cheap LR probe |
| `420m-moe` | 420m active | 64e-tk2-sh1 | 15B | 2 | **the main baseline** |
| `810m-moe` | 810M active | 64e-tk2-sh1 | 30B | 4 | scale-up rung |
| `1.5b-moe` | 1.5B active | 64e-tk2-sh1 | 30B | 4 | promotion-confirm rung |
| `1.3b` | 1.3B | dense | 100B | 8 | dense baseline |
| `2.7b` | 2.7B | dense | 300B | 16 | dense baseline |

The MoE ladder (270m → 1.5b) shares one routing recipe (64 routed top-2 + 1
shared at half-width, `moe-ffn ≈ hidden/1.75`, ~6% non-embedding sparsity;
DeepSeek-V3-style routing in `lib/common.sh`), so a recipe that wins at 420m
transfers up the same family.
`TRAIN_SAMPLES` is the documented default per size but is a per-experiment
choice — override it: `TRAIN_SAMPLES=7324219 bash submit.sh ...`. See
`../../SCALING.md` for the D/N rationale and which rungs to actually run.

## Clusters

The default cluster is `alps3` (GH200) — nothing changes if you never pass
`--cluster`. To run the same size×recipe on the AMD MI300A nodes:

```bash
bash submit.sh --cluster mi300 --size 420m-moe --recipe master
# -> job 420m-moe-master-mi300, run name 420m-moe-master-lr...-mi300
CLUSTER=mi300 SIZE=420m-moe RECIPE=master bash lib/dump-args.sh   # inspect
```

A cluster file (`clusters/<name>.sh`, contract in `clusters/alps3.sh`) owns
everything machine-specific: the container EDF (`../alps3.toml` /
`../mi300.toml`), `srun --mpi` flavor, precision args, MoE dispatcher,
arch-tagged caches/packages dirs, and the sbatch flags submit.sh adds
(partition/cpus/mem/reservation — note the alps3 reservation now lives in
`clusters/alps3.sh`, not the `train.sbatch` header).

`mi300` deltas vs `alps3` (each from schlag's verified multinode MoE bring-up,
`/iopsstor/scratch/cscs/schlag/v25_6_moe_test/run_moe_mn_hooks_rocm6.sbatch`):
**bf16 only** (no FP8 blockwise → losses are NOT comparable to fp8 GH200 runs),
`alltoall` MoE dispatcher, `--no-gradient-accumulation-fusion`, ROCm v25.6
container with cxi + aws_ofi_nccl hooks, separate `cache/triton-rocm` /
`_research/packages-rocm` (first run reinstalls pip deps there). The run name
gets a `-mi300` suffix so checkpoints on the shared filesystem never collide
with an alps3 run of the same composition.

## Tracking results: the leaderboard bridge

There are two surfaces, on purpose:

- **This framework = authoring.** DRY, env-var driven, fast to iterate.
- **`../../leaderboards/<size>/runs/` = archival.** Every entry is a *frozen,
  self-contained* sbatch that reproduces bitwise via `git checkout <sha> &&
  sbatch <file>` — it must NOT depend on this framework, which changes over
  time.

`freeze.sh` bridges them: it bakes a composition into a standalone leaderboard
entry (all args pinned inline, framework no longer required to run it).

```bash
# when a run is worth tracking, freeze it into the board (next free NN- prefix):
bash freeze.sh --size 420m-moe --recipe master --slug master --board 420m-moe --out auto
# then: run it, fill in the <FILL IN> loss/W&B/rank header fields, and add a
# row to leaderboards/420m-moe/README.md (entry / parent / change — see that file).
```

The frozen file's scientific args are verified identical to the live
composition; only the env-gated wandb block differs. Promotion lineage
(`entry`/`parent`/`change`) and the ranked table live in the board README, not
here — this framework produces entries, the board curates them.

## Long runs: `--auto-requeue`

Every run already **checkpoints before the wall clock kills it**: `common.sh`
passes `--exit-duration-in-mins 595` (5 min under the 10h default), so Megatron
saves and exits cleanly instead of getting `SIGKILL`ed. Resuming is automatic —
`--save` and `--load` are the same `CKPT_DIR`, derived from the run name, so the
*next* job for the same size+recipe+sha picks up where the last one stopped.

What's *not* automatic is launching that next job. The main baseline (420m-moe,
15B tokens) fits in one allocation, so by default nothing chains. For runs that
span allocations (the dense `1.3b`/`2.7b` rungs, or a bumped `TRAIN_SAMPLES`),
add `--auto-requeue`:

```bash
bash submit.sh --size 1.3b --recipe master --auto-requeue
```

This makes each job, on a **clean** exit, submit the next one with
`--dependency=afterany` until `TRAIN_SAMPLES` is reached. Safety rails:

- **Crash ≠ requeue.** Only `srun` exit 0 (finished or hit exit-duration) chains.
  A non-zero exit stops the chain so you investigate instead of looping the queue.
- **Done detection** reads `latest_checkpointed_iteration.txt` and compares
  `iter*GBS` against `TRAIN_SAMPLES`.
- **`afterany` dependency** means at most one job runs at a time → no two jobs
  writing the same `CKPT_DIR`.
- **`MAX_REQUEUES` (default 50)** backstops a runaway chain.

## The variable contract

A **size** file sets (required):
`NUM_LAYERS HIDDEN FFN_HIDDEN NUM_HEADS NUM_KV_HEADS SEQ_LEN MBS GBS
TRAIN_SAMPLES SAVE_INTERVAL APERTUS_TRACK` and `MOE_ARGS=(...)` (empty `()` for
dense). May set: `TP PP CP EP EMBEDDING_MULTIPLIER INIT_STD EXIT_DURATION_MINS
DEFAULT_NODES DEFAULT_TIME`.

A **recipe** file sets (required): `OPTIMIZER EXP_TAG RECIPE_ARGS=(...)`. May
set: `LR MIN_LR KNOB_STR WEIGHT_DECAY ADAM_BETA1 ADAM_BETA2 CLIP_GRAD
LR_DECAY_STYLE LR_WARMUP_SAMPLES LR_WSD_DECAY_STYLE LR_WSD_DECAY_SAMPLES
EXTRA_REG_ARGS=(...)`. Env-only: `OVERRIDE_OPT_SCHED=1` adds
`--override-opt-param-scheduler` (used by `sweep-cooldowns.sh` to re-schedule a
loaded checkpoint).

A **cluster** file may set (all guarded — env/size/recipe win): `EDF_TOML
SRUN_MPI SRUN_EXTRA_ARGS=(...) MIXED_PRECISION_ARGS=(...) MOE_DISPATCHER
CLUSTER_TRAIN_ARGS=(...) ARCH_TAG CLUSTER_TAG CLUSTER_SBATCH_FLAGS=(...)`.

The full annotated contracts are the headers of `lib/common.sh` and
`clusters/alps3.sh`.

## Run name

Auto-composed as `<size>-<EXP_TAG>-<KNOB_STR>`, e.g.
`420m-moe-master-lr1e-3-mlr8e-3-elr1`. Set `KNOB_STR` in your recipe to whatever
knobs you sweep so runs are self-describing in wandb. For an axis the recipe's
`KNOB_STR` doesn't cover (e.g. token budget), set `RUN_TAG=...` in the env and it
gets appended.

The run name fully determines `CKPT_DIR`, so **two runs with the same name share
a checkpoint dir** — that's what makes auto-requeue resume work, but it also
means every point in a sweep must produce a distinct name (see below).

## Sweeps

No new files, no sweep config — a sweep is a shell loop over `submit.sh`, because
every knob is an env var and the run name is derived from those knobs.

**Sweep the LR** (`KNOB_STR` already encodes `lr/mlr/elr`, so each point gets a
distinct name + checkpoint dir automatically):

```bash
for mlr in 2e-3 4e-3 8e-3 1.6e-2; do
    MLR=$mlr bash submit.sh --size 420m-moe --recipe master
done
# -> 420m-moe-master-lr1e-3-mlr2e-3-elr1, ...-mlr4e-3-..., ...
```

**Sweep training length** with `sweep-length.sh` — token budgets in billions, one
run per point (linear decay scales to each length automatically, no cooldown to tune):

```bash
bash sweep-length.sh --size 420m-moe --recipe master --tokens "7.5 15 30"
# -> ...-elr1-t7.5b, ...-t15b, ...-t30b   (add --auto-requeue for the long ones)
```

Two-axis sweep is just nested loops (`for mlr ...; do for t ...`).

**Sweep cooldowns from one constant-LR run** with `sweep-cooldowns.sh` — the
multiple-cooldown analysis: one base run at constant LR + WSD cooldowns branched
off its intermediate checkpoints (~20% of each branch budget):

```bash
bash sweep-cooldowns.sh --size 420m-moe --recipe master --base-tokens 15 --branches "3 6 9 15"
# -> base:      ...-elr1e-3-const-t15b                          (constant LR, auto-requeue)
# -> cooldowns: ...-const-t15b-d3b-cd0.6b, ...-d6b-cd1.2b, ...  (one per branch)
```

Branch points snap to the size's `SAVE_INTERVAL` (3B tokens at 420m-moe). Each
cooldown's `CKPT_DIR` is pre-seeded with a symlink to the base checkpoint, then
runs as a normal job: optimizer state loads, and `OVERRIDE_OPT_SCHED=1` swaps in
the branch-local WSD schedule. Run it again later (or use `--watch`) to submit
branches whose checkpoint doesn't exist yet; cooldowns must use the same `--nodes`
as the base run (per-rank optimizer shards). `--cooldown-frac`/`--cooldown-tokens`
override the cooldown length.

## Inspect without launching

```bash
SIZE=420m-moe RECIPE=master bash lib/dump-args.sh        # print the composed arg list
bash submit.sh --size 420m-moe --recipe master --dry-run # same, with node/time summary
```

## Verify equivalence to a legacy sbatch

The framework reproduces the old hand-written files exactly. To prove it for any
legacy file:

```bash
bash verify-equivalence.sh 420m-moe master ../../legacy/transformer-pp-350m-master.sbatch
# ✓ EQUIVALENT: 420m-moe/master matches transformer-pp-350m-master.sbatch
```

This is how a legacy sbatch is retired: build the size+recipe, confirm `✓`,
delete the old file.

## Interactive runs / debugging a recipe

`debug.sh` is the interactive twin of `submit.sh`: same `--size`/`--recipe`, but
it runs inline via `interactive-run.sh` (torchrun, no SLURM) for a few iters.
Run it from inside an interactive allocation (data defaults to the swissai blend;
set `MEGATRON_DATA_PATH` to a single prefix to override):

```bash
bash debug.sh --size 420m-moe --recipe my-idea               # 4 ranks, 20 iters
bash debug.sh --size 270m-moe --recipe my-idea --nproc 2 --iters 10
bash debug.sh --size 420m-moe --recipe my-idea -- --lr 1e-4   # flags after -- override
bash debug.sh --size 420m-moe --recipe my-idea --dry-run      # show args, don't launch
```

Defaults: 4 ranks, 20 iters, 5-min cap, deps installed once (`SKIP_DEPS=1` to
skip later). See `../interactive-run.sh` for how to grab the allocation.
