# Launch framework

A run is **one model size × one recipe**. You compose them instead of copying a
300-line sbatch. You only ever write the recipe file.

### TL;DR — add a new optimizer/baseline

```bash
cd _research/launch/framework
cp recipes/_template.sh recipes/my-idea.sh   # set OPTIMIZER, EXP_TAG, RECIPE_ARGS
bash submit.sh --size 350m-moe --recipe my-idea --dry-run   # inspect composed args
bash submit.sh --size 350m-moe --recipe my-idea             # launch
# sweep a knob with no new file (flows into the flag AND the run name):
MLR=1e-2 bash submit.sh --size 350m-moe --recipe my-idea
# scale the same recipe up later:
bash submit.sh --size 1.5b-moe --recipe my-idea --nodes 4
```

### TL;DR — debug a recipe interactively

```bash
# inside an alps3 interactive allocation, with MEGATRON_DATA_PATH set:
bash debug.sh --size 175m-moe --recipe my-idea --nproc 2 --iters 10   # fast inline run
bash debug.sh --size 350m-moe --recipe my-idea -- --lr 1e-4           # flags after -- override
bash debug.sh --size 350m-moe --recipe my-idea --dry-run              # show args, don't launch
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

`train.sbatch` just sources the three in order (`size` → `recipe` → `common`)
and hands off. `common.sh` fills in every default the fragments didn't set.

## Add a new optimizer / baseline

```bash
cp recipes/_template.sh recipes/my-idea.sh
$EDITOR recipes/my-idea.sh          # set OPTIMIZER, EXP_TAG, RECIPE_ARGS, LR...
bash submit.sh --size 350m-moe --recipe my-idea --dry-run   # inspect composed args
bash submit.sh --size 350m-moe --recipe my-idea             # launch
```

A recipe is usually 10–20 lines. The only required fields are `OPTIMIZER`,
`EXP_TAG`, and `RECIPE_ARGS=(...)`. Everything else (`LR`, `WEIGHT_DECAY`,
`ADAM_BETA1/2`, `CLIP_GRAD`, the WSD schedule, init std, embedding multiplier)
has a default in `common.sh` you override only when your idea needs it.

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
| `175m-moe` | 175M active | 16e-tk1-sh1 | ~12B | 1 | cheap LR probe |
| `350m-moe` | 350M active | 16e-tk1-sh1 | 15B | 2 | **the main baseline** |
| `760m-moe` | 760M active | 16e-tk1-sh1 | 30B | 4 | scale-up rung |
| `1.5b-moe` | 1.5B active | 16e-tk1-sh1 | 30B | 4 | promotion-confirm rung |
| `1.3b` | 1.3B | dense | 100B | 8 | dense baseline |
| `2.7b` | 2.7B | dense | 300B | 16 | dense baseline |

The MoE ladder (175m → 1.5b) shares one shape (16 routed top-1 + 1 shared,
`moe-ffn = ffn/2`), so a recipe that wins at 350m transfers up the same family.
`TRAIN_SAMPLES` is the documented default per size but is a per-experiment
choice — override it: `TRAIN_SAMPLES=7324219 bash submit.sh ...`. See
`../../SCALING.md` for the D/N rationale and which rungs to actually run.

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
bash freeze.sh --size 350m-moe --recipe master --slug master --board 350m --out auto
# then: run it, fill in the <FILL IN> loss/W&B/rank header fields, and add a
# row to leaderboards/350m/README.md (entry / parent / change — see that file).
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

What's *not* automatic is launching that next job. The main baseline (350m-moe,
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
EXTRA_REG_ARGS=(...)`.

The full annotated contract is the header of `lib/common.sh`.

## Run name

Auto-composed as `<size>-<EXP_TAG>-<KNOB_STR>`, e.g.
`350m-moe-master-lr1e-3-mlr8e-3-elr1`. Set `KNOB_STR` in your recipe to whatever
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
    MLR=$mlr bash submit.sh --size 350m-moe --recipe master
done
# -> 350m-moe-master-lr1e-3-mlr2e-3-elr1, ...-mlr4e-3-..., ...
```

**Sweep training length** with `sweep-length.sh` — token budgets in billions, one
run per point (linear decay scales to each length automatically, no cooldown to tune):

```bash
bash sweep-length.sh --size 350m-moe --recipe master --tokens "7.5 15 30"
# -> ...-elr1-t7.5b, ...-t15b, ...-t30b   (add --auto-requeue for the long ones)
```

Two-axis sweep is just nested loops (`for mlr ...; do for t ...`).

## Inspect without launching

```bash
SIZE=350m-moe RECIPE=master bash lib/dump-args.sh        # print the composed arg list
bash submit.sh --size 350m-moe --recipe master --dry-run # same, with node/time summary
```

## Verify equivalence to a legacy sbatch

The framework reproduces the old hand-written files exactly. To prove it for any
legacy file:

```bash
bash verify-equivalence.sh 350m-moe master ../transformer-pp-350m-master.sbatch
# ✓ EQUIVALENT: 350m-moe/master matches transformer-pp-350m-master.sbatch
```

This is how a legacy sbatch is retired: build the size+recipe, confirm `✓`,
delete the old file.

## Interactive runs / debugging a recipe

`debug.sh` is the interactive twin of `submit.sh`: same `--size`/`--recipe`, but
it runs inline via `interactive-run.sh` (torchrun, no SLURM) for a few iters.
Run it from inside an interactive allocation with `MEGATRON_DATA_PATH` set:

```bash
bash debug.sh --size 350m-moe --recipe my-idea               # 4 ranks, 20 iters
bash debug.sh --size 175m-moe --recipe my-idea --nproc 2 --iters 10
bash debug.sh --size 350m-moe --recipe my-idea -- --lr 1e-4   # flags after -- override
bash debug.sh --size 350m-moe --recipe my-idea --dry-run      # show args, don't launch
```

Defaults: 4 ranks, 20 iters, 5-min cap, deps installed once (`SKIP_DEPS=1` to
skip later). See `../interactive-run.sh` for how to grab the allocation.
