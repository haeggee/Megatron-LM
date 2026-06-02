# Leaderboards

> Detail reference for result tracking. The repo [README](../../README.md) has
> the big picture; runs are authored with the
> [launch framework](../launch/framework/README.md) and promoted here.

Per-size ranked lists of training runs on this fork of Megatron-LM. Each
entry points to a self-contained, re-runnable `.sbatch` and a W&B run.

## Structure

```
leaderboards/
  <size>/
    README.md       # ranked table + wandb links + one-line description
    runs/
      01-*.sbatch   # self-contained, no env-var branching, all hparams pinned
      02-*.sbatch
      ...
```

A self-contained sbatch runs with `sbatch runs/<NN>-<name>.sbatch` — no
env-var branching, no harness dependencies. The header of each file
records the rank at entry time, the git sha the run was executed at,
the W&B URL, and the final + min training loss.

To reproduce an entry bitwise: `git checkout <sha>` then
`sbatch leaderboards/<size>/runs/NN-<name>.sbatch`.

## Promoting a run to a leaderboard

When a run wins (or is otherwise worth tracking), add it as an entry:

1. Copy the executed sbatch into `leaderboards/<size>/runs/`. Pick the
   next free numeric prefix (`NN-`); the rest of the filename is the
   stable **slug** for that entry (e.g. `08-aurora-lr1e-2.sbatch` has
   slug `aurora-lr1e-2`).
2. Pin every hparam in the file: no env-var branching, no harness
   dependencies. The header should carry rank, sha, W&B URL, final +
   min loss.
3. Add a row to the leaderboard's `README.md` table with these stable
   columns:
   - `entry`: the slug from step 1.
   - `parent`: the slug of the entry this run builds on, or empty if
     it's a new root.
   - `change`: a one-line delta vs the parent (e.g. `LR 3.6e-4 -> 6e-4`,
     `add --qk-layernorm (RMSNorm on Q, K)`, `FP8 e4m3 blockwise
     (DeepSeek-V3)`). Roots use this column to describe the recipe
     family.
4. Reorder the table by `min loss` (or whatever metric matters for
   that leaderboard) and update the `rank` column. Rank is unstable
   over time; the slug + parent + change triple is the durable record
   of lineage.

## Sizes

The active boards mirror the MoE ladder in
[`launch/framework/sizes/`](../launch/framework/sizes/): 64 routed experts
top-2 + 1 shared, ~6% non-embedding sparsity at every rung (see the
[framework README](../launch/framework/README.md)).

| leaderboard | params (active / total) | tokens | nodes | purpose |
| --- | --- | ---: | ---: | --- |
| [`420m-moe`](420m-moe/README.md) | 0.42B / 2.5B | 15B | 2 | **the main baseline** |
| [`810m-moe`](810m-moe/README.md) | 0.81B / 6.7B | 30B | 4 | scale-up rung |
| [`1.5b-moe`](1.5b-moe/README.md) | 1.5B / 14.6B | 30B | 4 | promotion-confirm rung |
| [`1.3b`](1.3b/README.md) | 1.3B (dense) | 100B | 8 | dense baseline |
| [`2.7b`](2.7b/README.md) | 2.7B (dense) | 300B | 16 | dense baseline |
| [`350m-ablation`](350m-ablation/README.md) | 0.36B (dense) | 1B | 1 | legacy dense ablation track (~30 min/run), the worked example |

(`270m-moe` is an LR-probe rung, not a board — probe results stay in W&B.)

## Related folders

- `_research/launch/framework/` — the **authoring** surface. Compose a run as
  one `--size` × one `--recipe` instead of copying a 300-line sbatch; add a new
  optimizer/baseline idea as a ~15-line recipe file. See its `README.md`.
- `_research/launch/*.sbatch` — the older hand-written, fully-pinned launch
  files (being superseded by the framework). Still valid to run directly.

### Promoting from the framework

When a framework run wins, `freeze.sh` bakes it into a frozen, self-contained
entry here (all hparams pinned inline, no framework dependency — so `git
checkout <sha> && sbatch` still reproduces it bitwise):

```bash
bash _research/launch/framework/freeze.sh \
    --size 420m-moe --recipe <recipe> --slug <slug> --board 420m-moe --out auto
```

Then fill in the run's loss / W&B / rank in the generated header and add a table
row per the "Promoting a run" steps above. A frozen entry never sources the
framework, by design — that is what keeps old entries reproducible after the
framework changes.
