# 1.5b-moe Leaderboard

Best runs at **~1.5B active / ~14.6B total params (MoE 64e-tk2-sh1), ~30B
tokens, 4 nodes** (`--size 1.5b-moe`). The confirmation rung for promoting a
420m-moe winner. No entries yet.

- **Format & how to submit:** see the [leaderboards README](../README.md); the
  [350m-ablation board](../350m-ablation/README.md) is the worked example
  (legacy dense track, same table format).
- **Produce a candidate:**
  `bash ../../launch/framework/submit.sh --size 1.5b-moe --recipe <recipe>`,
  then freeze the winner with
  `../../launch/framework/freeze.sh --size 1.5b-moe --recipe <recipe> --board 1.5b-moe --out auto`
  and add a row here (promotion steps in the leaderboards README).
