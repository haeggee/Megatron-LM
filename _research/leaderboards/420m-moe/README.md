# 420m-moe Leaderboard — the main baseline

Best runs at **~420M active / ~2.5B total params (MoE 64e-tk2-sh1), ~15B
tokens, 2 nodes** (`--size 420m-moe`). No entries yet.

- **Format & how to submit:** see the [leaderboards README](../README.md); the
  [350m-ablation board](../350m-ablation/README.md) is the worked example
  (legacy dense track, same table format).
- **Produce a candidate:**
  `bash ../../launch/framework/submit.sh --size 420m-moe --recipe <recipe>`,
  then freeze the winner with
  `../../launch/framework/freeze.sh --size 420m-moe --recipe <recipe> --board 420m-moe --out auto`
  and add a row here (promotion steps in the leaderboards README).
