# 810m-moe Leaderboard

Best runs at **~810M active / ~6.7B total params (MoE 64e-tk2-sh1), ~30B
tokens, 4 nodes** (`--size 810m-moe`). No entries yet.

- **Format & how to submit:** see the [leaderboards README](../README.md); the
  [350m-ablation board](../350m-ablation/README.md) is the worked example
  (legacy dense track, same table format).
- **Produce a candidate:**
  `bash ../../launch/framework/submit.sh --size 810m-moe --recipe <recipe>`,
  then freeze the winner with
  `../../launch/framework/freeze.sh --size 810m-moe --recipe <recipe> --board 810m-moe --out auto`
  and add a row here (promotion steps in the leaderboards README).
