# 1.3b Leaderboard

Best runs at **1.3B params (dense), ~100B tokens, 8 nodes** (`--size 1.3b`).
No entries yet.

- **Format & how to submit:** see the [leaderboards README](../README.md); the
  [350m-ablation board](../350m-ablation/README.md) is the worked example.
- **Produce a candidate:**
  `bash ../../launch/framework/submit.sh --size 1.3b --recipe <recipe> --auto-requeue`
  (long run — `--auto-requeue` chains allocations), then freeze the winner with
  `../../launch/framework/freeze.sh --size 1.3b --recipe <recipe> --board 1.3b --out auto`
  and add a row here (promotion steps in the leaderboards README).
