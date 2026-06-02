# 2.7b Leaderboard

Best runs at **2.7B params (dense), ~300B tokens, 16 nodes** (`--size 2.7b`).
No entries yet.

- **Format & how to submit:** see the [leaderboards README](../README.md); the
  [350m-ablation board](../350m-ablation/README.md) is the worked example.
- **Produce a candidate:**
  `bash ../../launch/framework/submit.sh --size 2.7b --recipe <recipe> --auto-requeue`
  (long run — `--auto-requeue` chains allocations), then freeze the winner with
  `../../launch/framework/freeze.sh --size 2.7b --recipe <recipe> --board 2.7b --out auto`
  and add a row here (promotion steps in the leaderboards README).
