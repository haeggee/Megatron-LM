#!/bin/bash
#
# sweep-lr.sh — submit one run per learning-rate point.
#
# A thin loop over submit.sh: for each LR it exports LR=<val> and hands off.
# The recipe folds LR into KNOB_STR (default `lr${LR}`), so every point gets its
# own EXP_NAME -> own checkpoint dir + wandb run with no extra bookkeeping.
# Everything else — nodes/time defaults, --auto-requeue, --dry-run — is passed
# straight through to submit.sh.
#
# For Muon-family recipes, LR is the *matrix* LR; the scalar LR stays at the
# recipe default (override per-point with SCALAR_LR=... if you need to).
#
#   bash sweep-lr.sh --size 270m-moe --recipe adamw --lrs "5e-4 7.1e-4 1.4e-3 2e-3"
#   bash sweep-lr.sh --size 270m-moe --recipe muon  --lrs "5e-3 7.1e-3 1.4e-2 2e-2" --auto-requeue
#   bash sweep-lr.sh --size 270m-moe --recipe adamw --lrs "1e-3 2e-3" --dry-run
#
set -euo pipefail

FRAMEWORK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)

LRS=""; PASS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --size)    SIZE="$2"; shift 2 ;;
        --recipe)  RECIPE="$2"; shift 2 ;;
        --lrs)     LRS="$2"; shift 2 ;;        # space-separated LR values
        # passthrough to submit.sh:
        --nodes)        PASS+=(--nodes "$2"); shift 2 ;;
        --time)         PASS+=(--time "$2"); shift 2 ;;
        --auto-requeue) PASS+=(--auto-requeue); shift ;;
        --dry-run)      PASS+=(--dry-run); shift ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

: "${SIZE:?usage: sweep-lr.sh --size <size> --recipe <recipe> --lrs \"5e-4 1e-3 2e-3\" [--nodes N] [--time T] [--auto-requeue] [--dry-run]}"
: "${RECIPE:?usage: sweep-lr.sh --size <size> --recipe <recipe> --lrs \"...\" [...]}"
: "${LRS:?give --lrs \"5e-4 1e-3 2e-3\" (space-separated)}"

[ -f "$FRAMEWORK_DIR/sizes/$SIZE.sh" ] || {
    echo "no such size: $SIZE (see $FRAMEWORK_DIR/sizes/)" >&2; exit 1; }

echo ">>> LR sweep: size=$SIZE recipe=$RECIPE  (LRs: $LRS)"
for lr in $LRS; do
    echo ">>> --- LR=$lr ---"
    LR="$lr" bash "$FRAMEWORK_DIR/submit.sh" --size "$SIZE" --recipe "$RECIPE" "${PASS[@]}"
done
