#!/bin/bash
#
# submit.sh — convenience wrapper around `sbatch train.sbatch`.
#
# Reads DEFAULT_NODES / DEFAULT_TIME from the size file so you don't have to
# remember them, then submits. Any knob the recipe reads from the environment
# (LR, MLR, ...) is passed straight through.
#
#   bash submit.sh --size 420m-moe --recipe master
#   bash submit.sh --size 420m-moe --recipe master --nodes 4
#   MLR=1e-2 bash submit.sh --size 420m-moe --recipe master       # sweep a knob
#   bash submit.sh --size 1.3b --recipe master --auto-requeue     # chain jobs until done
#   bash submit.sh --size 420m-moe --recipe master --dry-run      # print args, no submit
#
set -euo pipefail

FRAMEWORK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)

NODES=""; TIME=""; DRY=""; AUTO_RQ=""
while [ $# -gt 0 ]; do
    case "$1" in
        --size)   SIZE="$2"; shift 2 ;;
        --recipe) RECIPE="$2"; shift 2 ;;
        --nodes)  NODES="$2"; shift 2 ;;
        --time)   TIME="$2"; shift 2 ;;
        --auto-requeue) AUTO_RQ=1; shift ;;
        --dry-run) DRY=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

: "${SIZE:?usage: submit.sh --size <size> --recipe <recipe> [--nodes N] [--time T] [--auto-requeue] [--dry-run]}"
: "${RECIPE:?usage: submit.sh --size <size> --recipe <recipe> [--nodes N] [--time T] [--auto-requeue] [--dry-run]}"

SIZE_FILE="$FRAMEWORK_DIR/sizes/$SIZE.sh"
[ -f "$SIZE_FILE" ] || { echo "no such size: $SIZE (see $FRAMEWORK_DIR/sizes/)" >&2; exit 1; }

# Pull DEFAULT_NODES / DEFAULT_TIME out of the size file without running the
# whole thing.
DEFAULT_NODES=$(grep -E '^DEFAULT_NODES=' "$SIZE_FILE" | head -1 | cut -d= -f2)
DEFAULT_TIME=$(grep -E '^DEFAULT_TIME=' "$SIZE_FILE" | head -1 | cut -d= -f2)
NODES=${NODES:-${DEFAULT_NODES:-2}}
TIME=${TIME:-${DEFAULT_TIME:-10:00:00}}

export SIZE RECIPE

if [ -n "$DRY" ]; then
    echo ">>> DRY RUN: SIZE=$SIZE RECIPE=$RECIPE NODES=$NODES TIME=$TIME"
    echo ">>> composed MEGATRON_ARGS:"
    SIZE="$SIZE" RECIPE="$RECIPE" bash "$FRAMEWORK_DIR/lib/dump-args.sh" | sed 's/^/  /'
    exit 0
fi

JOB_NAME=${JOB_NAME:-${SIZE}-${RECIPE//\//-}}   # subfolder recipes (wsd/muon) -> no slash in job name

# --auto-requeue: chain dependent jobs (same CKPT_DIR -> resume) until the run
# reaches TRAIN_SAMPLES. REQUEUE_TIME carries this run's --time so each chained
# allocation gets the same wall-clock; the chaining logic lives in common.sh.
EXPORT="ALL,SIZE=$SIZE,RECIPE=$RECIPE,FRAMEWORK_DIR=$FRAMEWORK_DIR"
if [ -n "$AUTO_RQ" ]; then
    EXPORT="$EXPORT,AUTO_REQUEUE=1,REQUEUE_TIME=$TIME"
    echo ">>> auto-requeue ON: will chain dependent jobs until TRAIN_SAMPLES is reached (clean exits only)"
fi

echo ">>> sbatch --nodes=$NODES --time=$TIME --job-name=$JOB_NAME  (SIZE=$SIZE RECIPE=$RECIPE)"
exec sbatch \
    --nodes="$NODES" \
    --time="$TIME" \
    --job-name="$JOB_NAME" \
    --export="$EXPORT" \
    "$FRAMEWORK_DIR/train.sbatch"
