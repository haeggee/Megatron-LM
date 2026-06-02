#!/bin/bash
#
# debug.sh — run a (SIZE, RECIPE) composition INTERACTIVELY for a few iters.
# The interactive twin of submit.sh: same --size/--recipe, but runs inline via
# interactive-run.sh (torchrun, no SLURM srun wrapper) with short-run defaults.
#
# Pre-req: you're already inside an interactive allocation on a GH200 node,
# inside the alps3 container. Data defaults to the swissai blend under DATA_ROOT;
# set MEGATRON_DATA_PATH to a single prefix to override. (See the header of
# ../interactive-run.sh for how to grab an allocation.)
#
#   bash debug.sh --size 420m-moe --recipe my-idea
#   bash debug.sh --size 270m-moe --recipe my-idea --nproc 2 --iters 10
#   bash debug.sh --size 420m-moe --recipe my-idea -- --lr 1e-4   # extra flags override
#   bash debug.sh --size 420m-moe --recipe my-idea --dry-run      # print args, don't run
#
# Defaults: 4 ranks, 20 iters, 5-min exit cap, deps installed once (SKIP_DEPS=1
# to skip on later runs).
#
set -euo pipefail

FRAMEWORK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
INTERACTIVE_RUN="$FRAMEWORK_DIR/../interactive-run.sh"

NPROC=${NPROC_PER_NODE:-4}; ITERS=20; DRY=""; EXTRA=()
while [ $# -gt 0 ]; do
    case "$1" in
        --size)    SIZE="$2"; shift 2 ;;
        --recipe)  RECIPE="$2"; shift 2 ;;
        --nproc)   NPROC="$2"; shift 2 ;;
        --iters)   ITERS="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        --)        shift; EXTRA=("$@"); break ;;
        *) echo "unknown arg: $1 (use -- to pass extra megatron flags)" >&2; exit 1 ;;
    esac
done
: "${SIZE:?usage: debug.sh --size <size> --recipe <recipe> [--nproc N] [--iters N] [--dry-run] [-- extra flags]}"
: "${RECIPE:?usage: debug.sh --size <size> --recipe <recipe> [--nproc N] [--iters N] [--dry-run] [-- extra flags]}"

# The framework trains in SAMPLES (--train-samples), and Megatron forbids mixing
# that with --train-iters. So a short debug run is just a small TRAIN_SAMPLES
# override (iters * GBS), keeping the run sample-based and identical in shape to
# a real launch — only shorter (LR also decays over this short budget).
SIZE_FILE="$FRAMEWORK_DIR/sizes/$SIZE.sh"
[ -f "$SIZE_FILE" ] || { echo "no such size: $SIZE (see $FRAMEWORK_DIR/sizes/)" >&2; exit 1; }
GBS=$(grep -E '^GBS=' "$SIZE_FILE" | head -1 | cut -d= -f2)
: "${GBS:?could not read GBS from $SIZE_FILE}"
TRAIN_SAMPLES=$(( ITERS * GBS ))

export SIZE RECIPE TRAIN_SAMPLES
# Debug runs land in a separate wandb board ("debug-<base>") so they don't
# clutter the real one. Override: WANDB_PROJECT_PREFIX= (empty) for the base
# board, or WANDB_PROJECT=... for an explicit name.
export WANDB_PROJECT_PREFIX="${WANDB_PROJECT_PREFIX:-debug-}"

if [ -n "$DRY" ]; then
    echo ">>> DRY RUN: SIZE=$SIZE RECIPE=$RECIPE  wandb-project=${WANDB_PROJECT:-${WANDB_PROJECT_PREFIX}megatron-lm-research-baseline}  (composed args below; not launching)"
    bash "$FRAMEWORK_DIR/lib/dump-args.sh" | sed 's/^/  /'
    echo ">>> would run: NPROC_PER_NODE=$NPROC TRAIN_SAMPLES=$TRAIN_SAMPLES ($ITERS iters * GBS $GBS) interactive-run.sh framework/train.sbatch --exit-duration-in-mins 5 ${EXTRA[*]}"
    exit 0
fi

echo ">>> debug: $ITERS iters -> TRAIN_SAMPLES=$TRAIN_SAMPLES (GBS=$GBS)"
# Short-run debug overrides go LAST so they win (argparse last-value-wins).
exec env NPROC_PER_NODE="$NPROC" SIZE="$SIZE" RECIPE="$RECIPE" TRAIN_SAMPLES="$TRAIN_SAMPLES" \
    bash "$INTERACTIVE_RUN" "$FRAMEWORK_DIR/train.sbatch" \
        --exit-duration-in-mins 5 \
        "${EXTRA[@]}"
