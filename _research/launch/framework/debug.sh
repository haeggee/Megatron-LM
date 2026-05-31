#!/bin/bash
#
# debug.sh — run a (SIZE, RECIPE) composition INTERACTIVELY for a few iters.
# The interactive twin of submit.sh: same --size/--recipe, but runs inline via
# interactive-run.sh (torchrun, no SLURM srun wrapper) with short-run defaults.
#
# Pre-req: you're already inside an interactive allocation on a GH200 node,
# inside the alps3 container, with MEGATRON_DATA_PATH set. (See the header of
# ../interactive-run.sh for how to grab one.)
#
#   bash debug.sh --size 350m-moe --recipe my-idea
#   bash debug.sh --size 175m-moe --recipe my-idea --nproc 2 --iters 10
#   bash debug.sh --size 350m-moe --recipe my-idea -- --lr 1e-4   # extra flags override
#   bash debug.sh --size 350m-moe --recipe my-idea --dry-run      # print args, don't run
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

export SIZE RECIPE

if [ -n "$DRY" ]; then
    echo ">>> DRY RUN: SIZE=$SIZE RECIPE=$RECIPE  (composed args below; not launching)"
    bash "$FRAMEWORK_DIR/lib/dump-args.sh" | sed 's/^/  /'
    echo ">>> would run: NPROC_PER_NODE=$NPROC interactive-run.sh framework/train.sbatch --train-iters $ITERS --exit-duration-in-mins 5 ${EXTRA[*]}"
    exit 0
fi

# Short-run debug overrides go LAST so they win (argparse last-value-wins).
exec env NPROC_PER_NODE="$NPROC" SIZE="$SIZE" RECIPE="$RECIPE" \
    bash "$INTERACTIVE_RUN" "$FRAMEWORK_DIR/train.sbatch" \
        --train-iters "$ITERS" \
        --exit-duration-in-mins 5 \
        "${EXTRA[@]}"
