#!/bin/bash
#
# verify-equivalence.sh — prove a (SIZE, RECIPE) composition produces the SAME
# Megatron arg list as a legacy hand-written sbatch.
#
#   bash verify-equivalence.sh <SIZE> <RECIPE> <legacy.sbatch>
#
# Run-name-dependent tokens (--save/--load/--tensorboard-dir/--wandb-exp-name)
# are normalized away; everything else must match exactly.
#
set -euo pipefail

SIZE=$1; RECIPE=$2; LEGACY=$3
FRAMEWORK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
REPO_ROOT=$(cd "$FRAMEWORK_DIR/../../.." && pwd)

[ -f "$LEGACY" ] || { echo "legacy sbatch not found: $LEGACY" >&2; exit 1; }
LEGACY=$(cd "$(dirname "$LEGACY")" && pwd)/$(basename "$LEGACY")   # absolutize before any cd

# Blank the value following each run-name-dependent flag so the diff focuses on
# substantive args, sort, so array-order differences between legacy files and
# the framework don't create false diffs.
normalize() {
    awk '
        prev ~ /^--(save|load|tensorboard-dir|wandb-exp-name|wandb-save-dir|data-path|data-cache-path)$/ { $0="<NORM>" }
        { print; prev=$0 }
    ' | sort
}

NEW_ARGS=$(SIZE="$SIZE" RECIPE="$RECIPE" MEGATRON_DATA_PATH=/tmp/DUMMY \
    bash "$FRAMEWORK_DIR/lib/dump-args.sh" | normalize)

# Capture legacy MEGATRON_ARGS by sourcing it with SLURM/mkdir stubbed.
LEGACY_ARGS=$(
    cd "$REPO_ROOT"
    srun() { :; }; scontrol() { hostname; }; mkdir() { :; }
    export -f srun scontrol mkdir
    export SLURM_SUBMIT_DIR=$REPO_ROOT MEGATRON_DATA_PATH=/tmp/DUMMY
    export SLURM_CPUS_PER_TASK=72 SLURM_NTASKS=4 SLURM_NNODES=1
    export SLURM_JOB_NODELIST=$(hostname) SLURM_JOB_ID=interactive
    source "$LEGACY" >/dev/null 2>&1 || true
    printf '%s\n' "${MEGATRON_ARGS[@]}"
)
LEGACY_ARGS=$(printf '%s\n' "$LEGACY_ARGS" | normalize)

echo "=== diff (legacy '<' vs composed '>'); empty = identical ==="
if diff <(printf '%s\n' "$LEGACY_ARGS") <(printf '%s\n' "$NEW_ARGS"); then
    echo "✓ EQUIVALENT: $SIZE/$RECIPE matches $(basename "$LEGACY")"
else
    echo "✗ DIFFERENCES above. Each '<' is in the legacy file only; each '>' is in the framework only."
    exit 1
fi
