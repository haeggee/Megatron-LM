#!/bin/bash
#
# freeze.sh — bake a (SIZE, RECIPE [+ env knobs]) composition into a
# self-contained, frozen sbatch suitable for promotion to a leaderboard.
#
# WHY: the framework is the AUTHORING surface (DRY, env-var driven). The
# leaderboard is the ARCHIVAL surface: every entry must reproduce bitwise via
# `git checkout <sha> && sbatch <file>`, independent of the framework, which may
# change. So a leaderboard entry must NOT source the framework — it pins every
# arg inline. This script renders that frozen file for you.
#
#   bash freeze.sh --size 350m-moe --recipe master --slug master --board 350m \
#       --rank new --out auto
#   MLR=1e-2 bash freeze.sh --size 350m-moe --recipe master --slug master-mlr1e-2 --out -
#
# --out auto  -> writes leaderboards/<board>/runs/NN-<slug>.sbatch (next free NN)
# --out -     -> prints to stdout
# --out FILE  -> writes FILE
#
# After the run finishes, fill in the FILL-IN header fields (final/min loss,
# W&B URL, rank) and add a row to leaderboards/<board>/README.md — see that
# file's "Promoting a run" section.
#
set -euo pipefail

FRAMEWORK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
REPO_ROOT=$(cd "$FRAMEWORK_DIR/../../.." && pwd)

SLUG=""; BOARD=""; RANK="new"; OUT="-"
while [ $# -gt 0 ]; do
    case "$1" in
        --size)   SIZE="$2"; shift 2 ;;
        --recipe) RECIPE="$2"; shift 2 ;;
        --slug)   SLUG="$2"; shift 2 ;;
        --board)  BOARD="$2"; shift 2 ;;
        --rank)   RANK="$2"; shift 2 ;;
        --out)    OUT="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done
: "${SIZE:?--size required}"
: "${RECIPE:?--recipe required}"
SLUG=${SLUG:-$RECIPE}

SHA=$(git -C "$REPO_ROOT" rev-parse --short HEAD)

# Resolve the composition's metadata (EXP_TAG/KNOB_STR/OPTIMIZER/nodes/time)
# by sourcing the fragments in a throwaway shell.
eval "$(
    cd "$REPO_ROOT"
    srun() { :; }; scontrol() { hostname; }
    : "${MEGATRON_DATA_PATH:=/tmp/DUMMY}"; export MEGATRON_DATA_PATH SIZE RECIPE
    export DRY_RUN=1 SLURM_SUBMIT_DIR=$REPO_ROOT
    source "$FRAMEWORK_DIR/sizes/$SIZE.sh"   >/dev/null 2>&1
    source "$FRAMEWORK_DIR/recipes/$RECIPE.sh" >/dev/null 2>&1
    printf 'META_OPT=%q\nMETA_TAG=%q\nMETA_KNOB=%q\nMETA_NODES=%q\nMETA_TIME=%q\nMETA_TRACK=%q\n' \
        "$OPTIMIZER" "$EXP_TAG" "${KNOB_STR:-lr${LR:-}}" \
        "${DEFAULT_NODES:-2}" "${DEFAULT_TIME:-10:00:00}" "${APERTUS_TRACK:-$SIZE}"
)"

FROZEN_EXP_NAME="${SIZE}-${SLUG}-${SHA}"

# Templatize the volatile path values back to shell variables so the frozen
# file derives them at run time (rather than pinning a stale absolute path).
templatize_args() {
    SIZE="$SIZE" RECIPE="$RECIPE" MEGATRON_DATA_PATH=/tmp/DUMMY \
        bash "$FRAMEWORK_DIR/lib/dump-args.sh" 2>/dev/null | awk '
        function emit(v) { printf "    %s\n", v }
        {
            if (prev=="--save" || prev=="--load")        { emit("\"$CKPT_DIR\""); prev=$0; next }
            if (prev=="--tensorboard-dir")               { emit("\"$WORKDIR/_research/results/runs/tb-${SLURM_JOB_ID:-interactive}\""); prev=$0; next }
            if (prev=="--data-path")                     { emit("\"$MEGATRON_DATA_PATH\""); prev=$0; next }
            if (prev=="--data-cache-path")               { emit("\"$DATASET_CACHE_DIR\""); prev=$0; next }
            if (prev=="--vocab-file")                    { emit("\"$WORKDIR/_research/data/gpt2-vocab.json\""); prev=$0; next }
            if (prev=="--merge-file")                    { emit("\"$WORKDIR/_research/data/gpt2-merges.txt\""); prev=$0; next }
            emit($0); prev=$0
        }'
}

JOB_NAME="${BOARD:-$SIZE}-${SLUG}"

render() {
cat <<HEADER
#!/bin/bash
#
# ${BOARD:-$SIZE} leaderboard — rank ${RANK}
# Size: ${SIZE}   Recipe: ${RECIPE}   Knobs: ${META_KNOB}
# Optimizer: ${META_OPT}
# Final loss: <FILL IN>   Min loss: <FILL IN>
# W&B: <FILL IN>
# Executed at git sha: ${SHA}
#
# Frozen from the launch framework with:
#   SIZE=${SIZE} RECIPE=${RECIPE} bash _research/launch/framework/submit.sh --size ${SIZE} --recipe ${RECIPE}
# To reproduce bitwise:
#   git checkout ${SHA}
#   sbatch <path-to-this-file>
#
#SBATCH --job-name=${JOB_NAME}
#SBATCH --time=${META_TIME}
#SBATCH --output=/iopsstor/scratch/cscs/%u/megatron-lm-research-baseline/_research/results/runs/%x-%j.log
#SBATCH --error=/iopsstor/scratch/cscs/%u/megatron-lm-research-baseline/_research/results/runs/%x-%j.log
#SBATCH --nodes=${META_NODES}
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --no-requeue
HEADER

# Self-contained scaffolding (quoted heredoc: \$VARS stay literal in output).
cat <<'SCAFFOLD'

set -euo pipefail
: "${MEGATRON_DATA_PATH:?please set MEGATRON_DATA_PATH to the Megatron-binary dataset prefix (without .bin/.idx)}"
echo "START TIME: $(date)"
export APERTUS_PHASE_SBATCH_START=$(date +%s.%N)

REPO_DIR=/iopsstor/scratch/cscs/$USER/megatron-lm-research-baseline
WORKDIR=${SLURM_SUBMIT_DIR:-$REPO_DIR}
cd "$WORKDIR"
PACKAGE_DIR=$REPO_DIR/_research/packages
DATASET_CACHE_DIR=$REPO_DIR/cache/dataset
TMP_CACHE_DIR=$REPO_DIR/cache/tmp

export PYTHONPATH=$WORKDIR:$PACKAGE_DIR:$WORKDIR${PYTHONPATH:+:$PYTHONPATH}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_CPP_LOG_LEVEL=WARNING
export TRITON_CACHE_DIR=$REPO_DIR/cache/triton
export TORCHINDUCTOR_CACHE_DIR=$REPO_DIR/cache/inductor
export TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER=0
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-72}
export TMPDIR=$TMP_CACHE_DIR
mkdir -p "$DATASET_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TMP_CACHE_DIR" "$PACKAGE_DIR" \
    _research/results/runs _research/results/performance _research/results/ckpts

MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST:-$(hostname)}" | head -n1)
MASTER_PORT=25678
PROJECT=${WANDB_PROJECT:-megatron-lm-research-baseline}
SCAFFOLD

# Interpolated identity + logging exports.
cat <<IDENT
EXP_NAME=${FROZEN_EXP_NAME}
CKPT_DIR=\$WORKDIR/_research/results/ckpts/\$EXP_NAME
mkdir -p "\$CKPT_DIR"
export APERTUS_LOG_DIR=\$WORKDIR/_research/results/performance
export APERTUS_FEATURE=transformer-pp-${META_OPT}
export APERTUS_TRACK=${META_TRACK}
export APERTUS_RUN_NAME=\$EXP_NAME-\${SLURM_JOB_ID:-interactive}
export APERTUS_LOG_ACT_STATS=1
export APERTUS_LOG_TOP1_ACC=1
export APERTUS_LOG_ROW_CV=1
export APERTUS_LOG_NEURON_STATS=1
export APERTUS_LOG_NEURON_INTERVAL=50
export APERTUS_LOG_PER_LAYER_GRADS=1
export APERTUS_LOG_LOSS_SPIKES=1

MEGATRON_ARGS=(
IDENT

templatize_args
echo ")"

# Self-contained wandb + launch epilogue.
cat <<'EPILOGUE'

if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_PROJECT=$PROJECT
    export WANDB_NAME=$EXP_NAME-${SLURM_JOB_ID:-interactive}
    MEGATRON_ARGS+=(
        --wandb-project "$PROJECT"
        --wandb-exp-name "$EXP_NAME-${SLURM_JOB_ID:-interactive}"
        --wandb-save-dir "$WORKDIR/_research/results/runs"
    )
else
    export WANDB_MODE=disabled
fi

MEGATRON_ARGS_STR="${MEGATRON_ARGS[*]}"
ENV_SETUP="export MASTER_ADDR=$MASTER_ADDR; export MASTER_PORT=$MASTER_PORT; \
    export RANK=\$SLURM_PROCID; export WORLD_SIZE=\$SLURM_NTASKS; \
    export LOCAL_RANK=\$SLURM_LOCALID; export LOCAL_WORLD_SIZE=$((${SLURM_NTASKS:-4}/${SLURM_NNODES:-1})); \
    export PYTHONPATH=$WORKDIR:$PACKAGE_DIR:$WORKDIR\${PYTHONPATH:+:\$PYTHONPATH}; cd $WORKDIR"
TRAINING_CMD="bash $WORKDIR/_research/launch/install_python_deps.sh $PACKAGE_DIR && python3 $WORKDIR/pretrain_gpt.py $MEGATRON_ARGS_STR"
echo "EXP_NAME: $EXP_NAME"; echo "CMD: $TRAINING_CMD"
export APERTUS_PHASE_PRE_SRUN=$(date +%s.%N)
srun -lu --mpi=pmix --network=disable_rdzv_get --environment="$WORKDIR/_research/launch/alps3.toml" \
    --cpus-per-task "${SLURM_CPUS_PER_TASK:-72}" --wait 120 \
    bash -c "export APERTUS_PHASE_CONTAINER_STARTED=\$(date +%s.%N); $ENV_SETUP; $TRAINING_CMD"
echo "END TIME: $(date)"
EPILOGUE
}

# Resolve --out auto -> next free NN- in the board's runs dir.
if [ "$OUT" = "auto" ]; then
    : "${BOARD:?--board required when --out auto}"
    RUNS_DIR="$REPO_ROOT/_research/leaderboards/$BOARD/runs"
    mkdir -p "$RUNS_DIR"
    NN=$(printf '%02d' "$(( $(ls "$RUNS_DIR"/[0-9][0-9]-*.sbatch 2>/dev/null | sed -E 's@.*/([0-9]+)-.*@\1@' | sort -n | tail -1 | sed 's/^0*//' ) + 1 ))")
    OUT="$RUNS_DIR/$NN-$SLUG.sbatch"
fi

if [ "$OUT" = "-" ]; then
    render
else
    render > "$OUT"
    echo "wrote $OUT" >&2
    echo "  next: run it, then fill in loss/W&B/rank in the header and add a row to the board README." >&2
fi
