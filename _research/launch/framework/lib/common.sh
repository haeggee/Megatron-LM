# shellcheck shell=bash
#
# common.sh — the invariant training scaffolding for every run.
#
# This file is SOURCED by train.sbatch AFTER the chosen size/ and recipe/
# fragments have run. Those fragments set a handful of variables and arrays
# (the "contract" below); this file fills in every default they didn't set,
# assembles the final MEGATRON_ARGS, wires up wandb, and launches via srun.
#
# You almost never edit this file. Everything experiment-specific lives in
# sizes/<size>.sh (model + data scale) and recipes/<name>.sh (optimizer/arch).
#
# ── contract ────────────────────────────────────────────────────────────────
# A size file MUST set:
#   NUM_LAYERS HIDDEN FFN_HIDDEN NUM_HEADS NUM_KV_HEADS SEQ_LEN
#   MBS GBS TRAIN_SAMPLES SAVE_INTERVAL APERTUS_TRACK
#   MOE_ARGS=(...)        # array; empty () for a dense model
# and MAY set (else defaulted here):
#   TP PP CP EP  EMBEDDING_MULTIPLIER  INIT_STD  EXIT_DURATION_MINS
#   DEFAULT_NODES DEFAULT_TIME         # read by submit.sh, not used here
#
# A recipe file MUST set:
#   OPTIMIZER            # --optimizer value (adam|muon|master|muown|...)
#   EXP_TAG              # short slug for the run name, e.g. "master"
#   RECIPE_ARGS=(...)    # optimizer/arch-specific flags (the actual idea)
# and MAY set (else defaulted here):
#   LR MIN_LR  KNOB_STR                # KNOB_STR is appended to EXP_NAME
#                                      # (RUN_TAG, from the env, is appended too)
#   WEIGHT_DECAY ADAM_BETA1 ADAM_BETA2 CLIP_GRAD
#   LR_DECAY_STYLE LR_WARMUP_SAMPLES LR_WSD_DECAY_STYLE LR_WSD_DECAY_SAMPLES
#   EXTRA_REG_ARGS=(...)               # escape hatch for odd reg flags
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

: "${SIZE:?set SIZE (see _research/launch/framework/sizes/)}"
: "${RECIPE:?set RECIPE (see _research/launch/framework/recipes/)}"
: "${MEGATRON_DATA_PATH:?please set MEGATRON_DATA_PATH to the Megatron-binary dataset prefix (without .bin/.idx), e.g. /path/to/climbmix_small}"

[ -z "${DRY_RUN:-}" ] && echo "START TIME: $(date)"
export APERTUS_PHASE_SBATCH_START=$(date +%s.%N)

# ── paths & environment (identical across every run) ─────────────────────────
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

mkdir -p "$DATASET_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
    "$TMP_CACHE_DIR" "$PACKAGE_DIR" \
    _research/results/runs _research/results/performance _research/results/ckpts

MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST:-$(hostname)}" | head -n1)
MASTER_PORT=${MASTER_PORT:-25678}

# ── defaults for anything the size/recipe fragment didn't set ────────────────
TP=${TP:-1}; PP=${PP:-1}; CP=${CP:-1}; EP=${EP:-1}
INIT_STD=${INIT_STD:-0.02}
# Embedding multiplier defaults to sqrt(HIDDEN) (e.g. 1024 -> 32.0).
EMBEDDING_MULTIPLIER=${EMBEDDING_MULTIPLIER:-$(awk "BEGIN{printf \"%.1f\", sqrt($HIDDEN)}")}
EXIT_DURATION_MINS=${EXIT_DURATION_MINS:-595}

LR=${LR:-1e-3}
MIN_LR=${MIN_LR:-1e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
ADAM_BETA1=${ADAM_BETA1:-0.9}
ADAM_BETA2=${ADAM_BETA2:-0.95}
LR_DECAY_STYLE=${LR_DECAY_STYLE:-WSD}
LR_WARMUP_SAMPLES=${LR_WARMUP_SAMPLES:-0}
LR_WSD_DECAY_STYLE=${LR_WSD_DECAY_STYLE:-minus_sqrt}
LR_WSD_DECAY_SAMPLES=${LR_WSD_DECAY_SAMPLES:-732422}
KNOB_STR=${KNOB_STR:-lr${LR}}

# Default to an empty array ONLY when the fragment didn't declare one. (The
# `("${X[@]:-}")` idiom would inject a phantom empty-string element instead.)
declare -p RECIPE_ARGS    >/dev/null 2>&1 || RECIPE_ARGS=()
declare -p MOE_ARGS       >/dev/null 2>&1 || MOE_ARGS=()
declare -p EXTRA_REG_ARGS >/dev/null 2>&1 || EXTRA_REG_ARGS=()

# ── run identity ─────────────────────────────────────────────────────────────
PROJECT=${WANDB_PROJECT:-megatron-lm-research-baseline}
# Run name = size + recipe tag + swept knobs (+ optional RUN_TAG to disambiguate
# any axis KNOB_STR doesn't already cover, e.g. token budget in a length sweep).
EXP_NAME=${EXP_NAME:-${SIZE}-${EXP_TAG}-${KNOB_STR}${RUN_TAG:+-${RUN_TAG}}}
CKPT_DIR=${CKPT_DIR:-$WORKDIR/_research/results/ckpts/$EXP_NAME}
mkdir -p "$CKPT_DIR"

export APERTUS_LOG_DIR=$WORKDIR/_research/results/performance
export APERTUS_FEATURE=${APERTUS_FEATURE:-transformer-pp}
export APERTUS_TRACK=${APERTUS_TRACK:-$SIZE}
export APERTUS_RUN_NAME=$EXP_NAME-${SLURM_JOB_ID:-interactive}
export APERTUS_LOG_ACT_STATS=1
export APERTUS_LOG_TOP1_ACC=1
export APERTUS_LOG_ROW_CV=1
export APERTUS_LOG_NEURON_STATS=1
export APERTUS_LOG_NEURON_INTERVAL=50
export APERTUS_LOG_PER_LAYER_GRADS=1
export APERTUS_LOG_LOSS_SPIKES=1

# ── invariant arg groups ─────────────────────────────────────────────────────
TRANSFORMER_ENGINE_ARGS=(
    --transformer-impl transformer_engine
    --main-grads-dtype fp32
)

NETWORK_SIZE_ARGS=(
    --num-layers "$NUM_LAYERS"
    --hidden-size "$HIDDEN"
    --ffn-hidden-size "$FFN_HIDDEN"
    --num-attention-heads "$NUM_HEADS"
    --group-query-attention
    --num-query-groups "$NUM_KV_HEADS"
    --max-position-embeddings "$SEQ_LEN"
    --position-embedding-type rope
    --rotary-base 500000
    --make-vocab-size-divisible-by 128
    --normalization RMSNorm
    --swiglu
    --qk-layernorm
    --untie-embeddings-and-output-weights
    --attention-backend auto
    --seq-length "$SEQ_LEN"
)

TRAINING_ARGS=(
    --micro-batch-size "$MBS"
    --global-batch-size "$GBS"
    --train-samples "$TRAIN_SAMPLES"
    --exit-duration-in-mins "$EXIT_DURATION_MINS"
    --log-interval 1
    --eval-interval "$TRAIN_SAMPLES"
    --eval-iters 0
    --optimizer "$OPTIMIZER"
    --dataloader-type single
    --manual-gc
    --manual-gc-interval 50
    --cross-entropy-loss-fusion
    --disable-bias-linear
    --no-check-for-nan-in-loss-and-grad
)

REGULARIZATION_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --weight-decay "$WEIGHT_DECAY"
    --adam-beta1 "$ADAM_BETA1"
    --adam-beta2 "$ADAM_BETA2"
)
[ -n "${CLIP_GRAD:-}" ] && REGULARIZATION_ARGS+=(--clip-grad "$CLIP_GRAD")
[ "${#EXTRA_REG_ARGS[@]}" -gt 0 ] && REGULARIZATION_ARGS+=("${EXTRA_REG_ARGS[@]}")

LEARNING_RATE_ARGS=(
    --lr "$LR"
    --min-lr "$MIN_LR"
    --lr-decay-style "$LR_DECAY_STYLE"
    --lr-warmup-samples "$LR_WARMUP_SAMPLES"
)
if [ "$LR_DECAY_STYLE" = "WSD" ]; then
    LEARNING_RATE_ARGS+=(
        --lr-wsd-decay-style "$LR_WSD_DECAY_STYLE"
        --lr-wsd-decay-samples "$LR_WSD_DECAY_SAMPLES"
    )
fi

INITIALIZATION_ARGS=(
    --seed 42
    --init-method-std "$INIT_STD"
    --embedding-multiplier "$EMBEDDING_MULTIPLIER"
)

MIXED_PRECISION_ARGS=(
    --bf16
    --fp8-format e4m3
    --fp8-recipe blockwise
)

DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size "$TP"
    --pipeline-model-parallel-size "$PP"
    --context-parallel-size "$CP"
    --expert-model-parallel-size "$EP"
    --use-distributed-optimizer
    --overlap-grad-reduce
    --sequence-parallel
)

LOGGING_ARGS=(
    --save "$CKPT_DIR"
    --load "$CKPT_DIR"
    --save-interval "$SAVE_INTERVAL"
    --ckpt-format torch
    --log-throughput
    --log-progress
    --log-memory-to-tensorboard
    --log-params-norm
    --tensorboard-dir "$WORKDIR/_research/results/runs/tb-${SLURM_JOB_ID:-interactive}"
    --tensorboard-log-interval 1
    --log-timers-to-tensorboard
)

TOKENIZER_ARGS=(
    --tokenizer-type GPT2BPETokenizer
    --vocab-file "$WORKDIR/_research/data/gpt2-vocab.json"
    --merge-file "$WORKDIR/_research/data/gpt2-merges.txt"
)

DATA_ARGS=(
    --data-path "$MEGATRON_DATA_PATH"
    --data-cache-path "$DATASET_CACHE_DIR"
    --split 99,1,0
    --num-workers 4
)

# Flat array — order matches the legacy hand-written sbatch files so that the
# composed arg list is byte-for-byte comparable (see verify-equivalence.sh).
# RECIPE_ARGS sits where each legacy file put its optimizer block.
MEGATRON_ARGS=(
    "${TRANSFORMER_ENGINE_ARGS[@]}"
    "${NETWORK_SIZE_ARGS[@]}"
)
[ "${#MOE_ARGS[@]}" -gt 0 ] && MEGATRON_ARGS+=("${MOE_ARGS[@]}")
MEGATRON_ARGS+=(
    "${TRAINING_ARGS[@]}"
    "${REGULARIZATION_ARGS[@]}"
)
[ "${#RECIPE_ARGS[@]}" -gt 0 ] && MEGATRON_ARGS+=("${RECIPE_ARGS[@]}")
MEGATRON_ARGS+=(
    "${LEARNING_RATE_ARGS[@]}"
    "${INITIALIZATION_ARGS[@]}"
    "${MIXED_PRECISION_ARGS[@]}"
    "${DISTRIBUTED_ARGS[@]}"
    "${LOGGING_ARGS[@]}"
    "${TOKENIZER_ARGS[@]}"
    "${DATA_ARGS[@]}"
)

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

# When sourced for arg inspection (DRY_RUN=1) or by interactive-run.sh, stop
# here: the caller just wants MEGATRON_ARGS populated, not a launch.
if [ -n "${DRY_RUN:-}" ] || [ -n "${COMMON_NO_LAUNCH:-}" ]; then
    echo "SIZE=$SIZE  RECIPE=$RECIPE  OPTIMIZER=$OPTIMIZER  EXP_NAME=$EXP_NAME" >&2
    return 0 2>/dev/null || exit 0
fi

MEGATRON_ARGS_STR="${MEGATRON_ARGS[*]}"

ENV_SETUP="export MASTER_ADDR=$MASTER_ADDR; \
    export MASTER_PORT=$MASTER_PORT; \
    export RANK=\$SLURM_PROCID; \
    export WORLD_SIZE=\$SLURM_NTASKS; \
    export LOCAL_RANK=\$SLURM_LOCALID; \
    export LOCAL_WORLD_SIZE=$((${SLURM_NTASKS:-4}/${SLURM_NNODES:-1})); \
    export PYTHONPATH=$WORKDIR:$PACKAGE_DIR:$WORKDIR\${PYTHONPATH:+:\$PYTHONPATH}; \
    cd $WORKDIR"

TRAINING_CMD="bash $WORKDIR/_research/launch/install_python_deps.sh $PACKAGE_DIR && python3 $WORKDIR/pretrain_gpt.py $MEGATRON_ARGS_STR"

echo "SIZE=$SIZE  RECIPE=$RECIPE  OPTIMIZER=$OPTIMIZER"
echo "EXP_NAME: $EXP_NAME"
echo "CKPT_DIR: $CKPT_DIR"
echo "CMD: $TRAINING_CMD"
export APERTUS_PHASE_PRE_SRUN=$(date +%s.%N)
SRUN_RC=0
srun -lu --mpi=pmix --network=disable_rdzv_get --environment="$WORKDIR/_research/launch/alps3.toml" \
    --cpus-per-task "${SLURM_CPUS_PER_TASK:-72}" --wait 120 \
    bash -c "export APERTUS_PHASE_CONTAINER_STARTED=\$(date +%s.%N); $ENV_SETUP; $TRAINING_CMD" \
    || SRUN_RC=$?

echo "END TIME: $(date)"

# ── auto-requeue: chain the next allocation if training isn't done yet ────────
# Enabled by `submit.sh --auto-requeue` (sets AUTO_REQUEUE=1 in the job env).
# We chain ONLY on a clean exit — rc 0 means Megatron either finished or hit
# --exit-duration-in-mins and saved a checkpoint. A crash (rc != 0) deliberately
# stops the chain so a human investigates instead of burning the queue on a loop.
# The next job shares the same EXP_NAME -> same CKPT_DIR -> resumes via --load.
maybe_auto_requeue() {
    [ -n "${AUTO_REQUEUE:-}" ] || return 0

    if [ "$SRUN_RC" -ne 0 ]; then
        echo ">>> auto-requeue: srun exited $SRUN_RC (not clean) — NOT chaining."
        return 0
    fi

    local rq_count=${REQUEUE_COUNT:-0} rq_max=${MAX_REQUEUES:-50}
    if [ "$rq_count" -ge "$rq_max" ]; then
        echo ">>> auto-requeue: hit MAX_REQUEUES=$rq_max — NOT chaining."
        return 0
    fi

    local marker="$CKPT_DIR/latest_checkpointed_iteration.txt"
    if [ ! -f "$marker" ]; then
        echo ">>> auto-requeue: no checkpoint written yet — NOT chaining."
        return 0
    fi
    local iter samples_done
    iter=$(tr -dc '0-9' < "$marker")
    if [ -z "$iter" ]; then
        echo ">>> auto-requeue: marker not numeric ('$(cat "$marker")') — NOT chaining."
        return 0
    fi
    samples_done=$(( iter * GBS ))
    if [ "$samples_done" -ge "$TRAIN_SAMPLES" ]; then
        echo ">>> auto-requeue: reached $samples_done/$TRAIN_SAMPLES samples — DONE."
        return 0
    fi

    echo ">>> auto-requeue: $samples_done/$TRAIN_SAMPLES samples (chain #$((rq_count+1))/$rq_max) — submitting dependent job."
    sbatch \
        --dependency=afterany:"${SLURM_JOB_ID}" \
        --nodes="${SLURM_NNODES:-1}" \
        --time="${REQUEUE_TIME:-${DEFAULT_TIME:-10:00:00}}" \
        --job-name="${SLURM_JOB_NAME:-$SIZE-$RECIPE}" \
        --export=ALL,SIZE="$SIZE",RECIPE="$RECIPE",FRAMEWORK_DIR="$FRAMEWORK_DIR",AUTO_REQUEUE=1,REQUEUE_COUNT=$((rq_count+1)),MAX_REQUEUES="$rq_max",REQUEUE_TIME="${REQUEUE_TIME:-${DEFAULT_TIME:-10:00:00}}" \
        "$FRAMEWORK_DIR/train.sbatch"
}
maybe_auto_requeue

exit "$SRUN_RC"
