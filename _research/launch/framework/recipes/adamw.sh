# shellcheck shell=bash
#
# adamw — plain AdamW baseline. The anchor every other recipe is compared to.

OPTIMIZER=adam
EXP_TAG=adamw

MATRIX_LR=${MATRIX_LR:-3e-4}
MIN_LR=${MIN_LR:-1e-5}
LR=${LR:-3e-4}
EMBEDDING_LR=${EMBEDDING_LR:-$LR}
KNOB_STR=mlr${MATRIX_LR}_lr${LR}_elr${EMBEDDING_LR}

WEIGHT_DECAY=0.1
ADAM_BETA1=0.9
ADAM_BETA2=0.95
CLIP_GRAD=1.0

# LR_DECAY_STYLE=WSD
LR_WARMUP_SAMPLES=128000
# LR_WSD_DECAY_STYLE=minus_sqrt
# LR_WSD_DECAY_SAMPLES=732422

# Plain AdamW needs no optimizer-specific flags.
RECIPE_ARGS=(
    --embedding-lr "$EMBEDDING_LR"
    --output-lr "$LR"
    --matrix-lr "$MATRIX_LR"
    --gains-lr "$LR"
    --min-lr-mode absolute
)

