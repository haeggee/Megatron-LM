# shellcheck shell=bash
#
# muown — Muon-with-own-norm variant. Muon on matrices, AdamW on scalars,
# with weight decay and grad clipping on (unlike plain muon).

OPTIMIZER=muown
EXP_TAG=muown

MATRIX_LR=${MATRIX_LR:-1e-3}                 # matrix LR
MIN_LR=${MIN_LR:-1e-5}
SCALAR_LR=${SCALAR_LR:-1e-3}
EMBEDDING_LR=${EMBEDDING_LR:-$SCALAR_LR}     # per-class overrides (default: SCALAR_LR)
OUTPUT_LR=${OUTPUT_LR:-$SCALAR_LR}
GAINS_LR=${GAINS_LR:-$SCALAR_LR}
KNOB_STR=mlr${MATRIX_LR}_lr${SCALAR_LR}
[ "$EMBEDDING_LR" != "$SCALAR_LR" ] && KNOB_STR=${KNOB_STR}_elr${EMBEDDING_LR}
[ "$OUTPUT_LR" != "$SCALAR_LR" ] && KNOB_STR=${KNOB_STR}_olr${OUTPUT_LR}
[ "$GAINS_LR" != "$SCALAR_LR" ] && KNOB_STR=${KNOB_STR}_glr${GAINS_LR}

WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
[ "$WEIGHT_DECAY" != 0.0 ] && KNOB_STR=${KNOB_STR}_wd${WEIGHT_DECAY}
ADAM_BETA1=0.9
ADAM_BETA2=0.95
CLIP_GRAD=1.0

RECIPE_ARGS=(
    --muon-nesterov
    --muon-momentum 0.95
    --muown-eps 1e-8
    # SCALAR_LR for everything Adam-managed, via the per-class knobs
    # (--muon-scalar-lr is now 1D-only; this reproduces the old lumped group).
    --matrix-lr "$MATRIX_LR"
    --embedding-lr "$EMBEDDING_LR"
    --output-lr "$OUTPUT_LR"
    --gains-lr "$GAINS_LR"
    --muon-scalar-weight-decay 0.0
)
