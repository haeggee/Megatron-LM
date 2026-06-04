# shellcheck shell=bash
#
# muown — Muon-with-own-norm variant. Muon on matrices, AdamW on scalars,
# with weight decay and grad clipping on (unlike plain muon).

OPTIMIZER=muown
EXP_TAG=muown

MATRIX_LR=${MATRIX_LR:-1e-3}                 # matrix LR
MIN_LR=${MIN_LR:-1e-5}
SCALAR_LR=${SCALAR_LR:-1e-3}
KNOB_STR=mlr${MATRIX_LR}_lr${SCALAR_LR}

WEIGHT_DECAY=0.0
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
    --embedding-lr "$SCALAR_LR"
    --output-lr "$SCALAR_LR"
    --gains-lr "$SCALAR_LR"
    --muon-scalar-weight-decay 0.0
)
