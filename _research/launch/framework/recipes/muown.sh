# shellcheck shell=bash
#
# muown — Muon-with-own-norm variant. Muon on matrices, AdamW on scalars,
# with weight decay and grad clipping on (unlike plain muon).

OPTIMIZER=muown
EXP_TAG=muown

LR=${LR:-1e-3}                 # matrix LR
MIN_LR=${MIN_LR:-1e-4}
SCALAR_LR=${SCALAR_LR:-1.5e-3}
KNOB_STR=lr${LR}

WEIGHT_DECAY=0.1
ADAM_BETA1=0.9
ADAM_BETA2=0.95
CLIP_GRAD=1.0

RECIPE_ARGS=(
    --muon-nesterov
    --muon-momentum 0.95
    --muown-eps 1e-8
    --muon-scalar-lr "$SCALAR_LR"
    --muon-scalar-weight-decay 0.0
)
