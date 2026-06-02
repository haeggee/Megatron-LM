# shellcheck shell=bash
#
# muon — Muon on matrices, AdamW on scalars (embeddings, norms, biases, router).

OPTIMIZER=muon
EXP_TAG=muon

LR=${LR:-1e-2}                 # matrix (Muon) LR
MIN_LR=${MIN_LR:-1e-5}
SCALAR_LR=${SCALAR_LR:-1e-3}
KNOB_STR=lr${LR}

WEIGHT_DECAY=0.1
ADAM_BETA1=0.9
ADAM_BETA2=0.95

RECIPE_ARGS=(
    --muon-scale-mode shape_scaling
    --muon-nesterov
    --muon-momentum 0.95
    --muon-scalar-lr "$SCALAR_LR"
    --muon-scalar-weight-decay 0.0
    --min-lr-mode absolute
)
