# shellcheck shell=bash
#
# adaptive-muon — Muon with spectral scaling, the base for the linear-attention
# (deltanet/gdn/mamba) architecture ablations.

OPTIMIZER=adaptive_muon
EXP_TAG=adaptive-muon

LR=${LR:-1e-2}
MIN_LR=${MIN_LR:-1e-5}
SCALAR_LR=${SCALAR_LR:-1.5e-3}
KNOB_STR=lr${LR}

WEIGHT_DECAY=0.0
ADAM_BETA1=0.9
ADAM_BETA2=0.95

RECIPE_ARGS=(
    --muon-scale-mode spectral
    --muon-nesterov
    --muon-momentum 0.95
    --muon-scalar-lr "$SCALAR_LR"
    --muon-scalar-weight-decay 0.0
)
