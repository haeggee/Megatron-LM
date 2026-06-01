# shellcheck shell=bash
#
# normuon — NorMuon: the same spectral-scaled Muon as adaptive-muon, but with
# the `normuon` second-moment normalization on the matrix update. That changes
# the matrix-LR regime sharply (natural LR is ~30x smaller than plain
# adaptive-muon's 1e-2), so it carries its own LR default.

OPTIMIZER=adaptive_muon
EXP_TAG=normuon

LR=${LR:-3.6e-4}              # matrix LR (much smaller than adaptive-muon's 1e-2)
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
    --adaptive-muon-moment2-method normuon
)
