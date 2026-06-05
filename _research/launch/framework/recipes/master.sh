# shellcheck shell=bash
#
# master — Adam/AdEMAMix + Muon-style orthogonalized updates + L2 hypersphere
# weight clipping + learnable per-axis gains. Embedding + LM head ALWAYS use
# the Adam branch (hardcoded). --master-min-lr-mode absolute: every per-group
# LR floor is MIN_LR.
#
# Knobs (override via env, e.g. `MLR=1e-2 ELR=3e-3 submit.sh ...`):
#   LR   scalar-group LR (LM head, biases, norm gains)
#   MLR  matrix LR — tune INDEPENDENTLY of LR
#   ELR  embedding LR — ABSOLUTE (was a multiplier; old ELR=3 ≡ new ELR=3e-3
#        at LR=1e-3)
#   OLR  output (LM head) LR — defaults to LR

OPTIMIZER=master
EXP_TAG=master

LR=${LR:-1e-3}
MLR=${MLR:-8e-3}
MIN_LR=${MIN_LR:-1e-5}
ELR=${ELR:-$LR}
OLR=${OLR:-$LR}
KNOB_STR=lr${LR}-mlr${MLR}-elr${ELR}
[ "$OLR" != "$LR" ] && KNOB_STR=${KNOB_STR}-olr${OLR}

# master's canonical regularization
WEIGHT_DECAY=0.0
ADAM_BETA1=0.9
ADAM_BETA2=0.95

# Linear decay to MIN_LR over all of TRAIN_SAMPLES (auto-scales with run length;
# no cooldown to tune). Overridable from the env so a sweep can switch styles
# without a new recipe, e.g. LR_DECAY_STYLE=WSD LR_WSD_DECAY_SAMPLES=...
LR_DECAY_STYLE=${LR_DECAY_STYLE:-linear}
LR_WARMUP_SAMPLES=${LR_WARMUP_SAMPLES:-0}

RECIPE_ARGS=(
    --use-orthogonal-updates
    --ademamix-alpha 0
    --hypersphere-mode flat
    --hypersphere-embedding-mode row
    --hypersphere-gains-mode rowcol
    --hypersphere-gains-mode-embedding none
    --muon-momentum 0.95
    --muon-nesterov
    --muon-scale-mode shape_up
    --matrix-lr "$MLR"
    --embedding-lr "$ELR"
    --gains-lr "$LR"
    --output-lr "$OLR"
    # router: use muon branch, normalize row-wise.
    --router-use-orthogonal-updates true
    --hypersphere-router-mode row
    --min-lr-mode absolute
    --hypersphere-scale-out-proj-init
    # gains
    --gain-parametrization softplus
)
