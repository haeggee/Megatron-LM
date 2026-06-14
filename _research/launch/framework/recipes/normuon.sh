# shellcheck shell=bash
#
# normuon — NorMuon: Muon + per-neuron second-moment rescale on the matrix
# update. MOMENT2 picks the variant:
#   normuonfix (default) — NorMuon rescale as a pure direction change; update
#     magnitude renormalized to plain Muon's, so --muon-scale-mode behaves
#     exactly as in the muon recipe and MATRIX_LR transfers (frob = sqrt(out)
#     under shape_scaling).
#   normuon — upstream NorMuon: the rescale cancels the scale-mode factor
#     (every matrix gets entry-RMS~1 updates; natural MATRIX_LR ~sqrt(hidden)
#     smaller, scale mode is a no-op).

OPTIMIZER=adaptive_muon
EXP_TAG=normuon

MATRIX_LR=${MATRIX_LR:-1e-2}              # matrix LR (muon-comparable under normuonfix)
MIN_LR=${MIN_LR:-1e-5}
SCALAR_LR=${SCALAR_LR:-1e-3}
EMBEDDING_LR=${EMBEDDING_LR:-$SCALAR_LR}  # per-class overrides (default: SCALAR_LR)
OUTPUT_LR=${OUTPUT_LR:-$SCALAR_LR}
GAINS_LR=${GAINS_LR:-$SCALAR_LR}
MOMENT2=${MOMENT2:-normuonfix}            # normuonfix | normuon | adamuon
KNOB_STR=fixed_lr${SCALAR_LR}_mlr${MATRIX_LR}
[ "$EMBEDDING_LR" != "$SCALAR_LR" ] && KNOB_STR=${KNOB_STR}_elr${EMBEDDING_LR}
[ "$OUTPUT_LR" != "$SCALAR_LR" ] && KNOB_STR=${KNOB_STR}_olr${OUTPUT_LR}
[ "$GAINS_LR" != "$SCALAR_LR" ] && KNOB_STR=${KNOB_STR}_glr${GAINS_LR}
[ "$MOMENT2" != normuonfix ] && KNOB_STR=${KNOB_STR}-${MOMENT2}

# Match muon.sh so muon-vs-normuonfix is apples-to-apples (WD equilibrium and
# warmup drive early param-norm growth far more than the moment2 method).
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
[ "$WEIGHT_DECAY" != 0.1 ] && KNOB_STR=${KNOB_STR}_wd${WEIGHT_DECAY}
ADAM_BETA1=0.9
ADAM_BETA2=0.95

LR_WARMUP_SAMPLES=${LR_WARMUP_SAMPLES:-128000}

RECIPE_ARGS=(
    --muon-scale-mode shape_scaling
    --muon-nesterov
    --muon-momentum 0.95
    # SCALAR_LR for everything Adam-managed, via the per-class knobs
    # (--muon-scalar-lr is now 1D-only; this reproduces the old lumped group).
    --matrix-lr "$MATRIX_LR"
    --embedding-lr "$EMBEDDING_LR"
    --output-lr "$OUTPUT_LR"
    --gains-lr "$GAINS_LR"
    --muon-scalar-weight-decay 0.0
    --adaptive-muon-moment2-method "$MOMENT2"
)
