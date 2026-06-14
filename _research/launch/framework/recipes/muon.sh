# shellcheck shell=bash
#
# muon — Muon on matrices (incl. MoE router), AdamW on everything else
# (embedding, LM head, norms, biases) at the base LR.
#
# Knobs (override via env, e.g. `MATRIX_LR=2e-2 submit.sh ...`):
#   MATRIX_LR  matrix (Muon) LR
#   LR         base LR — everything Adam-managed (embedding, output, gains,
#              and the router when ROUTER_ADAM=1)
#   ROUTER_ADAM=1  route the MoE router to AdamW via
#              --router-use-orthogonal-updates false; the router then uses the
#              base LR instead of MATRIX_LR, matching the master recipe's
#              adam-branch router. Default 0 (router on Muon).

OPTIMIZER=muon
EXP_TAG=muon

MATRIX_LR=${MATRIX_LR:-1e-2}                 # matrix (Muon) LR
MIN_LR=${MIN_LR:-1e-5}
LR=${LR:-1e-3}
EMBEDDING_LR=${EMBEDDING_LR:-$LR}            # per-class overrides (default: base LR)
OUTPUT_LR=${OUTPUT_LR:-$LR}
GAINS_LR=${GAINS_LR:-$LR}
KNOB_STR=lr${LR}-mlr${MATRIX_LR}
[ "$EMBEDDING_LR" != "$LR" ] && KNOB_STR=${KNOB_STR}-elr${EMBEDDING_LR}
[ "$OUTPUT_LR" != "$LR" ] && KNOB_STR=${KNOB_STR}-olr${OUTPUT_LR}
[ "$GAINS_LR" != "$LR" ] && KNOB_STR=${KNOB_STR}-glr${GAINS_LR}

WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
[ "$WEIGHT_DECAY" != 0.1 ] && KNOB_STR=${KNOB_STR}-wd${WEIGHT_DECAY}
ADAM_BETA1=0.9
ADAM_BETA2=0.95

LR_WARMUP_SAMPLES=${LR_WARMUP_SAMPLES:-128000}

RECIPE_ARGS=(
    --muon-scale-mode shape_scaling
    --muon-nesterov
    --muon-momentum 0.95
    # LR for everything Adam-managed, via the per-class knobs
    # (--muon-scalar-lr is now 1D-only; this reproduces the old lumped group).
    --matrix-lr "$MATRIX_LR"
    --embedding-lr "$EMBEDDING_LR"
    --output-lr "$OUTPUT_LR"
    --gains-lr "$GAINS_LR"
    --muon-scalar-weight-decay 0.0
    --min-lr-mode absolute
)

# Router-on-AdamW variant (tagged -radam so it gets its own EXP_NAME/ckpt dir).
ROUTER_ADAM=${ROUTER_ADAM:-0}
if [ "$ROUTER_ADAM" = 1 ]; then
    RECIPE_ARGS+=(--router-use-orthogonal-updates false)
    KNOB_STR=${KNOB_STR}-radam
fi
