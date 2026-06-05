# shellcheck shell=bash
#
# normuon-kimi — NorMuon with Kimi/Moonlight-style scaling: spectral scale
# mode + extra scale factor 0.2, so every matrix update has entry-RMS ~0.2
# (the AdamW-matched RMS from Moonlight, arXiv 2502.16982) and MATRIX_LR
# transfers from AdamW. MOMENT2 picks the variant:
#   normuonfix (default) — NorMuon rescale as a pure direction change; the
#     frob renorm re-applies scale_mode * extra_scale_factor exactly, so
#     update magnitudes match plain Muon under the same flags.
#   normuon — upstream NorMuon: the rescale cancels the scale-mode factor
#     AND the 0.2 (scale mode + extra factor are no-ops; every matrix gets
#     entry-RMS~1 updates) — kimi scaling has no effect there.

OPTIMIZER=adaptive_muon
EXP_TAG=normuon-kimi

MATRIX_LR=${MATRIX_LR:-3e-4}              # AdamW-transferable under kimi scaling (entry-RMS 0.2)
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
WEIGHT_DECAY=0.1
ADAM_BETA1=0.9
ADAM_BETA2=0.95

LR_WARMUP_SAMPLES=128000

RECIPE_ARGS=(
    --muon-scale-mode spectral
    --muon-extra-scale-factor 0.2
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
