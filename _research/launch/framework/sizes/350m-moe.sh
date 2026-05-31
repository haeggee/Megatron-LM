# shellcheck shell=bash
#
# 350m-moe — THE MAIN BASELINE.
#
# Transformer++ 350M-active MoE. 24L / 1024H / 16h / 4kv (GQA).
# MoE: 16 routed experts top-1 + 1 shared, ffn-hidden 1280 (iso-active to
# dense 2560). ~355M active / ~1.76B total params. FP8 blockwise.
# 15B tokens on ClimbMix: 3,662,109 samples * 4096 seq = 15.0B, ~57k steps.
#
# Node count is just a data-parallel layout knob — GBS and TRAIN_SAMPLES are
# fixed, so 2 vs 4 nodes only changes wall-clock, not the run. submit.sh
# defaults to DEFAULT_NODES; override with --nodes.

NUM_LAYERS=24
HIDDEN=1024
FFN_HIDDEN=2560
NUM_HEADS=16
NUM_KV_HEADS=4
SEQ_LEN=4096

MBS=8
GBS=128
TRAIN_SAMPLES=${TRAIN_SAMPLES:-3662109}   # 15.0B tokens (overridable for length sweeps)
SAVE_INTERVAL=5722             # ~10 saves over the run

EMBEDDING_MULTIPLIER=32.0      # sqrt(1024)

APERTUS_TRACK=transformer-pp-350m

MOE_ARGS=(
    --num-experts 16
    --moe-router-topk 1
    --moe-router-pre-softmax
    --moe-ffn-hidden-size 1280
    --moe-shared-expert-intermediate-size 1280
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-z-loss-coeff 1e-3
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-token-dispatcher-type alltoall
    --moe-per-layer-logging
)

# Defaults read by submit.sh (overridable on the CLI).
DEFAULT_NODES=2
DEFAULT_TIME=10:00:00
