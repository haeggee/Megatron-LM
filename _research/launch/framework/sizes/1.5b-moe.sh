# shellcheck shell=bash
#
# 1.5b-moe — top scale-up rung of the 16e-tk1-sh1 MoE ladder.
# 32L / 2048H / 32h / 8kv. MoE: 16 routed top-1 + 1 shared, width 2560
# (= ffn/2). The confirmation rung for promoting a 350m winner.

NUM_LAYERS=32
HIDDEN=2048
FFN_HIDDEN=5120
NUM_HEADS=32
NUM_KV_HEADS=8
SEQ_LEN=4096

MBS=4
GBS=128
TRAIN_SAMPLES=${TRAIN_SAMPLES:-7324219}   # ~30B tokens (override for other budgets)
SAVE_INTERVAL=5722

APERTUS_TRACK=transformer-pp-1.5b

MOE_ARGS=(
    --num-experts 16
    --moe-router-topk 1
    --moe-router-pre-softmax
    --moe-ffn-hidden-size 2560
    --moe-shared-expert-intermediate-size 2560
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-z-loss-coeff 1e-3
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-token-dispatcher-type alltoall
    --moe-per-layer-logging
)

DEFAULT_NODES=4
DEFAULT_TIME=12:00:00
