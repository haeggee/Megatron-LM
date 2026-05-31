# shellcheck shell=bash
#
# 760m-moe — second scale-up rung of the 16e-tk1-sh1 MoE ladder.
# 24L / 1536H / 24h / 8kv. MoE: 16 routed top-1 + 1 shared, width 2048
# (= ffn/2). Leaderboard ladder budget: 30B tokens, 4 nodes.

NUM_LAYERS=24
HIDDEN=1536
FFN_HIDDEN=4096
NUM_HEADS=24
NUM_KV_HEADS=8
SEQ_LEN=4096

MBS=4
GBS=128
TRAIN_SAMPLES=${TRAIN_SAMPLES:-7324219}   # ~30B tokens (leaderboard 760m budget)
SAVE_INTERVAL=5722

APERTUS_TRACK=transformer-pp-760m

MOE_ARGS=(
    --num-experts 16
    --moe-router-topk 1
    --moe-router-pre-softmax
    --moe-ffn-hidden-size 2048
    --moe-shared-expert-intermediate-size 2048
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-z-loss-coeff 1e-3
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-token-dispatcher-type alltoall
    --moe-per-layer-logging
)

DEFAULT_NODES=4
DEFAULT_TIME=10:00:00
