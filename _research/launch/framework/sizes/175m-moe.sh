# shellcheck shell=bash
#
# 175m-moe — smallest rung of the 16e-tk1-sh1 MoE ladder. Cheapest sanity /
# LR-probe point before spending on 350m. 16L / 768H / 12h / 4kv.
# MoE: 16 routed top-1 + 1 shared, width 960 (= ffn/2). 1 node.

NUM_LAYERS=16
HIDDEN=768
FFN_HIDDEN=1920
NUM_HEADS=12
NUM_KV_HEADS=4
SEQ_LEN=4096

MBS=16
GBS=128
TRAIN_SAMPLES=${TRAIN_SAMPLES:-2929920}   # ~12B tokens (override for other budgets)
SAVE_INTERVAL=2289

APERTUS_TRACK=transformer-pp-175m

MOE_ARGS=(
    --num-experts 16
    --moe-router-topk 1
    --moe-router-pre-softmax
    --moe-ffn-hidden-size 960
    --moe-shared-expert-intermediate-size 960
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 1e-2
    --moe-z-loss-coeff 1e-3
    --moe-grouped-gemm
    --moe-permute-fusion
    --moe-token-dispatcher-type alltoall
    --moe-per-layer-logging
)

DEFAULT_NODES=1
DEFAULT_TIME=04:00:00
