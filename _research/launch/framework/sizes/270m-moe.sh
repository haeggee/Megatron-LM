# shellcheck shell=bash
#
# 270m-moe — smallest rung of the MoE ladder. Cheapest sanity / LR-probe point
# before spending on 420m. 20L / 768H / 12h / 4kv. 1 node.
# MoE: 64 routed experts top-2 + 1 shared (half a routed expert's width), first
# layer dense then 14 MoE (~5% dense). moe_ffn = hidden/1.5. ~0.29B active /
# ~1.68B total; ~6.3% active/total (non-embedding) — iso-sparsity proxy for
# 670B-A40B. DeepSeek-V3-style routing (sigmoid + aux-loss-free bias) is
# invariant and lives in lib/common.sh; this file sets only expert geometry.

NUM_LAYERS=14
HIDDEN=768
FFN_HIDDEN=1920
NUM_HEADS=12
NUM_KV_HEADS=4
SEQ_LEN=4096

MBS=8
GBS=128
TRAIN_SAMPLES=${TRAIN_SAMPLES:-3584000}   # ~15B tokens (override for other budgets)
SAVE_INTERVAL=2289

APERTUS_TRACK=270a-moe

# Expert geometry only — routing policy is in common.sh.
MOE_ARGS=(
    --num-experts 64
    --moe-router-topk 2
    --moe-ffn-hidden-size 512
    --moe-shared-expert-intermediate-size 256
    --moe-layer-freq "([0]*1+[1]*13)"
)
EP=1                           # pure DP: all 64 experts replicated on each GPU

DEFAULT_NODES=2
DEFAULT_TIME=12:00:00
