#!/bin/bash
#
# sweep-batchsize.sh — batch-size robustness per optimizer at fixed tokens.
#
# Fixed size + token budget (default: the tuned 270m-moe @ 15B point, so the
# existing GBS=128 runs anchor the sweep for free); GBS varies, steps shrink
# 1/GBS. LRs start from the tuned optima (same block as sweep-scaling-laws.sh)
# and scale by (GBS/128)^BSPOW — sqrt by default, the standard Adam heuristic;
# for muon/master the right exponent IS the experiment, so verify the rule
# locally with LR_MULTS:
#
#   bash sweep-batchsize.sh --dry-run                      # GBS {32..512} x recipes
#   bash sweep-batchsize.sh --gbs "256 512" --recipes master
#   LR_MULTS="0.71 1 1.41" bash sweep-batchsize.sh --gbs 512   # rule check at the edge
#   MASTER_BSPOW=0 bash sweep-batchsize.sh ...             # batch-invariant master LR
#
# Warmup is in samples -> batch-invariant in tokens. Nodes auto-shrink so
# GBS % (nodes x 4 GPUs x MBS) == 0 (pure DP). Run names get -gbs<g>-t<tok>b
# (+ -lrx<m> for off-1 multipliers), e.g. 270m-moe-muon-lr1e-3-mlr1e-2-gbs512-t15b.
#
set -euo pipefail

FRAMEWORK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)

# ── tuned optima @ 270m-moe (HIDDEN=768), 15B tokens, GBS=128 ────────────────
# matrix / scalar(gains) / embedding / output — keep in sync with sweep-scaling-laws.sh.
ADAMW_MLR=3e-4;   ADAMW_LR=1e-3;   ADAMW_ELR=1e-3;   ADAMW_OLR=1e-3
MUON_MLR=1e-2;    MUON_LR=1e-3;    MUON_ELR=1e-3;    MUON_OLR=1e-3
MASTER_MLR=1e-2;  MASTER_LR=1e-3;  MASTER_ELR=1e-3;  MASTER_OLR=1e-3
MUOWN_MLR=1e-3;   MUOWN_LR=1e-3;   MUOWN_ELR=1e-3;   MUOWN_OLR=1e-3
NORMUON_MLR=1e-2; NORMUON_LR=1e-3; NORMUON_ELR=1e-3; NORMUON_OLR=1e-3

# Batch-LR exponent per recipe: LR x (GBS/REF_GBS)^BSPOW. Override via env.
ADAMW_BSPOW=${ADAMW_BSPOW:-0.5}
MUON_BSPOW=${MUON_BSPOW:-0.5}
MASTER_BSPOW=${MASTER_BSPOW:-0.5}
MUOWN_BSPOW=${MUOWN_BSPOW:-0.5}
NORMUON_BSPOW=${NORMUON_BSPOW:-0.5}

REF_GBS=128
LR_MULTS=${LR_MULTS:-1}      # local check around the rule, e.g. "0.71 1 1.41"

# ── grid ─────────────────────────────────────────────────────────────────────
SIZE=270m-moe
TOKENS=15                    # billions; 15 snaps to 3,584,000 samples (14.68B)
GBS_LIST="32 64 256 512"     # 128 = the existing scaling-law/LR-sweep runs
RECIPES="adamw muon master"
SEQ_LEN=4096

SKIP=""; DRY=""; CLUSTER="mi300"; NODES="4"; TIME=""
while [ $# -gt 0 ]; do
    case "$1" in
        --size)    SIZE="$2"; shift 2 ;;
        --tokens)  TOKENS="$2"; shift 2 ;;
        --gbs)     GBS_LIST="$2"; shift 2 ;;
        --recipes) RECIPES="$2"; shift 2 ;;
        --skip)    SKIP="$2"; shift 2 ;;             # GBS values to skip
        --cluster) CLUSTER="$2"; shift 2 ;;
        --nodes)   NODES="$2"; shift 2 ;;            # max nodes; shrinks per GBS
        --time)    TIME="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# mi300: MBS=4 verified (MBS=8 OOMs in an iter-1 burst, 2026-06-03 bring-up).
if [ -z "${MBS:-}" ]; then
    [ "$CLUSTER" = mi300 ] && MBS=4 || MBS=8
fi
[ "$CLUSTER" = mi300 ] && TIME=${TIME:-24:00:00}
GPUS_PER_NODE=4

if [ "$TOKENS" = 15 ] || [ "$TOKENS" = 14.68 ]; then
    SAMPLES=3584000   # exact budget of the existing 270m-moe LR-sweep runs
else
    SAMPLES=$(awk "BEGIN{printf \"%d\", $TOKENS*1e9/$SEQ_LEN}")
fi

scale() { awk -v b="$1" -v f="$2" -v m="$3" 'BEGIN{printf "%.4g", b*f*m}'; }

submit() {  # submit <env assignments...> -- <recipe>
    local env_args=()
    while [ "$1" != "--" ]; do env_args+=("$1"); shift; done
    shift
    if [ -n "$DRY" ]; then
        echo "DRY: ${env_args[*]} submit.sh --size $SIZE --recipe $1 --nodes $nodes"
    else
        env "${env_args[@]}" bash "$FRAMEWORK_DIR/submit.sh" \
            --size "$SIZE" --recipe "$1" --auto-requeue \
            ${CLUSTER:+--cluster "$CLUSTER"} --nodes "$nodes" \
            ${TIME:+--time "$TIME"}
    fi
}

n=0
for g in $GBS_LIST; do
    case " $SKIP " in *" $g "*) echo ">>> skip GBS=$g"; continue ;; esac

    # pure DP: shrink nodes until GBS divides nodes x GPUS_PER_NODE x MBS.
    nodes=$NODES
    while [ "$nodes" -gt 1 ] && [ $(( g % (nodes * GPUS_PER_NODE * MBS) )) -ne 0 ]; do
        nodes=$(( nodes - 1 ))
    done
    if [ $(( g % (nodes * GPUS_PER_NODE * MBS) )) -ne 0 ]; then
        echo ">>> GBS=$g not divisible by ${nodes}x${GPUS_PER_NODE}xMBS$MBS — skipping" >&2
        continue
    fi

    ITERS=$(( SAMPLES / g ))
    echo ">>> GBS=$g (nodes=$nodes iters=$ITERS tok/step=$(( g * SEQ_LEN / 1000 ))k)"

    for recipe in $RECIPES; do
        var=${recipe^^}
        f=$(awk -v g="$g" -v r="$REF_GBS" -v p="$(eval echo "\$${var}_BSPOW")" 'BEGIN{print (g/r)^p}')
        for mult in $LR_MULTS; do
            TAG=gbs${g}-t${TOKENS}b
            [ "$mult" != 1 ] && TAG=${TAG}-lrx${mult}
            mlr=$(scale "$(eval echo "\$${var}_MLR")" "$f" "$mult")
            elr=$(scale "$(eval echo "\$${var}_ELR")" "$f" "$mult")
            olr=$(scale "$(eval echo "\$${var}_OLR")" "$f" "$mult")
            glr=$(scale "$(eval echo "\$${var}_LR")" "$f" "$mult")
            if [ "$recipe" = master ]; then
                submit MBS="$MBS" GBS="$g" \
                       MLR="$mlr" LR="$glr" ELR="$elr" OLR="$olr" \
                       TRAIN_SAMPLES="$SAMPLES" RUN_TAG="$TAG" -- master
            else
                case $recipe in
                    adamw|muon)    scalar_knob=LR ;;
                    muown|normuon) scalar_knob=SCALAR_LR ;;
                    *) echo "no LR rule for recipe: $recipe" >&2; exit 1 ;;
                esac
                submit MBS="$MBS" GBS="$g" \
                       MATRIX_LR="$mlr" "$scalar_knob=$glr" \
                       EMBEDDING_LR="$elr" OUTPUT_LR="$olr" GAINS_LR="$glr" \
                       TRAIN_SAMPLES="$SAMPLES" RUN_TAG="$TAG" -- "$recipe"
            fi
            n=$((n+1))
        done
    done
done

echo ">>> $n jobs ${DRY:+(dry run, nothing submitted) }for SIZE=$SIZE @ ${TOKENS}B x GBS [$GBS_LIST] x recipes [$RECIPES] (LR_MULTS=[$LR_MULTS])"
