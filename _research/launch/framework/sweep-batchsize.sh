#!/bin/bash
#
# sweep-batchsize.sh — batch-size scaling per optimizer at a FIXED step count.
#
# Protocol: hold total optimizer steps constant (default 28,000 = the tuned
# 270m-moe @ 15B / GBS=128 runs), grow GBS, and raise the LR with it. Tokens
# scale linearly with GBS (TRAIN_SAMPLES = STEPS x GBS), so each rung costs
# GBS/128 x the reference run — this measures how much extra loss-per-step an
# optimizer extracts from bigger batches when LR follows the rule:
#
#   LR x (GBS/REF_GBS)^BSPOW        (sqrt by default — Adam heuristic; for
#                                    muon/master the exponent IS the experiment)
#
# Non-master weight decay follows WD = 0.1 x sqrt(b/l) (b = GBS/128, l =
# tokens/ref) — the same rule as sweep-scaling-laws. At fixed steps l = b, so
# it cancels to 0.1; it only bites if --steps moves off REF_STEPS.
#
#   bash sweep-batchsize.sh --dry-run
#   bash sweep-batchsize.sh --gbs "256 512" --recipes master
#   LR_MULTS="0.71 1 1.41" bash sweep-batchsize.sh --gbs 512   # rule check at the edge
#   MASTER_BSPOW=0 bash sweep-batchsize.sh ...             # batch-invariant master LR
#
# Warmup is pinned to 1000 steps for the warmup-using recipes
# (adamw/muon/normuon) by passing LR_WARMUP_SAMPLES = 1000 x GBS — at GBS=128
# that's the recipes' own 128000-sample default. Nodes auto-shrink so
# GBS % (nodes x 4 GPUs x MBS) == 0 (pure DP). Run names get -gbs<g>-s<steps>
# (+ -lrx<m> for off-1 multipliers), e.g. 270m-moe-muon-lr…-mlr…-gbs256-s28k.
#
set -euo pipefail

FRAMEWORK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)

# ── tuned optima @ 270m-moe (HIDDEN=768), 28k steps, GBS=128 ─────────────────
# matrix / scalar(gains) / embedding / output — keep in sync with sweep-scaling-laws.sh.
ADAMW_MLR=2.4e-3;   ADAMW_LR=1e-3;   ADAMW_ELR=1e-3;   ADAMW_OLR=1e-3
MUON_MLR=5e-3;    MUON_LR=1e-3;    MUON_ELR=1e-3;    MUON_OLR=1e-3
MASTER_MLR=1e-2;  MASTER_LR=1e-3;  MASTER_ELR=1e-3;  MASTER_OLR=1e-3
MUOWN_MLR=2.4e-3;   MUOWN_LR=1e-3;   MUOWN_ELR=1e-3;   MUOWN_OLR=1e-3
NORMUON_MLR=1e-2; NORMUON_LR=1e-3; NORMUON_ELR=1e-3; NORMUON_OLR=1e-3

# Batch-LR exponent per recipe: LR x (GBS/REF_GBS)^BSPOW. Override via env.
ADAMW_BSPOW=${ADAMW_BSPOW:-0.5}
MUON_BSPOW=${MUON_BSPOW:-0.5}
MASTER_BSPOW=${MASTER_BSPOW:-0.5}
MUOWN_BSPOW=${MUOWN_BSPOW:-0.5}
NORMUON_BSPOW=${NORMUON_BSPOW:-0.5}

REF_GBS=128
REF_STEPS=28000              # WD reference: l = STEPS*GBS / (REF_STEPS*REF_GBS)
WD_BASE=0.1                  # non-master; WD = WD_BASE*sqrt(b/l), no-op at fixed steps
WARMUP_STEPS=1000            # fixed in steps, not samples: LR_WARMUP_SAMPLES = 1000*GBS
LR_MULTS=${LR_MULTS:-1}      # local check around the rule, e.g. "0.71 1 1.41"

# ── grid ─────────────────────────────────────────────────────────────────────
SIZE=270m-moe
STEPS=28000                  # = 3,584,000 samples / GBS 128: the tuned runs
GBS_LIST="192 256"           # 128 = the existing scaling-law/LR-sweep runs
RECIPES="muon master"
SEQ_LEN=4096

SKIP=""; DRY=""; CLUSTER="mi300"; NODES="4"; TIME=""
while [ $# -gt 0 ]; do
    case "$1" in
        --size)    SIZE="$2"; shift 2 ;;
        --steps)   STEPS="$2"; shift 2 ;;
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

STEPS_TAG=$(awk "BEGIN{printf \"s%gk\", $STEPS/1000}")

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

    SAMPLES=$(( STEPS * g ))
    TOK_B=$(awk "BEGIN{printf \"%.2f\", $SAMPLES*$SEQ_LEN/1e9}")
    # WD ~ sqrt(b/l); b = g/REF_GBS, l = b*STEPS/REF_STEPS, so this reduces to
    # sqrt(REF_STEPS/STEPS) — constant 0.1 at the default step count.
    wd=$(awk -v w="$WD_BASE" -v g="$g" -v rg="$REF_GBS" -v s="$STEPS" -v rs="$REF_STEPS" \
        'BEGIN{b=g/rg; l=s*g/(rs*rg); printf "%.4g", w*sqrt(b/l)}')
    echo ">>> GBS=$g (nodes=$nodes steps=$STEPS samples=$SAMPLES tokens=${TOK_B}B tok/step=$(( g * SEQ_LEN / 1000 ))k wd=$wd)"

    for recipe in $RECIPES; do
        var=${recipe^^}
        f=$(awk -v g="$g" -v r="$REF_GBS" -v p="$(eval echo "\$${var}_BSPOW")" 'BEGIN{print (g/r)^p}')
        for mult in $LR_MULTS; do
            TAG=gbs${g}-${STEPS_TAG}
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
                # pin warmup to WARMUP_STEPS steps; muown has no warmup, leave it 0
                warmup_env=()
                [ "$recipe" != muown ] && warmup_env=(LR_WARMUP_SAMPLES=$(( WARMUP_STEPS * g )))
                submit MBS="$MBS" GBS="$g" \
                       MATRIX_LR="$mlr" "$scalar_knob=$glr" \
                       EMBEDDING_LR="$elr" OUTPUT_LR="$olr" GAINS_LR="$glr" \
                       WEIGHT_DECAY="$wd" ${warmup_env[@]+"${warmup_env[@]}"} \
                       TRAIN_SAMPLES="$SAMPLES" RUN_TAG="$TAG" -- "$recipe"
            fi
            n=$((n+1))
        done
    done
done

echo ">>> $n jobs ${DRY:+(dry run, nothing submitted) }for SIZE=$SIZE @ $STEPS steps x GBS [$GBS_LIST] x recipes [$RECIPES] (LR_MULTS=[$LR_MULTS])"
