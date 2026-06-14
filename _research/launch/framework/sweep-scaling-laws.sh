#!/bin/bash
#
# sweep-scaling-laws.sh — optimizer scaling-law grid: MoE ladder x token budgets.
#
# Dedicated runs, GBS fixed (128 in all sizes), linear decay to MIN_LR over the
# full budget. Per-optimizer LRs derive from the optimum tuned at 270m-moe
# (HIDDEN=768) / 15B tokens — set them once in the block below; the rules do
# the rest:
#
#   non-master:  matrix / embedding / output LR ~ 1/k     (k = HIDDEN/768, muP)
#                gains LR width-independent
#                ALL LRs ~ 1/sqrt(l)                      (l = tokens/15B)
#                weight decay ~ k / sqrt(l) off the 0.1 default
#   master:      all LRs width-independent, ~ 1/l^0.25; WD stays 0 (recipe)
#
#   bash sweep-scaling-laws.sh --dry-run                  # full grid: 5 recipes x 9
#   bash sweep-scaling-laws.sh --recipes "master muown" --sizes 810m-moe --tokens 65
#   bash sweep-scaling-laws.sh --skip "270m-moe:15"       # already trained
#
# Token grid {15, 23.3, 44.15}B aligns budgets across sizes; ~55 tok/active-param
# diagonal = 270m@15, 420m@23.3, 810m@44.15. Run names self-describe: t<budget>b
# tag + per-class LRs in KNOB_STR, e.g.
# 420m-moe-master-lr8.46e-4-mlr6.77e-3-elr8.46e-4-t23.3b.
#
set -euo pipefail

FRAMEWORK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)

# ── tuned optima @ 270m-moe (HIDDEN=768), 15B tokens ─────────────────────────
# matrix / scalar(gains) / embedding / output. Fill in from the 270m LR sweep.
ADAMW_MLR=2.4e-3;   ADAMW_LR=1e-3;   ADAMW_ELR=1e-3;   ADAMW_OLR=1e-3
MUON_MLR=5e-3;    MUON_LR=1e-3;    MUON_ELR=1e-3;    MUON_OLR=1e-3
MASTER_MLR=1e-2;  MASTER_LR=1e-3;  MASTER_ELR=1e-3;  MASTER_OLR=1e-3


MUOWN_MLR=2.4e-3;   MUOWN_LR=1e-3;   MUOWN_ELR=1e-3;   MUOWN_OLR=1e-3
NORMUON_MLR=1e-2; NORMUON_LR=1e-3; NORMUON_ELR=1e-3; NORMUON_OLR=1e-3

REF_HIDDEN=768
REF_TOKENS=14.68
WD_BASE=0.1       # non-master recipes; scaled ~ k/sqrt(l) (LR~1/k cancels in LR*WD)

# ── grid ─────────────────────────────────────────────────────────────────────
# SIZES="270m-moe 420m-moe 600m-moe 810m-moe"
SIZES="270m-moe 420m-moe"
# tokens are set to be ~55 tok/active-param diagonal
# 270m@15, 420m@23.3, 600m@31.8, 810m@44.15
TOKENS="7.5 14.68"   # billions; 15 snaps to 3,584,000 samples (14.68B)
# RECIPES="adamw muon master"
RECIPES="muon master"
SEQ_LEN=4096                 # constant across ladder

SKIP=""; DRY=""; CLUSTER="mi300"; NODES="4"; PASS=(); TIME="";
while [ $# -gt 0 ]; do
    case "$1" in
        --sizes)   SIZES="$2"; shift 2 ;;
        --tokens)  TOKENS="$2"; shift 2 ;;
        --recipes) RECIPES="$2"; shift 2 ;;
        --skip)    SKIP="$2"; shift 2 ;;             # "size:tokens ..." pairs
        --cluster) CLUSTER="$2"; shift 2 ;;
        --nodes)   NODES="$2"; shift 2 ;;
        --time)    TIME="$2"; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# mi300: MBS=4 verified (MBS=8 OOMs in an iter-1 burst, 2026-06-03 bring-up),
# 810m halves it; alps3 (GH200) takes 2x the mi300 setting. Env MBS overrides.
mbs_for() {  # mbs_for <size>
    local m=4
    [[ "$1" == 810m-moe* ]] && m=2
    [ "$CLUSTER" != mi300 ] && m=$(( m * 2 ))
    echo "$m"
}
[ "$CLUSTER" = mi300 ] && TIME=${TIME:-24:00:00}

scale() { awk -v b="$1" -v k="$2" -v l="$3" -v p="$4" 'BEGIN{printf "%.4g", b/k*l^-p}'; }

submit() {  # submit <env assignments...> -- <recipe>
    local env_args=()
    while [ "$1" != "--" ]; do env_args+=("$1"); shift; done
    shift
    if [ -n "$DRY" ]; then
        echo "DRY: ${env_args[*]} submit.sh --size $size --recipe $1"
    else
        env "${env_args[@]}" bash "$FRAMEWORK_DIR/submit.sh" \
            --size "$size" --recipe "$1" --auto-requeue \
            ${CLUSTER:+--cluster "$CLUSTER"} ${NODES:+--nodes "$NODES"} \
            ${TIME:+--time "$TIME"} \
            ${PASS[@]+"${PASS[@]}"}
    fi
}

n=0
for size in $SIZES; do
    SIZE_FILE="$FRAMEWORK_DIR/sizes/$size.sh"
    [ -f "$SIZE_FILE" ] || { echo "no such size: $size" >&2; exit 1; }
    HIDDEN=$(grep -E '^HIDDEN=' "$SIZE_FILE" | head -1 | cut -d= -f2)
    GBS=$( (source "$SIZE_FILE" >/dev/null 2>&1; echo "$GBS") )   # resolves ${GBS:-128}
    k=$(awk "BEGIN{print $HIDDEN/$REF_HIDDEN}")
    MBS_RUN=${MBS:-$(mbs_for "$size")}

    for tok in $TOKENS; do
        case " $SKIP " in *" $size:$tok "*)
            echo ">>> skip $size @ ${tok}B"; continue ;;
        esac
        l=$(awk "BEGIN{print $tok/$REF_TOKENS}")
        if [ "$tok" = 15 ] || [ "$tok" = 14.68 ]; then
            SAMPLES=3584000   # exact budget of the existing 270m-moe LR-sweep runs
        else
            SAMPLES=$(awk "BEGIN{printf \"%d\", $tok*1e9/$SEQ_LEN}")
        fi
        ITERS=$(( SAMPLES / GBS ))
        TAG=mi300-t${tok}b
   
   
        echo ">>> $size @ ${tok}B (k=$k l=$l samples=$SAMPLES iters=$ITERS)"

        for recipe in $RECIPES; do
            if [ "$recipe" = master ]; then
                # width-independent; everything ~ 1/l^0.25
                submit MBS="$MBS_RUN" \
                       MLR="$(scale "$MASTER_MLR" 1 "$l" 0.25)" \
                       LR="$(scale "$MASTER_LR" 1 "$l" 0.25)" \
                       ELR="$(scale "$MASTER_ELR" 1 "$l" 0.25)" \
                       OLR="$(scale "$MASTER_OLR" 1 "$l" 0.25)" \
                       TRAIN_SAMPLES="$SAMPLES" RUN_TAG="$TAG" -- master
            else
                # matrix/embedding/output ~ 1/k, gains width-free; all ~ 1/sqrt(l)
                # WD goes the other way in width: ~ k/sqrt(l)
                var=${recipe^^}
                mlr=$(scale "$(eval echo "\$${var}_MLR")" "$k" "$l" 0.5)
                elr=$(scale "$(eval echo "\$${var}_ELR")" "$k" "$l" 0.5)
                olr=$(scale "$(eval echo "\$${var}_OLR")" "$k" "$l" 0.5)
                glr=$(scale "$(eval echo "\$${var}_LR")" 1 "$l" 0.5)
                wd=$(awk -v b="$WD_BASE" -v k="$k" -v l="$l" 'BEGIN{printf "%.4g", b*k/sqrt(l)}')
                case $recipe in
                    adamw)         scalar_knob=LR ;;
                    muon)          scalar_knob=LR ;;
                    muown|normuon) scalar_knob=SCALAR_LR ;;
                    *) echo "no LR rule for recipe: $recipe" >&2; exit 1 ;;
                esac
                submit MBS="$MBS_RUN" \
                       MATRIX_LR="$mlr" "$scalar_knob=$glr" \
                       EMBEDDING_LR="$elr" OUTPUT_LR="$olr" GAINS_LR="$glr" \
                       WEIGHT_DECAY="$wd" \
                       TRAIN_SAMPLES="$SAMPLES" RUN_TAG="$TAG" -- "$recipe"
            fi
            n=$((n+1))
        done
    done
done

echo ">>> $n jobs ${DRY:+(dry run, nothing submitted) }across sizes [$SIZES] x tokens [$TOKENS] x recipes [$RECIPES]"
