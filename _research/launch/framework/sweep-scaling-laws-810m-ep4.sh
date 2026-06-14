#!/bin/bash
#
# sweep-scaling-laws-810m-ep4.sh — MoE scaling-law token sweep, EP=4.
#
# Same grid and per-optimizer LR rules as sweep-scaling-laws.sh, but run with
# expert-model-parallel-size 4 instead of the baseline pure-DP (EP=1). Defaults
# to the 810m-moe rung; pass `--size 600m-moe` (or any 64-expert rung) to sweep a
# different one. With 64 routed experts, EP=4 shards them 16/rank, so the global
# GPU count must stay divisible by 4 (4 nodes x 4 GPUs = 16 -> DP=4); 64 % 4 = 0
# holds for every rung in the ladder.
#
# Why a dedicated file rather than `EP=4 bash sweep-scaling-laws.sh`:
#   * runs are tagged `ep4` so EXP_NAME/CKPT_DIR never collide with the EP=1
#     baselines of the same recipe+token budget on the shared filesystem;
#   * the MoE dispatcher defaults to `allgather` (same as the EP=1 baselines and
#     the cluster default), so EP runs stay loss-comparable to them; set env
#     MOE_DISPATCHER=alltoall to switch to the standard EP token-routing path
#     (the one clusters/mi300.sh notes as verified on ROCm) if needed.
# The size file (sizes/810m-moe.sh) already honors env EP, so EP travels through
# submit.sh -> sbatch --export=ALL -> train.sbatch unchanged.
#
#   bash sweep-scaling-laws-810m-ep4.sh --dry-run            # recipes x token grid
#   bash sweep-scaling-laws-810m-ep4.sh --recipes master --tokens 44
#   bash sweep-scaling-laws-810m-ep4.sh --size 600m-moe      # the in-between rung
#   bash sweep-scaling-laws-810m-ep4.sh --skip "810m-moe:44" # already trained
#   EP=8 bash sweep-scaling-laws-810m-ep4.sh                 # override shard count
#
set -euo pipefail

FRAMEWORK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)

# Expert parallelism + dispatcher. Exported so they flow through `env ... bash
# submit.sh` -> --export=ALL into train.sbatch's source of sizes/810m-moe.sh
# (which reads EP=${EP:-1}) and lib/common.sh (MOE_DISPATCHER default).
export EP=${EP:-4}
export MOE_DISPATCHER=${MOE_DISPATCHER:-allgather}

# ── tuned optima @ 270m-moe (HIDDEN=768), 15B tokens ─────────────────────────
# matrix / scalar(gains) / embedding / output. Mirror sweep-scaling-laws.sh.
ADAMW_MLR=2.4e-3;   ADAMW_LR=1e-3;   ADAMW_ELR=1e-3;   ADAMW_OLR=1e-3
MUON_MLR=5e-3;    MUON_LR=1e-3;    MUON_ELR=1e-3;    MUON_OLR=1e-3
MASTER_MLR=1e-2;  MASTER_LR=1e-3;  MASTER_ELR=1e-3;  MASTER_OLR=1e-3
MUOWN_MLR=2.4e-3;   MUOWN_LR=1e-3;   MUOWN_ELR=1e-3;   MUOWN_OLR=1e-3
NORMUON_MLR=1e-2; NORMUON_LR=1e-3; NORMUON_ELR=1e-3; NORMUON_OLR=1e-3

REF_HIDDEN=768
REF_TOKENS=14.68
WD_BASE=0.1       # non-master recipes; scaled ~ k/sqrt(l) (LR~1/k cancels in LR*WD)

# ── grid ─────────────────────────────────────────────────────────────────────
SIZE="${SIZE:-810m-moe}"     # default rung; --size overrides (e.g. 600m-moe)
# tokens are set to be ~55 tok/active-param diagonal; 810m diagonal point = 44.15B
# (600m diagonal = 31.8B)
TOKENS="7.5 14.68"         # billions; 15 snaps to 3,584,000 samples (14.68B)
RECIPES="adamw muon master"
SEQ_LEN=4096                 # constant across ladder

SKIP=""; DRY=""; CLUSTER="mi300"; NODES="4"; PASS=(); TIME="";
while [ $# -gt 0 ]; do
    case "$1" in
        --size)    SIZE="$2"; shift 2 ;;
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

# EP must divide the global GPU count (4 GPUs/node here) and 64 experts.
GPUS=$(( NODES * 4 ))
[ $(( GPUS % EP )) -eq 0 ] || { echo "EP=$EP must divide GPUS=$GPUS (nodes=$NODES x 4)" >&2; exit 1; }
[ $(( 64 % EP )) -eq 0 ]   || { echo "EP=$EP must divide the 64 routed experts" >&2; exit 1; }

# mi300: MBS=4 verified (MBS=8 OOMs in an iter-1 burst, 2026-06-03 bring-up),
# 810m halves it; alps3 (GH200) takes 2x the mi300 setting. EP shards experts so
# there is usually headroom to raise MBS, but keep the conservative default and
# let env MBS override. Env MBS overrides.
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
        echo "DRY: EP=$EP MOE_DISPATCHER=$MOE_DISPATCHER ${env_args[*]} submit.sh --size $size --recipe $1"
    else
        env "${env_args[@]}" bash "$FRAMEWORK_DIR/submit.sh" \
            --size "$size" --recipe "$1" --auto-requeue \
            ${CLUSTER:+--cluster "$CLUSTER"} ${NODES:+--nodes "$NODES"} \
            ${TIME:+--time "$TIME"} \
            ${PASS[@]+"${PASS[@]}"}
    fi
}

n=0
size="$SIZE"
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
    TAG=ep${EP}-n${NODES}-t${tok}b  # ep+nodes pin the DP world size (NODES*4/EP);
                                    # the torch-format dist-opt ckpt is sharded to
                                    # that layout and can't reshard, so a different
                                    # node count must not share a CKPT_DIR. CLUSTER_TAG
                                    # (in EXP_NAME) already separates clusters.

    echo ">>> $size @ ${tok}B EP=$EP (k=$k l=$l samples=$SAMPLES iters=$ITERS)"

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

echo ">>> $n jobs ${DRY:+(dry run, nothing submitted) }for $SIZE EP=$EP x tokens [$TOKENS] x recipes [$RECIPES]"
