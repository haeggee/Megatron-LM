#!/bin/bash
#
# sweep-cooldowns.sh — one constant-LR base run, many cooldowns branched off it.
#
# The standard "multiple cooldowns" analysis: train ONE run at constant LR to the
# longest budget, then branch a WSD cooldown off intermediate checkpoints. Each
# branch is a normal framework run whose CKPT_DIR is pre-seeded with a symlink to
# the base run's iter_XXXXXXX checkpoint, so resume / singleton / auto-requeue
# all work unchanged. The branch loads the full optimizer state and the LR
# schedule is replaced via --override-opt-param-scheduler (OVERRIDE_OPT_SCHED=1):
# constant at LR until the branch point, then LR_WSD_DECAY_STYLE down to MIN_LR
# over the cooldown stretch.
#
#   # base run (15B at constant LR) + cooldowns at 6B/9B/15B, 20% cooldown each:
#   bash sweep-cooldowns.sh --size 420m-moe --recipe master \
#       --base-tokens 15 --branches "6 9 15"
#
#   # already have the base run? skip it and just submit cooldowns:
#   bash sweep-cooldowns.sh ... --no-base
#
#   # branch checkpoint doesn't exist yet? poll and launch them as they appear:
#   nohup bash sweep-cooldowns.sh ... --watch &
#
# Sweep a knob across the WHOLE pyramid the usual way: MLR=1e-2 bash sweep-cooldowns.sh ...
#
# Branch points snap to the size's SAVE_INTERVAL (e.g. 420m-moe saves every 3B
# tokens). Cooldown length = --cooldown-frac (default 0.2) x branch budget. Run
# names look like <size>-master-lr...-const-t15b-d6b-cd1.2b, so all sweep points
# coexist in W&B with self-describing lineage.
#
# Cooldowns load per-rank optimizer shards (torch ckpt + distributed optimizer):
# total GPUs of a cooldown MUST match the base run -> --nodes is shared.
#
set -euo pipefail

FRAMEWORK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
CKPT_ROOT=$(cd "$FRAMEWORK_DIR/../../.." && pwd)/_research/results/ckpts

BASE_TOKENS=""; BRANCHES=""; CD_FRAC=0.2; CD_TOKENS=""
NO_BASE=""; WATCH=""; DRY=""; NODES=""; PASS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --size)           SIZE="$2"; shift 2 ;;
        --recipe)         RECIPE="$2"; shift 2 ;;
        --base-tokens)    BASE_TOKENS="$2"; shift 2 ;;   # billions, length of the constant run
        --branches)       BRANCHES="$2"; shift 2 ;;      # space-separated branch points, billions
        --cooldown-frac)  CD_FRAC="$2"; shift 2 ;;       # fraction of the branch budget (default 0.2)
        --cooldown-tokens) CD_TOKENS="$2"; shift 2 ;;    # absolute cooldown length in billions
        --no-base)        NO_BASE=1; shift ;;            # base run already exists or queued
        --watch)          WATCH=1; shift ;;              # poll until all branch ckpts exist
        --dry-run)        DRY=1; shift ;;
        --nodes)          NODES="$2"; PASS+=(--nodes "$2"); shift 2 ;;
        --time)           PASS+=(--time "$2"); shift 2 ;;
        --auto-requeue)   PASS+=(--auto-requeue); shift ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

usage="usage: sweep-cooldowns.sh --size <size> --recipe <recipe> --base-tokens B --branches \"6 9 15\" [--cooldown-frac 0.2 | --cooldown-tokens C] [--no-base] [--watch] [--nodes N] [--time T] [--auto-requeue] [--dry-run]"
: "${SIZE:?$usage}"; : "${RECIPE:?$usage}"
: "${BASE_TOKENS:?$usage}"; : "${BRANCHES:?$usage}"

SIZE_FILE="$FRAMEWORK_DIR/sizes/$SIZE.sh"
[ -f "$SIZE_FILE" ] || { echo "no such size: $SIZE" >&2; exit 1; }
SEQ_LEN=$(grep -E '^SEQ_LEN=' "$SIZE_FILE" | head -1 | cut -d= -f2)
GBS=$(grep -E '^GBS=' "$SIZE_FILE" | head -1 | cut -d= -f2)
SAVE_INTERVAL=$(grep -E '^SAVE_INTERVAL=' "$SIZE_FILE" | head -1 | cut -d= -f2 | awk '{print $1}')
: "${SEQ_LEN:?}" "${GBS:?}" "${SAVE_INTERVAL:?}"

# Same trick as submit.sh: source size+recipe in a subshell to get the run-name
# prefix (size-EXP_TAG-KNOB_STR). Knob env vars (MLR, ...) flow into KNOB_STR so
# every swept point gets its own base run + cooldown family.
NAME_PART=$( { source "$SIZE_FILE"; source "$FRAMEWORK_DIR/recipes/$RECIPE.sh"; } >/dev/null 2>&1; echo "${EXP_TAG:-${RECIPE//\//-}}${KNOB_STR:+-${KNOB_STR}}" ) \
    || { echo "could not derive name from $RECIPE" >&2; exit 1; }

BASE_SAMPLES=$(awk "BEGIN{printf \"%d\", $BASE_TOKENS*1e9/$SEQ_LEN}")
BASE_TAG=$(awk "BEGIN{printf \"const-t%gb\", $BASE_TOKENS}")
BASE_NAME=${SIZE}-${NAME_PART}-${BASE_TAG}
BASE_DIR=$CKPT_ROOT/$BASE_NAME

echo ">>> cooldown sweep: size=$SIZE recipe=$RECIPE base=${BASE_TOKENS}B branches=[$BRANCHES] frac=$CD_FRAC"
echo ">>> base run: $BASE_NAME (save every $SAVE_INTERVAL iters = $(awk "BEGIN{printf \"%.2f\", $SAVE_INTERVAL*$GBS*$SEQ_LEN/1e9}")B tokens)"

# ── base run: constant LR over the longest budget ────────────────────────────
if [ -z "$NO_BASE" ]; then
    echo ">>> --- base: TRAIN_SAMPLES=$BASE_SAMPLES, constant LR ---"
    [ -n "$DRY" ] || env LR_DECAY_STYLE=constant TRAIN_SAMPLES="$BASE_SAMPLES" RUN_TAG="$BASE_TAG" \
        bash "$FRAMEWORK_DIR/submit.sh" --size "$SIZE" --recipe "$RECIPE" --auto-requeue "${PASS[@]}"
fi

# ── branch points: snap to SAVE_INTERVAL, compute cooldown stretch ───────────
SUBMITTED=()
submit_branch() {
    local tok=$1
    local want_iters snap_iters branch_samples cd_samples total tag
    want_iters=$(awk "BEGIN{printf \"%d\", $tok*1e9/$SEQ_LEN/$GBS}")
    snap_iters=$(( (want_iters + SAVE_INTERVAL/2) / SAVE_INTERVAL * SAVE_INTERVAL ))
    [ "$snap_iters" -ge "$SAVE_INTERVAL" ] || snap_iters=$SAVE_INTERVAL
    branch_samples=$(( snap_iters * GBS ))
    if [ -n "$CD_TOKENS" ]; then
        cd_samples=$(awk "BEGIN{printf \"%d\", $CD_TOKENS*1e9/$SEQ_LEN}")
    else
        cd_samples=$(awk "BEGIN{printf \"%d\", $branch_samples*$CD_FRAC}")
    fi
    total=$(( branch_samples + cd_samples ))
    tag=$(awk "BEGIN{printf \"%s-d%.3gb-cd%.3gb\", \"$BASE_TAG\", $branch_samples*$SEQ_LEN/1e9, $cd_samples*$SEQ_LEN/1e9}")

    local iter_dir ckpt_dir
    iter_dir=$(printf 'iter_%07d' "$snap_iters")
    ckpt_dir=$CKPT_ROOT/${SIZE}-${NAME_PART}-${tag}

    echo ">>> --- $tag : branch@iter $snap_iters ($branch_samples samples) + cooldown $cd_samples -> TRAIN_SAMPLES=$total ---"
    [ -n "$DRY" ] && return 0

    [ -d "$BASE_DIR/$iter_dir" ] || return 1   # base run hasn't reached this point yet

    # Seed CKPT_DIR (idempotent; never touch it once the cooldown made progress).
    mkdir -p "$ckpt_dir"
    [ -e "$ckpt_dir/$iter_dir" ] || ln -s "$BASE_DIR/$iter_dir" "$ckpt_dir/$iter_dir"
    [ -f "$ckpt_dir/latest_checkpointed_iteration.txt" ] || echo "$snap_iters" > "$ckpt_dir/latest_checkpointed_iteration.txt"

    env LR_DECAY_STYLE=WSD LR_WSD_DECAY_SAMPLES="$cd_samples" TRAIN_SAMPLES="$total" \
        OVERRIDE_OPT_SCHED=1 RUN_TAG="$tag" \
        bash "$FRAMEWORK_DIR/submit.sh" --size "$SIZE" --recipe "$RECIPE" "${PASS[@]}"
}

PENDING=($BRANCHES)
while :; do
    REMAINING=()
    for tok in "${PENDING[@]}"; do
        submit_branch "$tok" || REMAINING+=("$tok")
    done
    PENDING=("${REMAINING[@]:-}")
    [ -z "${PENDING[*]// }" ] && break
    if [ -z "$WATCH" ]; then
        echo ">>> branch checkpoints not yet available: ${PENDING[*]} — rerun later or use --watch" >&2
        exit 0
    fi
    echo ">>> waiting on branch checkpoints: ${PENDING[*]} (poll every 10 min)"
    sleep 600
done
echo ">>> all cooldowns submitted."
