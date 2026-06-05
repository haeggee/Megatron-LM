#!/bin/bash
#
# sweep-master-length-transfer.sh — matrix-LR sweep per training length:
# does master's optimal matrix LR transfer across training lengths?
#
# A thin grid over submit.sh: for each (iters, MLR) point it sets
# TRAIN_SAMPLES = iters*GBS plus a RUN_TAG, exports MLR=<val>, and hands off.
# Unlike sweep-lr.sh (which sweeps LR — for master that's the SCALAR group),
# this sweeps the matrix LR only; scalar/embedding LRs stay at the recipe
# defaults (override per-point with LR=/ELR= if needed). master folds MLR into
# KNOB_STR (lr…-mlr…-elr…) and the length + exponent land in RUN_TAG, so every
# point gets its own EXP_NAME -> own checkpoint dir + wandb run, e.g.
#   150m-master-lr0.000841-mlr8e-3-elr0.000841-20k-p0.25
#
# Linear decay (master's default) decays to MIN_LR over exactly TRAIN_SAMPLES,
# so the schedule auto-scales with each length point — no cooldown to retune.
#
# --powers sweeps the LR-vs-length scaling hypothesis: for each exponent p
# EVERY LR class is scaled as x' = x / k^p with k = iters/10000 (10k is the
# reference length; k<1 scales UP) — the matrix LR (so --mlrs gives the grid
# AT 10k), the scalar LR, and ELR, which follows LR via master's
# ELR=${ELR:-$LR} default. At k=1 every p arm is the same run, so
# duplicate (length, MLR, lr') points are submitted only once, tagged with the
# first p that produced them (the 10k point of every p curve is the -p0 run).
#
# Defaults to the 150m quick-run rung + master recipe with a geometric x2 grid
# centered on master's MLR default (8e-3) at the size file's default length;
# 150m.sh routes these runs to the len-ablation- wandb project.
#
#   bash sweep-master-length-transfer.sh                            # 150m, master, default grid @ default length
#   bash sweep-master-length-transfer.sh --iters "5000 10000 20000" # the actual transfer grid
#   bash sweep-master-length-transfer.sh --mlrs "4e-3 8e-3 1.6e-2" --iters "10000"
#   bash sweep-master-length-transfer.sh --iters "10000 20000 40000" --powers "0 0.25 0.5"
#   bash sweep-master-length-transfer.sh --dry-run                  # list submits, don't launch
#
set -euo pipefail

FRAMEWORK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)

SIZE=150m; RECIPE=master
MBS=8
MLRS="1e-3 5e-3 1e-2 2e-2 4e-2"
LR=1e-3
ITERS=""      # empty -> single point at the size default
POWERS="0"    # LR-vs-length exponents p in lr' = LR / (iters/10k)^p
CLUSTER=mi300
PASS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --size)    SIZE="$2"; shift 2 ;;
        --recipe)  RECIPE="$2"; shift 2 ;;
        --mbs)     MBS="$2"; shift 2 ;;
        --lr)      LR="$2"; shift 2 ;;
        --mlrs)    MLRS="$2"; shift 2 ;;       # space-separated matrix-LR values
        --iters)   ITERS="$2"; shift 2 ;;      # space-separated training lengths in iterations
        --powers)  POWERS="$2"; shift 2 ;;     # space-separated p values, e.g. "0 0.25 0.5"
        # passthrough to submit.sh:
        --cluster)      CLUSTER="$2"; shift 2 ;;
        --nodes)        PASS+=(--nodes "$2"); shift 2 ;;
        --time)         PASS+=(--time "$2"); shift 2 ;;
        --auto-requeue) PASS+=(--auto-requeue); shift ;;
        --dry-run)      PASS+=(--dry-run); shift ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

SIZE_FILE="$FRAMEWORK_DIR/sizes/$SIZE.sh"
[ -f "$SIZE_FILE" ] || { echo "no such size: $SIZE (see $FRAMEWORK_DIR/sizes/)" >&2; exit 1; }

# GBS converts iters <-> samples; size files set it as a plain literal.
GBS=$(grep -E '^GBS=' "$SIZE_FILE" | head -1 | cut -d= -f2)
: "${GBS:?could not read GBS from $SIZE_FILE}"

# Build the list of (samples, tag) length points. The tag is the iteration
# count ("10k"), appended to EXP_NAME via RUN_TAG. With no --iters we still
# tag the run with the effective default length (env TRAIN_SAMPLES override,
# else the size file's `TRAIN_SAMPLES=${TRAIN_SAMPLES:-N}` default).
POINTS=()   # entries are "SAMPLES:TAG"
if [ -n "$ITERS" ]; then
    for it in $ITERS; do
        POINTS+=("$((it * GBS)):$(awk "BEGIN{printf \"%gk\", $it/1000}")")
    done
else
    s=${TRAIN_SAMPLES:-$(sed -n 's/^TRAIN_SAMPLES=${TRAIN_SAMPLES:-\([0-9]*\)}.*/\1/p' "$SIZE_FILE")}
    : "${s:?could not read the TRAIN_SAMPLES default from $SIZE_FILE (give --iters)}"
    POINTS+=("$s:$(awk "BEGIN{printf \"%gk\", $s/$GBS/1000}")")
fi

echo ">>> length-transfer sweep: size=$SIZE recipe=$RECIPE cluster=$CLUSTER  (MLRs: $MLRS | lengths: ${POINTS[*]##*:} | powers: $POWERS)"
declare -A SUBMITTED=()
for p in "${POINTS[@]}"; do
    s=${p%%:*}; tag=${p##*:}
    for pw in $POWERS; do
        # x' = x / k^p, k = iters/10k, applied to every LR class. The scaled
        # values land in KNOB_STR (ELR follows LR), so each surviving point
        # names itself. Keep the canonical spelling when unscaled.
        lr=$(awk "BEGIN{ printf \"%.3g\", $LR / (($s/$GBS/10000) ^ $pw) }")
        awk "BEGIN{ exit !($lr == $LR) }" && lr=$LR
        for mlr0 in $MLRS; do
            mlr=$(awk "BEGIN{ printf \"%.3g\", $mlr0 / (($s/$GBS/10000) ^ $pw) }")
            awk "BEGIN{ exit !($mlr == $mlr0) }" && mlr=$mlr0
            key="$s|$mlr|$lr"
            if [ -n "${SUBMITTED[$key]:-}" ]; then
                echo ">>> --- iters=$tag p=$pw MLR=$mlr LR=$lr : same as an earlier point (k=1 arms coincide), skipping ---"
                continue
            fi
            SUBMITTED[$key]=1
            echo ">>> --- iters=$tag p=$pw MLR=$mlr0->$mlr LR=$LR->$lr ---"
            MBS="$MBS" LR="$lr" MLR="$mlr" TRAIN_SAMPLES="$s" RUN_TAG="$tag-p$pw" \
                bash "$FRAMEWORK_DIR/submit.sh" --cluster "$CLUSTER" --size "$SIZE" --recipe "$RECIPE" \
                ${PASS[@]+"${PASS[@]}"}
        done
    done
done
