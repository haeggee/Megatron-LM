#!/bin/bash
#
# sweep-length.sh — submit one run per training-length point.
#
# A thin loop over submit.sh: for each token budget it sets TRAIN_SAMPLES and a
# RUN_TAG (so each point gets its own name + checkpoint dir), then hands off.
# Everything else — nodes/time defaults, --auto-requeue, --dry-run — is just
# passed straight through to submit.sh.
#
# Linear decay is the default here on purpose: it decays to MIN_LR over exactly
# TRAIN_SAMPLES, so the schedule auto-scales with each length and there's no
# cooldown to retune per point. Switch with --decay WSD (uses the recipe's WSD
# cooldown) or --decay keep (leave the recipe's own setting untouched).
#
#   bash sweep-length.sh --size 350m-moe --recipe master --tokens "7.5 15 30"
#   bash sweep-length.sh --size 350m-moe --recipe master --tokens "15 30 60" --auto-requeue
#   bash sweep-length.sh --size 350m-moe --recipe master --samples "1831055 3662109"
#   bash sweep-length.sh --size 350m-moe --recipe master --tokens "15 30" --dry-run
#
set -euo pipefail

FRAMEWORK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)

TOKENS=""; SAMPLES=""; DECAY=linear; PASS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --size)    SIZE="$2"; shift 2 ;;
        --recipe)  RECIPE="$2"; shift 2 ;;
        --tokens)  TOKENS="$2"; shift 2 ;;     # space-separated, in BILLIONS
        --samples) SAMPLES="$2"; shift 2 ;;    # space-separated raw sample counts
        --decay)   DECAY="$2"; shift 2 ;;      # linear (default) | WSD | keep
        # passthrough to submit.sh:
        --nodes)        PASS+=(--nodes "$2"); shift 2 ;;
        --time)         PASS+=(--time "$2"); shift 2 ;;
        --auto-requeue) PASS+=(--auto-requeue); shift ;;
        --dry-run)      PASS+=(--dry-run); shift ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

: "${SIZE:?usage: sweep-length.sh --size <size> --recipe <recipe> (--tokens \"7.5 15 30\" | --samples \"N N\") [--decay linear|WSD|keep] [--nodes N] [--time T] [--auto-requeue] [--dry-run]}"
: "${RECIPE:?usage: sweep-length.sh --size <size> --recipe <recipe> (--tokens \"...\" | --samples \"...\") [...]}"
if [ -z "$TOKENS" ] && [ -z "$SAMPLES" ]; then
    echo "give one of --tokens \"7.5 15 30\" (billions) or --samples \"1831055 ...\"" >&2; exit 1
fi
if [ -n "$TOKENS" ] && [ -n "$SAMPLES" ]; then
    echo "give --tokens OR --samples, not both" >&2; exit 1
fi

SIZE_FILE="$FRAMEWORK_DIR/sizes/$SIZE.sh"
[ -f "$SIZE_FILE" ] || { echo "no such size: $SIZE (see $FRAMEWORK_DIR/sizes/)" >&2; exit 1; }
SEQ_LEN=$(grep -E '^SEQ_LEN=' "$SIZE_FILE" | head -1 | cut -d= -f2)
: "${SEQ_LEN:?could not read SEQ_LEN from $SIZE_FILE}"

# Build the list of (samples, tag) points. We always label the tag by token
# budget (more readable than raw samples), computing whichever we weren't given.
POINTS=()   # entries are "SAMPLES:TAG"
if [ -n "$TOKENS" ]; then
    for tok in $TOKENS; do
        s=$(awk "BEGIN{printf \"%d\", $tok*1e9/$SEQ_LEN}")
        tag=$(awk "BEGIN{printf \"t%gb\", $tok}")
        POINTS+=("$s:$tag")
    done
else
    for s in $SAMPLES; do
        tag=$(awk "BEGIN{printf \"t%gb\", $s*$SEQ_LEN/1e9}")
        POINTS+=("$s:$tag")
    done
fi

# How to set the schedule for every point.
DECAY_ENV=()
case "$DECAY" in
    keep) ;;                                   # leave the recipe's own setting
    *)    DECAY_ENV=(LR_DECAY_STYLE="$DECAY") ;;
esac

echo ">>> length sweep: size=$SIZE recipe=$RECIPE decay=$DECAY  (${#POINTS[@]} points, seq_len=$SEQ_LEN)"
for p in "${POINTS[@]}"; do
    s=${p%%:*}; tag=${p##*:}
    echo ">>> --- $tag : TRAIN_SAMPLES=$s ---"
    env "${DECAY_ENV[@]}" TRAIN_SAMPLES="$s" RUN_TAG="$tag" \
        bash "$FRAMEWORK_DIR/submit.sh" --size "$SIZE" --recipe "$RECIPE" "${PASS[@]}"
done
