#!/bin/bash
#
# warm-restart.sh — continue a finished run from one of its checkpoints under a
# fresh LR schedule, crossing optimizer-state reuse with a warm-restart warmup.
#
# Given a SOURCE run that saved iter_<branch> (e.g. the 10k checkpoint of
# 150m-master-...-10k), this launches up to four continuation runs that each
# train --cont-iters MORE steps with a peak->min linear anneal, differing only
# along two axes:
#
#     opt state:   REUSE the source optimizer moments  vs.  FRESH moments
#     warmup:      jump straight to peak LR             vs.  --warmup-iters ramp
#
#   arm 1  reopt-nowarm   reuse moments, no warmup   (jump to peak, anneal)
#   arm 2  reopt-warmNNN  reuse moments, NNN-step warmup then anneal
#   arm 3  newopt-nowarm  fresh moments, no warmup   (jump to peak, anneal)
#   arm 4  newopt-warmNNN fresh moments, NNN-step warmup then anneal
#
# How each axis is realized (see lib/common.sh + megatron OptimizerParamScheduler):
#
#   REUSE moments  -> a normal in-place RESUME: CKPT_DIR is pre-seeded with a
#       symlink to the source iter_<branch> (+ tracker), so Megatron loads the
#       model, the (layer-wise distributed) optimizer state, and the scheduler,
#       continuing the global step counter at <branch>. OVERRIDE_OPT_SCHED=1
#       replaces the stored schedule with the args below. The data loader and
#       counter continue from the branch point.
#         * no-warmup arm: LR_WARMUP_SAMPLES = branch_samples, so the warmup
#           boundary lands exactly on the resume step -> LR is at peak there and
#           anneals from there. (The pre-branch ramp is never trained.)
#         * warmup arm: LR_WARMUP_START_SAMPLES = branch_samples and
#           LR_WARMUP_SAMPLES = branch_samples + warmup_samples, so the ramp
#           begins AT the resume step and runs init->peak over warmup_samples,
#           then anneals. This "shifted warmup" is the only way to warm up from
#           a low LR while the global counter continues (a plain warmup is
#           anchored at step 0 and would already be at ~peak by the branch).
#
#   FRESH moments  -> the SAME seeded full resume as the reuse arms (model,
#       LEARNABLE GAINS, scheduler and data position all loaded), plus
#       RESET_OPT_MOMENTS=1: after the load, Megatron zeros the Adam/momentum
#       buffers (exp_avg/exp_avg_sq/exp_avg_slow, NorMuon v, the gains' own m/v)
#       and the step, while KEEPING the gains. The gains live in optimizer state
#       and are baked into the saved weight (p = normalize(w)*phi(gains)) but
#       can't be recovered from the weight alone, so a plain fresh optimizer
#       (--no-load-optim) would silently reset them to identity — hence we load
#       then selectively zero. Net: gains reused exactly, momentum genuinely
#       fresh. (See megatron --reset-optimizer-moments.)
#
# This makes a clean 2x2: ALL four arms continue the same data stream from the
# branch, keep the learnable gains, and have matched LR trajectories (same peak,
# anneal length and warmup length); they differ ONLY in {opt momentum reused vs
# fresh} x {warmup vs not}.
#
# Optimizer-state shards are per-DP-rank, so ALL arms MUST use the same node/GPU
# count as the source run (--nodes; defaults to the size's DEFAULT_NODES, which
# is what the source used). These runs fit one allocation, so no --auto-requeue:
# its chained jobs would drop the OVERRIDE_OPT_SCHED / warmup env, AND
# RESET_OPT_MOMENTS re-zeros the momentum on every load (so a mid-run resume
# would wipe accumulated momentum). If a job dies, delete its CKPT_DIR and
# relaunch that arm.
#
# Recipe knobs that defined the source run (LR/MLR/ELR/MBS/...) MUST be supplied
# in the env so the per-group peak LRs AND the run-name prefix match the source:
#
#   LR=1e-3 MLR=2e-2 ELR=1e-3 MBS=8 bash warm-restart.sh \
#       --src 150m-master-lr1e-3-mlr2e-2-elr1e-3-mi300-10k \
#       --size 150m --recipe master --cluster mi300 \
#       --cont-iters 10000 --warmup-iters 500
#
#   # inspect without launching:
#   ... bash warm-restart.sh ... --dry-run
#   # launch only some arms:
#   ... bash warm-restart.sh ... --arms "2 4"
#
set -euo pipefail

FRAMEWORK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
REPO_DIR=$(cd "$FRAMEWORK_DIR/../../.." && pwd)
CKPT_ROOT=$REPO_DIR/_research/results/ckpts
# Submit from the repo root so sbatch's SLURM_SUBMIT_DIR — which common.sh uses
# as WORKDIR (paths, EDF, CKPT_DIR) — is the repo root regardless of where this
# script was invoked. (Running it from inside the framework dir would otherwise
# prepend that path to everything and the EDF/checkpoint lookups would fail.)
cd "$REPO_DIR"

SIZE=150m; RECIPE=master; CLUSTER=mi300
SRC=""; BRANCH_ITERS=""; CONT_ITERS=10000; WARMUP_ITERS=500
ARMS="1 2 3 4"; TAG=""; DRY=""; PASS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --src)          SRC="$2"; shift 2 ;;
        --size)         SIZE="$2"; shift 2 ;;
        --recipe)       RECIPE="$2"; shift 2 ;;
        --cluster)      CLUSTER="$2"; shift 2 ;;
        --branch-iters) BRANCH_ITERS="$2"; shift 2 ;;   # default: source tracker
        --cont-iters)   CONT_ITERS="$2"; shift 2 ;;      # extra steps to train
        --warmup-iters) WARMUP_ITERS="$2"; shift 2 ;;    # warmup arms' ramp length
        --arms)         ARMS="$2"; shift 2 ;;            # subset of "1 2 3 4"
        --tag)          TAG="$2"; shift 2 ;;             # name suffix -> distinct run/ckpt
        --nodes)        PASS+=(--nodes "$2"); shift 2 ;;
        --time)         PASS+=(--time "$2"); shift 2 ;;
        --reservation)  PASS+=(--reservation "$2"); shift 2 ;;
        --dry-run)      DRY=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

usage="usage: [LR=.. MLR=.. ELR=.. MBS=..] warm-restart.sh --src <ckpt-dir-name> [--size $SIZE] [--recipe $RECIPE] [--cluster $CLUSTER] [--branch-iters N] [--cont-iters $CONT_ITERS] [--warmup-iters $WARMUP_ITERS] [--arms \"1 2 3 4\"] [--nodes N] [--time T] [--dry-run]"
: "${SRC:?$usage}"

SIZE_FILE="$FRAMEWORK_DIR/sizes/$SIZE.sh"
CLUSTER_FILE="$FRAMEWORK_DIR/clusters/$CLUSTER.sh"
RECIPE_FILE="$FRAMEWORK_DIR/recipes/$RECIPE.sh"
[ -f "$SIZE_FILE" ]    || { echo "no such size: $SIZE" >&2; exit 1; }
[ -f "$CLUSTER_FILE" ] || { echo "no such cluster: $CLUSTER" >&2; exit 1; }
[ -f "$RECIPE_FILE" ]  || { echo "no such recipe: $RECIPE" >&2; exit 1; }

SRC_DIR=$CKPT_ROOT/$SRC
[ -d "$SRC_DIR" ] || { echo "source ckpt dir not found: $SRC_DIR" >&2; exit 1; }

SEQ_LEN=$(grep -E '^SEQ_LEN=' "$SIZE_FILE" | head -1 | cut -d= -f2)
GBS=$( (source "$SIZE_FILE" >/dev/null 2>&1; echo "$GBS") )   # resolves ${GBS:-128}
: "${SEQ_LEN:?}" "${GBS:?}"

# Branch point: the source iteration to continue from (its checkpoint must exist).
if [ -z "$BRANCH_ITERS" ]; then
    BRANCH_ITERS=$(tr -dc '0-9' < "$SRC_DIR/latest_checkpointed_iteration.txt" 2>/dev/null || true)
    : "${BRANCH_ITERS:?could not read $SRC_DIR/latest_checkpointed_iteration.txt — pass --branch-iters}"
fi
ITER_DIR=$(printf 'iter_%07d' "$BRANCH_ITERS")
[ -d "$SRC_DIR/$ITER_DIR" ] || { echo "branch checkpoint missing: $SRC_DIR/$ITER_DIR" >&2; exit 1; }

BRANCH_SAMPLES=$(( BRANCH_ITERS * GBS ))
CONT_SAMPLES=$(( CONT_ITERS * GBS ))
WARMUP_SAMPLES=$(( WARMUP_ITERS * GBS ))
# All arms resume in place (counter continues at the branch), so TRAIN_SAMPLES
# spans branch+continuation and the warmup/decay are anchored at the branch step.
REUSE_TRAIN_SAMPLES=$(( BRANCH_SAMPLES + CONT_SAMPLES ))

# Name prefix (size-EXP_TAG-KNOB_STR) + cluster tag, exactly as common.sh builds
# EXP_NAME, so we can pre-seed the reuse arms' CKPT_DIRs. Knob env (LR/MLR/...)
# flows into KNOB_STR, so the prefix matches the source run's name.
NAME_PART=$( { source "$SIZE_FILE"; source "$RECIPE_FILE"; } >/dev/null 2>&1; echo "${EXP_TAG:-${RECIPE//\//-}}${KNOB_STR:+-${KNOB_STR}}" )
CLUSTER_TAG=$( (source "$CLUSTER_FILE" >/dev/null 2>&1; echo "${CLUSTER_TAG:-}") )

echo ">>> warm-restart: src=$SRC branch@$BRANCH_ITERS (+$CONT_ITERS steps), warmup=$WARMUP_ITERS"
echo ">>>   prefix=${SIZE}-${NAME_PART}${CLUSTER_TAG}  train_samples=$REUSE_TRAIN_SAMPLES (all arms full-resume from the branch)"
echo ">>>   knobs: LR=${LR:-<recipe default>} MLR=${MLR:-<recipe default>} ELR=${ELR:-<recipe default>} MBS=${MBS:-<size default>}"

WARM_TAG="warm${WARMUP_ITERS}"

# Pre-seed a reuse arm's CKPT_DIR with a symlink to the source branch checkpoint
# (+ tracker), so the first job resumes in place. Idempotent; never touched once
# the run has saved its own checkpoints.
seed_reuse_dir() {
    local exp_name=$1 dir
    [ -n "$DRY" ] && return 0
    dir=$CKPT_ROOT/$exp_name
    mkdir -p "$dir"
    [ -e "$dir/$ITER_DIR" ] || ln -s "$SRC_DIR/$ITER_DIR" "$dir/$ITER_DIR"
    [ -f "$dir/latest_checkpointed_iteration.txt" ] || echo "$BRANCH_ITERS" > "$dir/latest_checkpointed_iteration.txt"
}

launch() {
    local run_tag=$1; shift            # RUN_TAG (-> name suffix)
    local exp_name=${SIZE}-${NAME_PART}${CLUSTER_TAG}-${run_tag}
    echo ">>> --- arm: $exp_name ---"
    echo ">>>     env: $*"
    if [ -n "$DRY" ]; then
        SIZE="$SIZE" RECIPE="$RECIPE" CLUSTER="$CLUSTER" DRY_RUN=1 RUN_TAG="$run_tag" \
            env "$@" bash "$FRAMEWORK_DIR/submit.sh" --size "$SIZE" --recipe "$RECIPE" --cluster "$CLUSTER" --dry-run "${PASS[@]}" \
            | sed 's/^/        /'
        return 0
    fi
    env RUN_TAG="$run_tag" "$@" \
        bash "$FRAMEWORK_DIR/submit.sh" --size "$SIZE" --recipe "$RECIPE" --cluster "$CLUSTER" "${PASS[@]}"
}

# Optional --tag suffix -> distinct run name + checkpoint dir (so a re-run with
# a tweaked mechanism coexists with earlier results instead of colliding).
SFX="${TAG:+-$TAG}"

for arm in $ARMS; do
    case "$arm" in
        1)  # reuse moments, no warmup: resume in place, peak at branch -> anneal
            rt="from${BRANCH_ITERS}-reopt-nowarm${SFX}"
            seed_reuse_dir "${SIZE}-${NAME_PART}${CLUSTER_TAG}-${rt}"
            launch "$rt" \
                TRAIN_SAMPLES="$REUSE_TRAIN_SAMPLES" \
                LR_DECAY_STYLE=linear LR_WARMUP_SAMPLES="$BRANCH_SAMPLES" \
                OVERRIDE_OPT_SCHED=1 ;;
        2)  # reuse moments, shifted warmup at the resume step -> anneal
            rt="from${BRANCH_ITERS}-reopt-${WARM_TAG}${SFX}"
            seed_reuse_dir "${SIZE}-${NAME_PART}${CLUSTER_TAG}-${rt}"
            launch "$rt" \
                TRAIN_SAMPLES="$REUSE_TRAIN_SAMPLES" \
                LR_DECAY_STYLE=linear \
                LR_WARMUP_START_SAMPLES="$BRANCH_SAMPLES" \
                LR_WARMUP_SAMPLES="$(( BRANCH_SAMPLES + WARMUP_SAMPLES ))" \
                OVERRIDE_OPT_SCHED=1 ;;
        3)  # fresh moments, no warmup: full resume (keeps the learnable gains +
            # data position), then RESET_OPT_MOMENTS zeros the Adam/momentum
            # buffers. Schedule identical to arm 1 (peak at the resume step).
            rt="from${BRANCH_ITERS}-newopt-nowarm${SFX}"
            seed_reuse_dir "${SIZE}-${NAME_PART}${CLUSTER_TAG}-${rt}"
            launch "$rt" \
                TRAIN_SAMPLES="$REUSE_TRAIN_SAMPLES" \
                LR_DECAY_STYLE=linear LR_WARMUP_SAMPLES="$BRANCH_SAMPLES" \
                OVERRIDE_OPT_SCHED=1 RESET_OPT_MOMENTS=1 ;;
        4)  # fresh moments, shifted warmup at the resume step then anneal.
            # Schedule identical to arm 2; gains kept, momentum reset.
            rt="from${BRANCH_ITERS}-newopt-${WARM_TAG}${SFX}"
            seed_reuse_dir "${SIZE}-${NAME_PART}${CLUSTER_TAG}-${rt}"
            launch "$rt" \
                TRAIN_SAMPLES="$REUSE_TRAIN_SAMPLES" \
                LR_DECAY_STYLE=linear \
                LR_WARMUP_START_SAMPLES="$BRANCH_SAMPLES" \
                LR_WARMUP_SAMPLES="$(( BRANCH_SAMPLES + WARMUP_SAMPLES ))" \
                OVERRIDE_OPT_SCHED=1 RESET_OPT_MOMENTS=1 ;;
        *) echo "unknown arm: $arm (expected 1..4)" >&2; exit 1 ;;
    esac
done
echo ">>> done."
