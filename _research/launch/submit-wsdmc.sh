#!/usr/bin/env bash
# Submit a WSDMC run: 1 main sbatch + N cooldown sbatches each conditioned on
# the main job via SLURM --dependency=afterok.
#
# Usage:
#   ./submit-wsdmc.sh <main-sbatch> [cooldown-sbatch]
#   ./submit-wsdmc.sh _research/launch/transformer-pp-350m-wsdmc-aurora-qkn-moe-32e-tk3-sh1.sbatch
#
# Tunables (env vars):
#   WSDMC_N_CKPTS         number of checkpoints (default 5)
#   WSDMC_COOLDOWN_FRAC   cooldown length as fraction of accumulated tokens
#                         at each checkpoint (default 0.2)
#   WSDMC_GBS             global batch size (default 128). Used to convert
#                         --train-samples / --save-interval if main sbatch
#                         specifies samples; for iter-based, set 1.
#
# Invariant enforced: --train-samples must equal WSDMC_N_CKPTS * --save-interval
# (i.e. final save coincides with end of training, no orphan partial cycle).

set -euo pipefail

MAIN_SBATCH="${1:?usage: submit-wsdmc.sh <main-sbatch> [cooldown-sbatch]}"
# Default cooldown: same dir, same suffix-after-'wsdmc-' as main.
# transformer-pp-350m-wsdmc-FOO.sbatch -> wsdmc-cooldown-FOO.sbatch
DEFAULT_COOLDOWN_SUFFIX=$(basename "$MAIN_SBATCH" | sed -E 's/^.*-wsdmc-//; s/\.sbatch$//')
COOLDOWN_SBATCH="${2:-$(dirname "$MAIN_SBATCH")/wsdmc-cooldown-${DEFAULT_COOLDOWN_SUFFIX}.sbatch}"

[ -f "$MAIN_SBATCH" ]     || { echo "ERROR: main sbatch not found: $MAIN_SBATCH" >&2; exit 1; }
[ -f "$COOLDOWN_SBATCH" ] || { echo "ERROR: cooldown sbatch not found: $COOLDOWN_SBATCH" >&2; exit 1; }

N_CKPTS="${WSDMC_N_CKPTS:-5}"
COOLDOWN_FRAC="${WSDMC_COOLDOWN_FRAC:-0.2}"
GBS="${WSDMC_GBS:-128}"

# Parse main sbatch for --train-samples and --save-interval (as iters).
# These appear inside the MEGATRON_ARGS=( ... ) array in the sbatch body.
get_arg() {
    local flag="$1"
    grep -E "^[[:space:]]*$flag[[:space:]]+[0-9]+" "$MAIN_SBATCH" | head -n1 | awk '{print $2}'
}

TRAIN_SAMPLES=$(get_arg "--train-samples")
SAVE_INTERVAL=$(get_arg "--save-interval")

[ -n "$TRAIN_SAMPLES" ] || { echo "ERROR: --train-samples not found in $MAIN_SBATCH" >&2; exit 1; }
[ -n "$SAVE_INTERVAL" ] || { echo "ERROR: --save-interval not found in $MAIN_SBATCH" >&2; exit 1; }

TRAIN_ITERS=$(( TRAIN_SAMPLES / GBS ))

# Invariant
EXPECTED_INTERVAL=$(( TRAIN_ITERS / N_CKPTS ))
if [ "$SAVE_INTERVAL" -ne "$EXPECTED_INTERVAL" ] || [ $(( SAVE_INTERVAL * N_CKPTS )) -ne "$TRAIN_ITERS" ]; then
    echo "ERROR: invariant violated: train-iters ($TRAIN_ITERS) must equal N_CKPTS ($N_CKPTS) * save-interval ($SAVE_INTERVAL); got expected interval $EXPECTED_INTERVAL" >&2
    exit 1
fi

# Build experiment name. The main sbatch's EXP_NAME default is
# "<jobname>-<git-sha>"; we recompute it here so submit-time and run-time
# agree, and pass via env so cooldowns can target the same ckpt dir.
GIT_SHA=$(git rev-parse --short HEAD)
JOB_NAME=$(grep -E '^#SBATCH --job-name=' "$MAIN_SBATCH" | head -n1 | sed 's/^#SBATCH --job-name=//')
WSDMC_EXP_NAME="${JOB_NAME}-${GIT_SHA}"
WORKDIR="$(pwd)"
WSDMC_CKPT_DIR="$WORKDIR/_research/results/ckpts/$WSDMC_EXP_NAME"

echo "WSDMC submission plan"
echo "  main sbatch:     $MAIN_SBATCH"
echo "  cooldown sbatch: $COOLDOWN_SBATCH"
echo "  exp name:        $WSDMC_EXP_NAME"
echo "  ckpt dir:        $WSDMC_CKPT_DIR"
echo "  train iters:     $TRAIN_ITERS"
echo "  save interval:   $SAVE_INTERVAL"
echo "  N checkpoints:   $N_CKPTS"
echo "  cooldown frac:   $COOLDOWN_FRAC"
echo

# Submit main
MAIN_OUT=$(sbatch --parsable --export=ALL,WSDMC_EXP_NAME="$WSDMC_EXP_NAME",WSDMC_CKPT_DIR="$WSDMC_CKPT_DIR" "$MAIN_SBATCH")
MAIN_JID=$(echo "$MAIN_OUT" | tr -d '\n' | awk '{print $NF}')
echo "Submitted MAIN: jobid=$MAIN_JID"

# Submit one cooldown per checkpoint
for i in $(seq 1 "$N_CKPTS"); do
    CKPT_ITER=$(( SAVE_INTERVAL * i ))
    CKPT_TOKENS=$(( CKPT_ITER * GBS * 4096 ))                          # GBS * seq=4096
    # COOLDOWN_ITERS = round(CKPT_ITER * COOLDOWN_FRAC), at least 1
    COOLDOWN_ITERS=$(awk -v ci="$CKPT_ITER" -v f="$COOLDOWN_FRAC" 'BEGIN{ v=ci*f; printf "%d", (v<1?1:v+0.5) }')
    ENDPOINT_TOKENS=$(( (CKPT_ITER + COOLDOWN_ITERS) * GBS * 4096 ))
    ENDPOINT_GB=$(awk -v t="$ENDPOINT_TOKENS" 'BEGIN{ printf "%.1f", t/1e9 }')

    # SLURM time budget: ~1.0 sec/iter on the MoE-32E-tk3-sh1 config + 30%
    # margin + 5 min container/init overhead. Min 30 min.
    SLURM_SECS=$(awk -v c="$COOLDOWN_ITERS" 'BEGIN{ s=c*1.0*1.3+300; if(s<1800)s=1800; printf "%d", s }')
    SLURM_HHMM=$(printf '%02d:%02d:00' $((SLURM_SECS/3600)) $(( (SLURM_SECS%3600)/60 )))

    JNAME="${WSDMC_EXP_NAME}-cd-${ENDPOINT_GB}B"

    CD_OUT=$(sbatch --parsable \
        --dependency=afterok:"$MAIN_JID" \
        --time="$SLURM_HHMM" \
        --job-name="$JNAME" \
        --export=ALL,WSDMC_EXP_NAME="$WSDMC_EXP_NAME",WSDMC_CKPT_DIR="$WSDMC_CKPT_DIR",WSDMC_CKPT_ITER="$CKPT_ITER",WSDMC_COOLDOWN_ITERS="$COOLDOWN_ITERS" \
        "$COOLDOWN_SBATCH")
    CD_JID=$(echo "$CD_OUT" | tr -d '\n' | awk '{print $NF}')

    printf "Submitted COOLDOWN %d/%s: jobid=%s ckpt_iter=%d cooldown_iters=%d endpoint=%sB time=%s\n" \
        "$i" "$N_CKPTS" "$CD_JID" "$CKPT_ITER" "$COOLDOWN_ITERS" "$ENDPOINT_GB" "$SLURM_HHMM"
done

echo
echo "All jobs submitted. Track with:"
echo "  sacct -X -j $MAIN_JID --format=JobID,JobName,State,Elapsed,End"
