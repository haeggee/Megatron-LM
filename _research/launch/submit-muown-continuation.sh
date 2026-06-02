#!/usr/bin/env bash
# Submit continuation jobs for transformer-pp-350m-muown-lr1e-3-fp8-moe-16e-tk1-sh1.
#
# Usage:
#   bash submit-muown-continuation.sh                    # 1 fresh job
#   bash submit-muown-continuation.sh <after_jid>        # chain 1 job after <after_jid>
#   N_CHAIN=3 bash submit-muown-continuation.sh <after_jid>  # chain 3 jobs
#   MASTER_CKPT_DIR=/path/to/ckpt bash submit-muown-continuation.sh
#
# MASTER_CKPT_DIR defaults to the b5074514a checkpoint (the only one with saved
# iterations). Override if the run was relaunched under a different path.

set -euo pipefail

SBATCH=${SBATCH:-_research/legacy/transformer-pp-350m-muown-lr1e-3-fp8-moe-16e-tk1-sh1.sbatch}
N_CHAIN=${N_CHAIN:-1}
AFTER_JID=${1:-}
CKPT_DIR=${MASTER_CKPT_DIR:-$PWD/_research/results/ckpts/transformer-pp-350m-muown-lr1e-3-fp8-moe-16e-tk1-sh1-b5074514a}

[ -f "$SBATCH" ] || { echo "ERROR: sbatch not found: $SBATCH" >&2; exit 1; }
[ -d "$CKPT_DIR" ] || { echo "ERROR: CKPT_DIR not found: $CKPT_DIR" >&2; exit 1; }

echo "Submit muown continuation: N_CHAIN=$N_CHAIN  CKPT_DIR=$CKPT_DIR"
[ -n "$AFTER_JID" ] && echo "  Chaining after job $AFTER_JID" || echo "  No predecessor (fresh submit)"
echo

LAST_JID=${AFTER_JID}
for c in $(seq 1 "$N_CHAIN"); do
    DEP_ARG=()
    [ -n "$LAST_JID" ] && DEP_ARG=(--dependency=afterany:"$LAST_JID")
    LAST_JID=$(sbatch --parsable \
        "${DEP_ARG[@]}" \
        --export=ALL,MASTER_CKPT_DIR="$CKPT_DIR",MEGATRON_DATA_PATH="$MEGATRON_DATA_PATH" \
        "$SBATCH")
    printf "  chain=%d/%d  jobid=%s\n" "$c" "$N_CHAIN" "$LAST_JID"
done
