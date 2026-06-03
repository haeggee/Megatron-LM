#!/bin/bash
#
# sweep-optimizers.sh — LR sweep for the optimizer bake-off, three arms:
#
#   adamw          sweeps LR          (matrix LR; embedding/scalar fixed at 1e-3)
#   muon           sweeps MATRIX_LR   (Muon on matrices incl. router)
#   muon + radam   sweeps MATRIX_LR   (ROUTER_ADAM=1: router on AdamW at base LR)
#
# Defaults target the mi300 cluster (MBS=4 is the verified APU setting, see
# clusters/mi300.sh) on the cheap LR-probe size. Run names self-describe
# (lr/mlr/-radam/-mi300), so every point gets its own checkpoint dir.
#
#   bash sweep-optimizers.sh                          # 270m-moe, mi300, 4 nodes
#   bash sweep-optimizers.sh --size 420m-moe --nodes 8
#   bash sweep-optimizers.sh --cluster alps3          # GH200 instead
#   bash sweep-optimizers.sh --auto-requeue           # chain jobs until TRAIN_SAMPLES
#   bash sweep-optimizers.sh --dry-run                # list submits, don't launch
#   ADAMW_LRS="3e-4 6e-4" MUON_MLRS="1e-2" bash sweep-optimizers.sh
#
# NOTE: GBS=128 sizes (270m/420m) cap at 8 nodes with MBS=4 (grad-accum = 1).
#
set -euo pipefail

FRAMEWORK_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)

SIZE=270m-moe; CLUSTER=mi300; NODES=4; DRY=""; EXTRA_SUBMIT_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --size)    SIZE="$2"; shift 2 ;;
        --cluster) CLUSTER="$2"; shift 2 ;;
        --nodes)   NODES="$2"; shift 2 ;;
        --auto-requeue) EXTRA_SUBMIT_ARGS+=(--auto-requeue); shift ;;
        --dry-run) DRY=1; shift ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# mi300: MBS=4 verified (MBS=8 OOMs in an iter-1 burst, 2026-06-03 bring-up).
[ "$CLUSTER" = mi300 ] && export MBS=${MBS:-4}

# Geometric x2 grids centered on each recipe's default.
ADAMW_LRS=${ADAMW_LRS:-"1.5e-4 3e-4 6e-4 1.2e-3"}
MUON_MLRS=${MUON_MLRS:-"5e-3 1e-2 2e-2 4e-2"}

submit() {  # submit <env assignments...> -- passed straight to submit.sh
    local env_args=()
    while [ "$1" != "--" ]; do env_args+=("$1"); shift; done
    shift
    if [ -n "$DRY" ]; then
        echo "DRY: ${env_args[*]} submit.sh --cluster $CLUSTER --size $SIZE --nodes $NODES $* ${EXTRA_SUBMIT_ARGS[*]}"
    else
        env "${env_args[@]}" bash "$FRAMEWORK_DIR/submit.sh" \
            --cluster "$CLUSTER" --size "$SIZE" --nodes "$NODES" "$@" \
            ${EXTRA_SUBMIT_ARGS[@]+"${EXTRA_SUBMIT_ARGS[@]}"}
    fi
}

n=0
for lr in $ADAMW_LRS; do
    submit LR="1e-3" MATRIX_LR="$lr" -- --recipe adamw; n=$((n+1))
done
for mlr in $MUON_MLRS; do
    submit MATRIX_LR="$mlr" -- --recipe muon; n=$((n+1))
done
for mlr in $MUON_MLRS; do
    submit MATRIX_LR="$mlr" ROUTER_ADAM=1 -- --recipe muon; n=$((n+1))
done
for mlr in $MUON_MLRS; do
    submit MLR="$mlr" -- --recipe master; n=$((n+1))
done
for mlr in $MUON_MLRS; do
    submit MATRIX_LR="$mlr" -- --recipe normuon; n=$((n+1))
done


echo ">>> $n jobs ${DRY:+(dry run, nothing submitted) }for SIZE=$SIZE CLUSTER=$CLUSTER NODES=$NODES${MBS:+ MBS=$MBS}"
