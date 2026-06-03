# shellcheck shell=bash
#
# mi300 — AMD MI300A nodes (--partition=mi300), ROCm Megatron v25.6 container.
# Same node layout as alps3 (4 GPUs / 4 tasks per node), so size files and
# node-count math carry over unchanged.
#
# Every delta below is taken from the verified multinode MoE bring-up:
#   /iopsstor/scratch/cscs/schlag/v25_6_moe_test/run_moe_mn_hooks_rocm6.sbatch
#   + beverin_v25_6_hooks.toml (cxi + aws_ofi_nccl rocm6 hooks)
# Deltas vs alps3:
#   * bf16 only — FP8 blockwise is unverified on ROCm TE, so mi300 numbers are
#     NOT directly comparable to fp8 GH200 runs (override MIXED_PRECISION_ARGS
#     in a recipe to experiment)
#   * alltoall MoE dispatcher (allgather unverified on ROCm)
#   * --no-gradient-accumulation-fusion (the fused kernel is CUDA-only)
#   * --mpi=pmi2, no --network=disable_rdzv_get; all NCCL/libfabric/HSA/NVTE
#     env lives in mi300.toml, not here
#   * separate triton/inductor caches + packages dir (gfx942 kernels,
#     ROCm-built wheels must not shadow the CUDA ones)
#   * CLUSTER_TAG=-mi300 so EXP_NAME/CKPT_DIR never collide with an alps3 run
#     of the same size+recipe+knobs on the shared filesystem

EDF_TOML=${EDF_TOML:-mi300.toml}
SRUN_MPI=${SRUN_MPI:-pmi2}
declare -p SRUN_EXTRA_ARGS >/dev/null 2>&1 || SRUN_EXTRA_ARGS=()
declare -p MIXED_PRECISION_ARGS >/dev/null 2>&1 || MIXED_PRECISION_ARGS=(
    --bf16
)
MOE_DISPATCHER=${MOE_DISPATCHER:-allgather}
# ROCm TE lacks the TE>=2.1 fused-permutation symbols the repo's Megatron
# requires for --moe-permute-fusion ("fused permutation is not available") —
# use the unfused fallback. (The container's own ROCm Megatron fork tolerates
# it, but we run the repo's code.)
MOE_PERMUTE_FUSION=${MOE_PERMUTE_FUSION:-0}
declare -p CLUSTER_TRAIN_ARGS >/dev/null 2>&1 || CLUSTER_TRAIN_ARGS=(
    --no-gradient-accumulation-fusion
)
ARCH_TAG=${ARCH_TAG:--rocm}
CLUSTER_TAG=${CLUSTER_TAG:--mi300}
# --mem=0 = all memory of the node (MI300A node memory differs from GH200's;
# the verified bring-up set no --mem at all). 48 cpus/task * 4 tasks per node.
declare -p CLUSTER_SBATCH_FLAGS >/dev/null 2>&1 || CLUSTER_SBATCH_FLAGS=(
    --partition=mi300
    --cpus-per-task=48
    --mem=0
)
