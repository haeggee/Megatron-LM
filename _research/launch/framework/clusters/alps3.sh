# shellcheck shell=bash
#
# alps3 — GH200 nodes, the default cluster. Every value here is exactly what
# was hard-coded in common.sh / train.sbatch before clusters/ existed, so the
# composed args with this file selected are byte-identical to the pre-clusters
# framework (checked via lib/dump-args.sh diff when the overlay was added).
#
# ── cluster contract ─────────────────────────────────────────────────────────
# A cluster file MAY set (defaults in common.sh reproduce alps3):
#   EDF_TOML                   # container EDF filename under _research/launch/
#   SRUN_MPI                   # srun --mpi flavor (pmix | pmi2)
#   SRUN_EXTRA_ARGS=(...)      # extra srun flags
#   MIXED_PRECISION_ARGS=(...) # precision flags (bf16 / fp8 recipe)
#   MOE_DISPATCHER             # --moe-token-dispatcher-type value
#   MOE_PERMUTE_FUSION         # 1 (default) adds --moe-permute-fusion; 0 omits
#   CLUSTER_TRAIN_ARGS=(...)   # appended to TRAINING_ARGS
#   ARCH_TAG                   # suffix for triton/inductor caches + packages dir
#   CLUSTER_TAG                # suffix appended to EXP_NAME (and job name) so
#                              # runs on different clusters never share CKPT_DIR
#   CLUSTER_SBATCH_FLAGS=(...) # sbatch flags submit.sh (and auto-requeue) adds
#                              # on top of train.sbatch's generic header
# Everything uses ${VAR:-} / declare-guards so the env and earlier-sourced
# size/recipe fragments win over the cluster default.
# ─────────────────────────────────────────────────────────────────────────────

EDF_TOML=${EDF_TOML:-alps3.toml}
SRUN_MPI=${SRUN_MPI:-pmix}
declare -p SRUN_EXTRA_ARGS >/dev/null 2>&1 || SRUN_EXTRA_ARGS=(--network=disable_rdzv_get)
declare -p MIXED_PRECISION_ARGS >/dev/null 2>&1 || MIXED_PRECISION_ARGS=(
    --bf16
    --fp8-format e4m3
    --fp8-recipe blockwise
)
MOE_DISPATCHER=${MOE_DISPATCHER:-allgather}
MOE_PERMUTE_FUSION=${MOE_PERMUTE_FUSION:-1}
declare -p CLUSTER_TRAIN_ARGS >/dev/null 2>&1 || CLUSTER_TRAIN_ARGS=()
ARCH_TAG=${ARCH_TAG:-}
CLUSTER_TAG=${CLUSTER_TAG:-}
declare -p CLUSTER_SBATCH_FLAGS >/dev/null 2>&1 || CLUSTER_SBATCH_FLAGS=(
    --cpus-per-task=72
    --mem=460000
)
