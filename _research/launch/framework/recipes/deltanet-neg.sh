# shellcheck shell=bash
#
# deltanet-neg — DeltaNet linear attention with negative eigenvalues, on
# adaptive-muon. This is what a typical arch ablation looks like: inherit a
# base optimizer recipe, then add the few flags that ARE the idea.
#
# (Legacy equivalent: transformer-pp-350m-ablation-muon-deltanet-neg.sbatch —
#  ~220 lines; here it is 4 lines of actual content.)

source "$(dirname "${BASH_SOURCE[0]}")/adaptive-muon.sh"

EXP_TAG=deltanet-neg
RECIPE_ARGS+=(
    --linear-attention-allow-neg-eigval
    --linear-attention-qk-norm rmsnorm
    --linear-attention-freq 4
)
