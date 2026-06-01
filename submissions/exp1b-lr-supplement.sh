source .env  # export WANDB_API_KEY and HF_TOKEN here.
# Exp 1b: base-LR supplement for the exp 1 optimizer comparison.
# Holds the matrix LR and embedding LR at the central exp-1 values, and
# sweeps the BASE LR (--lr, which now only controls norms/biases/1D params
# in the explicit-LR convention) per optimizer. This anchors the fairness
# claim: "each opt was given the chance to find its preferred base LR
# before we read off the matrix-LR sweep in exp 1."
#
# Pair with exp1-opt-comparison.sh: same architecture per arm; adam/muon
# keep their conventional wd=0.1 + warmup=1000, master variants keep
# wd=0 + --no-warmup.

INV_LAYERS=0.083
SQRT_MODELDIM=22.62
NODES=2
MODEL_SIZE=110

# central matrix LR = 2^(6/2) * 1e-3 = 8e-3 (=k=6 in the exp 1 sweep).
MATRIX_LR=$(python3 -c "print(0.001 * 2**(6/2))")
EMB_LR=0.003
# Pin the LM-head output LR so the --lr sweep does not also move the output
# layer (weight + its gains). Without this, --output-lr is unset and the head
# falls back to config.lr == base_lr, conflating the base/1D sweep with the
# output LR. Fixed at the k=0 base-LR center.
OUT_LR=0.001
# Pin the gains LR independent of both the swept base LR and the matrix LR.
# Without --glr, gains inherit their param-group LR (matrix gains -> matrix_lr,
# output gains -> output_lr), so they are never controlled directly. Fixed at
# the k=0 base-LR center to match OUT_LR.
GAINS_LR=0.001
BASE_LR=0.001

# ---- adam ----
# for k in -3 -2 -1 0 1 2 3; do
# 	base_lr=$(python3 -c "print(0.001 * 2**($k/2))")
# 	MATRIX_LR=$(python3 -c "print(0.001 * 2**(6/2))")
# 	bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
# 		--eval-every 1000 --eval-iters 50 \
# 		--opt master --alpha 0 \
# 		--post-norm \
# 		--fixed-layer-scale $INV_LAYERS \
# 		--upscale-embedding $SQRT_MODELDIM \
# 		--qk-norm RMSNorm \
# 		--wd 0.1 --decay linear \
# 		--warmup-iters 1000 \
# 		--untie-embed \
# 		--lr $base_lr --matrix-lr $MATRIX_LR --embedding-lr $EMB_LR \
# 		--extra-name exp1b-lrsup-adam \
# 		$*
# done

# ---- muon ----
# for k in -3 -2 -1 0 1 2 3; do
# 	base_lr=$(python3 -c "print(0.001 * 2**($k/2))")
# 	bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
# 		--eval-every 1000 --eval-iters 50 \
# 		--opt master --master-orthogonalize --alpha 0 \
# 		--b1 0.95 --mb1 0.9 --muon-scale shape_up --muon-nesterov \
# 		--post-norm \
# 		--fixed-layer-scale $INV_LAYERS \
# 		--upscale-embedding $SQRT_MODELDIM \
# 		--qk-norm RMSNorm \
# 		--wd 0.1 --decay linear \
# 		--warmup-iters 1000 \
# 		--untie-embed \
# 		--lr $base_lr --matrix-lr $MATRIX_LR --embedding-lr $EMB_LR \
# 		--extra-name exp1b-lrsup-muon \
# 		$*
# done

# # ---- master-muon, no gains ----
# for k in -3 -2 -1 0 1 2 3; do
# 	base_lr=$(python3 -c "print(0.001 * 2**($k/2))")
# 	bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
# 		--eval-every 1000 --eval-iters 50 \
# 		--opt master --master-orthogonalize --alpha 0 \
# 		--hs flat --hs-embed row --hs-embed-no-orthogonal \
# 		--b1 0.95 --mb1 0.9 --muon-scale shape_up --muon-nesterov \
# 		--post-norm \
# 		--fixed-layer-scale $INV_LAYERS \
# 		--upscale-embedding $SQRT_MODELDIM \
# 		--qk-norm RMSNorm \
# 		--wd 0 --decay linear --no-warmup \
# 		--untie-embed \
# 		--lr $base_lr --matrix-lr $MATRIX_LR --embedding-lr $EMB_LR \
# 		--extra-name exp1b-lrsup-master-nogains \
# 		$*
# done

# # ---- master-muon + gains ----
for k in 4 6 8; do
# for k in 4 5 6; do
	GAINS_LR=$(python3 -c "print(0.001 * 2**($k/2))")
	bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
		--eval-every 1000 --eval-iters 50 \
		--opt master --master-orthogonalize --alpha 0 \
		--hs flat --hs-embed row --hs-embed-no-orthogonal \
		--b1 0.95 --mb1 0.9 --muon-scale shape_up --muon-nesterov \
		--hs-g rowcol --hs-g-embed none \
		--post-norm \
		--hs-g-param softplus \
		--fixed-layer-scale $INV_LAYERS \
		--upscale-embedding $SQRT_MODELDIM \
		--qk-norm RMSNorm \
		--wd 0 --decay linear --no-warmup \
		--untie-embed \
		--matrix-lr $MATRIX_LR --embedding-lr $EMB_LR --output-lr $OUT_LR --glr $GAINS_LR --lr $BASE_LR \
		--extra-name exp1b-gains \
		$*
done
