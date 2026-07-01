source .env  # export WANDB_API_KEY and HF_TOKEN here.
# Exp 14: rank-k gains on 110M.
# Generalizes the rank-1 row+col gains (G = row⊗col) to a learned rank-k
# multiplier G = 1 + A@B (A:[m,k], B:[k,n]) applied elementwise to each matrix.
# Same master+adam+hypersphere setup as the active exp1 arms; matrix LR is swept
# absolutely. Arm A reproduces the rowcol baseline; arm B sweeps lowrank rank.
#
# lowrank gains init to exactly identity (B=0, A random), so no gain_parametrization
# (phi) is used — the correction is inherently additive. wd on A/B pulls G -> I.

INV_LAYERS=0.083
SQRT_MODELDIM=22.62
NODES=2
MODEL_SIZE=110
INV_SQRT_MODELDIM=0.044

BASE_LR=0.001
EMB_LR=0.003

# matrix LR sweep: 2^(k/2) * 1e-3 for k in 0..4 (matches active exp1 arms).

# # ---- baseline: rank-1 row+col gains (current best) ----
# for k in 0 1 2 3 4; do
# 	matrix_lr=$(python3 -c "print(0.001 * 2**($k/2))")
# 	bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
# 		--eval-every 1000 --eval-iters 50 \
# 		--opt master --master-orthogonalize --alpha 0 \
# 		--hs flat --hs-embed row --hs-embed-no-orthogonal \
# 		--b1 0.95 --mb1 0.9 --muon-scale shape_up --muon-nesterov \
# 		--hs-g rowcol --hs-g-embed none \
# 		--post-norm \
# 		--hs-g-param softplus \
# 		--fixed-layer-scale $INV_LAYERS \
# 		--upscale-embedding $SQRT_MODELDIM \
# 		--qk-norm RMSNorm \
# 		--wd 0 --decay linear --no-warmup \
# 		--untie-embed \
# 		--lr $BASE_LR --matrix-lr $matrix_lr --embedding-lr $EMB_LR \
# 		--extra-name exp14-rowcol-baseline \
# 		$*
# done

# ---- new: rank-k lowrank gains, sweep matrix LR x rank ----
for rank in 4; do
	for k in 9; do
		matrix_lr=$(python3 -c "print(0.001 * 2**($k/2))")
		bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
			--eval-every 1000 --eval-iters 50 \
			--opt master --master-orthogonalize --alpha 0 \
			--hs flat --hs-embed row --hs-embed-no-orthogonal \
			--b1 0.95 --mb1 0.9 --muon-scale shape_up --muon-nesterov \
			--hs-g lowrank --hs-g-rank $rank --hs-g-embed none \
			--post-norm \
			--fixed-layer-scale $INV_LAYERS \
			--upscale-embedding $SQRT_MODELDIM \
			--qk-norm RMSNorm \
			--wd 0 --decay linear --no-warmup \
			--untie-embed \
			--lr $BASE_LR --matrix-lr $matrix_lr --embedding-lr $EMB_LR \
			--extra-name exp14-lora \
			$*
	done
done
