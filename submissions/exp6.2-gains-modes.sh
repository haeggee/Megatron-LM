source .env  # export WANDB_API_KEY and HF_TOKEN here.
# Exp 6.2: gains modes comparison on 110M (master-muon).
# Vary the gains mode (--hs-g) across {flat, embed, row}, with --hs fixed to flat.

INV_LAYERS=0.083
SQRT_MODELDIM=22.62
NODES=2
MODEL_SIZE=110

BASE_LR=0.001
EMB_LR=0.003

# ---- with gains, flat HS ----
# for gainsmode in flat embed row col; do
# 	for k in 3 4 5 6 7 8; do
# 		matrix_lr=$(python3 -c "print(0.001 * 2**($k/2))")
# 		bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
# 			--eval-every 1000 --eval-iters 50 \
# 			--opt master --master-orthogonalize --alpha 0 \
# 			--hs flat --hs-embed row --hs-embed-no-orthogonal \
# 			--b1 0.95 --mb1 0.9 --muon-scale shape_up --muon-nesterov \
# 			--hs-g $gainsmode --hs-g-embed none \
# 			--post-norm \
# 			--hs-g-param softplus \
# 			--fixed-layer-scale $INV_LAYERS \
# 			--upscale-embedding $SQRT_MODELDIM \
# 			--qk-norm RMSNorm \
# 			--wd 0 --decay linear --no-warmup \
# 			--untie-embed \
# 			--lr $BASE_LR --matrix-lr $matrix_lr --embedding-lr $EMB_LR \
# 			--extra-name exp6.2-hs-${gainsmode} \
# 			$*
# 	done
# done


# softplus, direct, exp parametrization
# for param in direct exp; do
# 	for k in 3 4 5 6 7 8; do
# 		matrix_lr=$(python3 -c "print(0.001 * 2**($k/2))")
# 		bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
# 			--eval-every 1000 --eval-iters 50 \
# 			--opt master --master-orthogonalize --alpha 0 \
# 			--hs flat --hs-embed row --hs-embed-no-orthogonal \
# 			--b1 0.95 --mb1 0.9 --muon-scale shape_up --muon-nesterov \
# 			--hs-g rowcol --hs-g-embed none \
# 			--hs-g-param $param \
# 			--post-norm \
# 			--fixed-layer-scale $INV_LAYERS \
# 			--upscale-embedding $SQRT_MODELDIM \
# 			--qk-norm RMSNorm \
# 			--wd 0 --decay linear --no-warmup \
# 			--untie-embed \
# 			--lr $BASE_LR --matrix-lr $matrix_lr --embedding-lr $EMB_LR \
# 			--extra-name exp6.2-hs-gains-${param} \
# 			$*
# 	done
# done


# ---- with gains, row HS ----
for gainsmode in embed rowcol; do
	for k in 3 4 5 6 7 8; do
		matrix_lr=$(python3 -c "print(0.001 * 2**($k/2))")
		bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
			--eval-every 1000 --eval-iters 50 \
			--opt master --master-orthogonalize --alpha 0 \
			--hs embed --hs-embed row --hs-embed-no-orthogonal \
			--b1 0.95 --mb1 0.9 --muon-scale shape_up --muon-nesterov \
			--hs-g $gainsmode --hs-g-embed none \
			--post-norm \
			--hs-g-param softplus \
			--fixed-layer-scale $INV_LAYERS \
			--upscale-embedding $SQRT_MODELDIM \
			--qk-norm RMSNorm \
			--wd 0 --decay linear --no-warmup \
			--untie-embed \
			--lr $BASE_LR --matrix-lr $matrix_lr --embedding-lr $EMB_LR \
			--extra-name exp6.2-hs-embed \
			$*
	done
done