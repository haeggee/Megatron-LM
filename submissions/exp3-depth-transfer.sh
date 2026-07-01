source .env  # export WANDB_API_KEY and HF_TOKEN here.
# Exp 3: LR transfer in depth for master-muon + gains.
# Fix hidden_size = 512, vary num_layers: 12 / 18 / 24.
# Sweep absolute matrix LR per size; base + embedding LR held fixed.

SQRT_MODELDIM=22.62   # sqrt(512); constant across depth sweep.

# config: NODES MODEL_SIZE INV_LAYERS
configs=(
	# "2 90  0.167"   # 6L
	"2 110 0.083 12"   # 12L
	"2 130 0.056 18"   # 18L
	"2 150 0.042 24"   # 24L
	"4 170 0.033 30"   # 30L
)

BASE_LR=0.001
EMB_LR=0.003

# for config in "${configs[@]}"; do
# 	read -r NODES MODEL_SIZE INV_LAYERS <<< "$config"
# 	for k in 3 4 5 6 7 8; do
# 		matrix_lr=$(python3 -c "print(0.001 * 2**($k/2))")
# 		bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
# 			--eval-every 1000 --eval-iters 50 \
# 			--opt master --master-orthogonalize --alpha 0 \
# 			--hs flat --hs-embed row --hs-embed-no-orthogonal \
# 			--b1 0.95 --mb1 0.9 --muon-scale shape_up --muon-nesterov \
# 			--hs-g rowcol --hs-g-embed none \
# 			--post-norm \
# 			--hs-g-param softplus \
# 			--fixed-layer-scale $INV_LAYERS \
# 			--upscale-embedding $SQRT_MODELDIM \
# 			--qk-norm RMSNorm \
# 			--wd 0 --decay linear --no-warmup \
# 			--untie-embed \
# 			--lr $BASE_LR --matrix-lr $matrix_lr --embedding-lr $EMB_LR \
# 			--extra-name exp3-depth \
# 			$*
# 	done
# done


# for config in "${configs[@]}"; do
# 	read -r NODES MODEL_SIZE INV_LAYERS N_LAYERS <<< "$config"
# 	INV_SQRT_2L=$(python3 -c "import math; print(f'{1 / math.sqrt(2 * $N_LAYERS):.3f}')")
# 	# for k in 5 6 7; do
# 	for k in 8 9; do
# 		matrix_lr=$(python3 -c "print(0.001 * 2**($k/2))")
# 		bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
# 			--eval-every 1000 --eval-iters 50 \
# 			--opt master --master-orthogonalize --alpha 0 \
# 			--hs flat --hs-embed row --hs-embed-no-orthogonal \
# 			--b1 0.95 --mb1 0.9 --muon-scale shape_up --muon-nesterov \
# 			--hs-g rowcol --hs-g-embed none \
# 			--post-norm \
# 			--hs-g-param softplus \
# 			--fixed-layer-scale $INV_SQRT_2L \
# 			--upscale-embedding $SQRT_MODELDIM \
# 			--qk-norm RMSNorm \
# 			--wd 0 --decay linear --no-warmup \
# 			--untie-embed \
# 			--lr $BASE_LR --matrix-lr $matrix_lr --embedding-lr $EMB_LR \
# 			--extra-name exp3-depth \
# 			$*
# 	done
# done


# Fixed residual scale 1/sqrt(2L), pre-norm only (no sandwich norm / no post-norm).
for config in "${configs[@]}"; do
	read -r NODES MODEL_SIZE INV_LAYERS N_LAYERS <<< "$config"
	INV_SQRT_2L=$(python3 -c "import math; print(f'{1 / math.sqrt(2 * $N_LAYERS):.3f}')")
	for k in 4 5 6 7 8 9; do
		matrix_lr=$(python3 -c "print(0.001 * 2**($k/2))")
		bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
			--eval-every 1000 --eval-iters 50 \
			--opt master --master-orthogonalize --alpha 0 \
			--hs flat --hs-embed row --hs-embed-no-orthogonal \
			--b1 0.95 --mb1 0.9 --muon-scale shape_up --muon-nesterov \
			--hs-g rowcol --hs-g-embed none \
			--hs-g-param softplus \
			--fixed-layer-scale $INV_SQRT_2L \
			--upscale-embedding $SQRT_MODELDIM \
			--qk-norm RMSNorm \
			--wd 0 --decay linear --no-warmup \
			--untie-embed \
			--lr $BASE_LR --matrix-lr $matrix_lr --embedding-lr $EMB_LR \
			--extra-name exp3-depth-prenorm \
			$*
	done
done



# for config in "${configs[@]}"; do
# 	read -r NODES MODEL_SIZE INV_LAYERS N_LAYERS <<< "$config"
# 	INV_2L=$(python3 -c "import math; print(f'{1 / (2 * $N_LAYERS):.3f}')")
# 	for k in 5 6 7; do
# 		matrix_lr=$(python3 -c "print(0.001 * 2**($k/2))")
# 		bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
# 			--eval-every 1000 --eval-iters 50 \
# 			--opt master --master-orthogonalize --alpha 0 \
# 			--hs flat --hs-embed row --hs-embed-no-orthogonal \
# 			--b1 0.95 --mb1 0.9 --muon-scale shape_up --muon-nesterov \
# 			--hs-g rowcol --hs-g-embed none \
# 			--post-norm \
# 			--hs-g-param softplus \
# 			--fixed-layer-scale $INV_2L \
# 			--upscale-embedding $SQRT_MODELDIM \
# 			--qk-norm RMSNorm \
# 			--wd 0 --decay linear --no-warmup \
# 			--untie-embed \
# 			--lr $BASE_LR --matrix-lr $matrix_lr --embedding-lr $EMB_LR \
# 			--extra-name exp3-depth \
# 			$*
# 	done
# done
