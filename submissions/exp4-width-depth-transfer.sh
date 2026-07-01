source .env  # export WANDB_API_KEY and HF_TOKEN here.
# Exp 4: combined depth + width transfer for master-muon + gains.
# Ratio-preserving sweep: hidden/layers = 512/12 = 768/18 = 1024/24 ≈ 42.67.
# 110 (12L,512) -> 232 (18L,768) -> 430 (24L,1024).

# config: NODES MODEL_SIZE INV_LAYERS SQRT_MODELDIM
configs=(
	# "2 70  0.1111 19.60"  # 9L,  hidden 384
	# "2 110 0.0833 22.62"  # 12L, hidden 512
	"4 232 0.056 27.71"  # 18L, hidden 768
	# "4 430 0.042 32.00"  # 24L, hidden 1024
)

BASE_LR=0.001
EMB_LR=0.003

# for config in "${configs[@]}"; do
# 	read -r NODES MODEL_SIZE INV_LAYERS SQRT_MODELDIM <<< "$config"
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
# 			--extra-name exp4-width-depth \
# 			$*
# 	done
# done


# adam
# ---- adam (master, no orthogonalize, no hypersphere) ----
# for config in "${configs[@]}"; do
# 	read -r NODES MODEL_SIZE INV_LAYERS SQRT_MODELDIM <<< "$config"
# 	for k in 3 4 5 6 7 8; do
# 		matrix_lr=$(python3 -c "print(0.001 * 2**($k/2))")
# 		bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
# 			--eval-every 1000 --eval-iters 50 \
# 			--opt master --alpha 0 \
# 			--post-norm \
# 			--fixed-layer-scale $INV_LAYERS \
# 			--upscale-embedding $SQRT_MODELDIM \
# 			--qk-norm RMSNorm \
# 			--wd 0.1 --decay linear \
# 			--warmup-iters 1000 \
# 			--untie-embed \
# 			--lr $BASE_LR --matrix-lr $matrix_lr --embedding-lr $EMB_LR \
# 			--extra-name exp4-width-depth-adam \
# 			$*
# 	done
# done


# muon
# ---- muon (master + orthogonalize, no hypersphere) ----
for config in "${configs[@]}"; do
	read -r NODES MODEL_SIZE INV_LAYERS SQRT_MODELDIM <<< "$config"
	for k in 3 4 5 6 7 8; do
		matrix_lr=$(python3 -c "print(0.001 * 2**($k/2))")
		bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
			--eval-every 1000 --eval-iters 50 \
			--opt master --master-orthogonalize --alpha 0 \
			--b1 0.95 --mb1 0.9 --muon-scale unit_rms_norm --muon-nesterov \
			--post-norm \
			--fixed-layer-scale $INV_LAYERS \
			--upscale-embedding $SQRT_MODELDIM \
			--qk-norm RMSNorm \
			--wd 0.1 --decay linear \
			--warmup-iters 1000 \
			--untie-embed \
			--lr $BASE_LR --matrix-lr $matrix_lr --embedding-lr $EMB_LR \
			--extra-name exp4-width-depth-muon \
			$*
	done
done