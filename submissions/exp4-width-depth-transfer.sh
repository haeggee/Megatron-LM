source .env  # export WANDB_API_KEY and HF_TOKEN here.
# Exp 4: combined depth + width transfer for master-muon + gains.
# 110 (12L,512) -> 390 (16L,1024) -> 1 (16L,2048). 1B is expensive --
# uncomment when ready.

# config: NODES MODEL_SIZE INV_LAYERS SQRT_MODELDIM
configs=(
	"2 110 0.083 22.62"   # 12L, hidden 512
	"2 390 0.0625 32.00"  # 16L, hidden 1024
	# "4 1   0.0625 45.25"  # 16L, hidden 2048
)

BASE_LR=0.001
EMB_LR=0.003

for config in "${configs[@]}"; do
	read -r NODES MODEL_SIZE INV_LAYERS SQRT_MODELDIM <<< "$config"
	for k in 3 4 5 6 7 8; do
		matrix_lr=$(python3 -c "print(0.001 * 2**($k/2))")
		bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
			--eval-every 1000 --eval-iters 50 \
			--opt master --master-orthogonalize --alpha 0 \
			--hs flat --hs-embed row --hs-embed-no-orthogonal \
			--b1 0.95 --mb1 0.9 --muon-scale shape_up --muon-nesterov \
			--hs-g rowcol --hs-g-embed none \
			--post-norm \
			--fixed-layer-scale $INV_LAYERS \
			--upscale-embedding $SQRT_MODELDIM \
			--qk-norm RMSNorm \
			--wd 0 --decay linear --no-warmup \
			--untie-embed \
			--lr $BASE_LR --matrix-lr $matrix_lr --embedding-lr $EMB_LR \
			--extra-name exp4-width-depth \
			$*
	done
done
