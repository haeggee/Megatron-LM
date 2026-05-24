source .env  # export WANDB_API_KEY and HF_TOKEN here.
# Exp 2: LR transfer in width for master-muon + gains.
# Fix depth = 12 layers, vary hidden_size: 512 / 768 / 1024 / 2048.
# Sweep absolute matrix LR per size; base + embedding LR held fixed.

# config: NODES MODEL_SIZE INV_LAYERS SQRT_MODELDIM
configs=(
	# "2 110 0.083 22.62"   # 12L, hidden 512
	# "2 193 0.083 27.71"   # 12L, hidden 768
	"2 292 0.083 32.00"   # 12L, hidden 1024
	"4 860 0.083 45.25"   # 12L, hidden 2048
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
			--hs-g-param softplus \
			--fixed-layer-scale $INV_LAYERS \
			--upscale-embedding $SQRT_MODELDIM \
			--qk-norm RMSNorm \
			--wd 0 --decay linear --no-warmup \
			--untie-embed \
			--lr $BASE_LR --matrix-lr $matrix_lr --embedding-lr $EMB_LR \
			--extra-name exp2-width \
			$*
	done
done
