source .env  # export WANDB_API_KEY and HF_TOKEN here.

# Base constants copied from 110m-gains-transfer.sh.
INV_LAYERS=0.083  # approx 1/n_layers

configs=(
	"2 110 0.044  22.62"   # model dim: 512
	# "2 193 0.036  27.71"   # model dim: 768
	# "2 292 0.031  32.00"   # model dim: 1024
	# "4 860 0.022  45.25"   # model dim: 2048
)

# Compare gain parametrizations: direct (= current, g), offset (= 1+g, init g=0),
# softplus (= softplus(g), init g=ln(e-1)). Base config matches the active "gains"
# block in 110m-gains-transfer.sh. Adds `--hs-g-param <mode>` per run; "direct"
# reproduces the existing baseline.
for config in "${configs[@]}"; do
	read -r NODES MODEL_SIZE INV_SQRTMODELDIM SQRT_MODELDIM <<< "$config"
	for lr in 0.001; do
		for k in 6; do
			# mlr as power of sqrt(2)
			mlr=$(python3 -c "print(2**($k/2))")
			echo "mlr: $mlr"
			for param in softplus; do
				bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
					--eval-every 1000 --eval-iters 50 \
					--opt master --master-orthogonalize --alpha 0 --hs flat --hs-embed row \
					--hs-embed-no-orthogonal \
					--b1 0.95 --mb1 0.9 --muon-scale shape_up \
					--hs-g rowcol \
					--hs-g-embed none \
					--hs-g-param $param \
					--untie-embed \
					--lr $lr \
					--mlr $mlr \
					--elr 3 \
					--muon-nesterov \
					--post-norm \
					--fixed-layer-scale $INV_LAYERS \
					--upscale-embedding $SQRT_MODELDIM \
					--qk-norm RMSNorm \
					--wd 0 --decay linear \
					--no-warmup \
					--extra-name exp9-gains-param-$param \
					$*
			done
		done
	done
done
