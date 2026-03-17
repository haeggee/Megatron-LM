source .env  # export WANDB_API_KEY and HF_TOKEN here.
INV_LAYERS=0.083  # approx 1/n_layers
INV_SQRTMODELDIM=0.044  # approx 1/sqrt(hidden_dim)
SQRT_KDIM=11.31  # approx sqrt(k_dim)
INV_SQRTKDIM=0.088  # approx 1/sqrt(k_dim)
SQRT_MODELDIM=22.62  # approx sqrt(hidden_dim)

# baseline arch (so preLN with swiglu), but using spherical optimizer
bash submissions/submit.sh 110 \
	--opt master --alpha 0 --hs embed --hs-embed --hs-split-heads \
	--no-warmup --wd 0 \
	--decay cos \
	--init $INV_SQRTMODELDIM \
	$*
