
source .env  # export WANDB_API_KEY and HF_TOKEN here.
# 1/sqrt(L)
INV_SQRT_L=0.28  # slightly less to avoid blow ups at initialization.
INV_SQRTMODELDIM=0.044  # approx 1/sqrt(hidden_dim)
INV_LAYERS=0.083  # approx 1/n_layers
SQRT_MODELDIM=22.62  # approx sqrt(hidden_dim)

# Default FOG-swiglu hyper adamw settings.
bash submissions/submit.sh 110 \
	--opt master --alpha 0 --hs embed --hs-embed --hs-split-heads \
	--no-pre-norm --post-norm --no-final-layernorm \
	--layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
	--upscale-embedding $SQRT_MODELDIM \
	--qk-norm RMSNorm \
	--logits-layer-scale 1 --logits-layer-scale-scale $INV_SQRTMODELDIM \
	--wd 0 --no-warmup --decay cos \
	$*
