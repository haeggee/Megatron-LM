source .env  # export WANDB_API_KEY and HF_TOKEN here.
LS=0.083  # approx 1/n_layers
LSS=0.044  # approx 1/sqrt(hidden_dim)
SS=11.31  # approx sqrt(k_dim)
MS=22.62  # approx sqrt(hidden_dim)
bash submissions/submit.sh 110 \
	--no-pre-norm --no-final-layernorm \
	--opt ademamix --alpha 0 --hb row --hb-nu --hb-embed --hb-split-heads \
	--normalization L2Norm --no-learnable-norms --post-norm --post-block-norm --use-stream-minus-residual --layer-scale $LS --layer-scale-scale $LSS \
	--softmax-scale $SS --qk-norm L2Norm --qk-layer-scale 1 --qk-layer-scale-scale $LSS \
	--mlp-layer-scale 1 --mlp-layer-scale-gate-scale $MS  \
	--logits-layer-scale 1 --logits-layer-scale-scale $LSS \
	--no-warmup --wd 0 \
	--init $LSS \
	$*
