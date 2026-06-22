source .env  # export WANDB_API_KEY and HF_TOKEN here.
# Exp 13: matrix-LR sweep for ngpt, in the exp1/exp5 decoupled-LR style.
# Base ngpt config taken verbatim from submissions/110m/110m-ngpt.sh
# (adam-master on the sphere: --hs embed --hs-embed --hs-split-heads, alpha 0,
#  L2Norm, no-learnable-norms, post-norm/post-block-norm, stream-minus-residual,
#  fixed layer/qk/mlp/logits scales, no warmup, wd 0, init 1/sqrt(d)).
#
# Changes vs the baseline:
#   - decay cos -> linear, annealing to the default MIN_LR=1e-8.
#   - LRs decoupled & absolute (as in exp5):
#       --lr           1e-3  (gains: norm/layer-scale gains + 1D params)
#       --embedding-lr 3e-3  (fixed)
#       --output-lr    3e-3  (fixed; requires --untie-embed)
#       --matrix-lr    SWEPT
#   - --untie-embed so the head LR is its own group.
#
# matrix LR sweep: 0.001 * 2^(k/2) for k in 3..8 -> 2.8e-3 .. 1.6e-2
# (same grid as exp1/exp5).

INV_LAYERS=0.083          # approx 1/n_layers
INV_SQRTMODELDIM=0.044    # approx 1/sqrt(hidden_dim)
SQRT_KDIM=11.31           # approx sqrt(k_dim)
SQRT_MODELDIM=22.62       # approx sqrt(hidden_dim)

NODES=2
MODEL_SIZE=110

BASE_LR=0.001
EMB_LR=0.003
HEAD_LR=0.001

# for k in 0 1 2; do # 3 4 5 6 7 8; do
# 	matrix_lr=$(python3 -c "print(0.001 * 2**($k/2))")
# 	bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
# 		--eval-every 1000 --eval-iters 50 \
# 		--no-pre-norm --no-final-layernorm \
# 		--opt master --alpha 0 --hs embed --hs-embed --hs-split-heads \
# 		--normalization L2Norm --no-learnable-norms --post-norm --post-block-norm --use-stream-minus-residual \
# 		--layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
# 		--softmax-scale $SQRT_KDIM --qk-norm L2Norm --qk-layer-scale 1 --qk-layer-scale-scale $INV_SQRTMODELDIM \
# 		--mlp-layer-scale 1 --mlp-layer-scale-gate-scale $SQRT_MODELDIM \
# 		--logits-layer-scale 1 --logits-layer-scale-scale $INV_SQRTMODELDIM \
# 		--no-warmup --wd 0 --decay linear \
# 		--init $INV_SQRTMODELDIM \
# 		--untie-embed \
# 		--lr $BASE_LR --matrix-lr $matrix_lr --embedding-lr $EMB_LR --output-lr $HEAD_LR \
# 		--extra-name exp13-ngpt \
# 		$*
# done

# with layerscale scale 1
# for k in 2 3 4 5; do
# 	matrix_lr=$(python3 -c "print(0.001 * 2**($k/2))")
# 	bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
# 		--eval-every 1000 --eval-iters 50 \
# 		--no-pre-norm --no-final-layernorm \
# 		--opt master --alpha 0 --hs embed --hs-embed --hs-split-heads \
# 		--normalization L2Norm --no-learnable-norms --post-norm --post-block-norm --use-stream-minus-residual \
# 		--layer-scale $INV_LAYERS --layer-scale-scale 1 \
# 		--softmax-scale $SQRT_KDIM --qk-norm L2Norm --qk-layer-scale 1 --qk-layer-scale-scale 1 \
# 		--mlp-layer-scale 1 --mlp-layer-scale-gate-scale $SQRT_MODELDIM \
# 		--logits-layer-scale 1 --logits-layer-scale-scale $INV_SQRTMODELDIM \
# 		--no-warmup --wd 0 --decay linear \
# 		--init $INV_SQRTMODELDIM \
# 		--untie-embed \
# 		--lr $BASE_LR --matrix-lr $matrix_lr --embedding-lr $EMB_LR --output-lr $HEAD_LR \
# 		--extra-name exp13-ngpt-all-but-lg-lss1 \
# 		$*
# done



# with layerscale scale 1
for k in 2 3 4 5; do
	matrix_lr=$(python3 -c "print(0.001 * 2**($k/2))")
	bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
		--eval-every 1000 --eval-iters 50 \
		--no-pre-norm --no-final-layernorm \
		--opt master --alpha 0 --hs embed --hs-embed --hs-split-heads \
		--normalization L2Norm --no-learnable-norms --post-norm --post-block-norm --use-stream-minus-residual \
		--layer-scale 1 --layer-scale-scale $INV_SQRTMODELDIM \
		--softmax-scale $SQRT_KDIM --qk-norm L2Norm --qk-layer-scale 1 --qk-layer-scale-scale $INV_SQRTMODELDIM \
		--mlp-layer-scale 1 --mlp-layer-scale-gate-scale $SQRT_MODELDIM \
		--logits-layer-scale 1 --logits-layer-scale-scale $INV_SQRTMODELDIM \
		--no-warmup --wd 0 --decay linear \
		--init $INV_SQRTMODELDIM \
		--untie-embed \
		--lr $BASE_LR --matrix-lr $matrix_lr --embedding-lr $EMB_LR --output-lr $HEAD_LR \
		--extra-name exp13-ngpt-ls1 \
		$*
done
