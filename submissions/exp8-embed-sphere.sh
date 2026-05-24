source .env  # export WANDB_API_KEY and HF_TOKEN here.
# Exp 8: embeddings-on-sphere ablation on 110M (master-muon + gains).
# In both arms the embedding is updated by adam (no muon orthogonalization);
# the only difference is whether it's projected back to the unit sphere.
#  - on-sphere : --hs-embed row --hs-embed-no-orthogonal
#  - off-sphere: no --hs-embed at all -> unconstrained adam update.
# Embedding LR is now matched across arms via the explicit --embedding-lr flag
# (this was silently NOT matched before the chained_adam fix).

INV_LAYERS=0.083
SQRT_MODELDIM=22.62
NODES=2
MODEL_SIZE=110

BASE_LR=0.001
EMB_LR=0.003

# ---- embeddings on sphere (default) ----
# for k in 3 4 5 6 7 8; do
# 	matrix_lr=$(python3 -c "print(0.001 * 2**($k/2))")
# 	bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
# 		--eval-every 1000 --eval-iters 50 \
# 		--opt master --master-orthogonalize --alpha 0 \
# 		--hs flat --hs-embed row --hs-embed-no-orthogonal \
# 		--b1 0.95 --mb1 0.9 --muon-scale shape_up --muon-nesterov \
# 		--hs-g rowcol --hs-g-embed none \
# 		--post-norm \
# 		--fixed-layer-scale $INV_LAYERS \
# 		--upscale-embedding $SQRT_MODELDIM \
# 		--qk-norm RMSNorm \
# 		--wd 0 --decay linear --no-warmup \
# 		--untie-embed \
# 		--lr $BASE_LR --matrix-lr $matrix_lr --embedding-lr $EMB_LR \
# 		--extra-name exp8-embed-sphere \
# 		$*
# done

# ---- embeddings off sphere (unconstrained adam) ----
for k in 3 4 5 6 7 8; do
	matrix_lr=$(python3 -c "print(0.001 * 2**($k/2))")
	bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
		--eval-every 1000 --eval-iters 50 \
		--opt master --master-orthogonalize --alpha 0 \
		--hs flat \
		--b1 0.95 --mb1 0.9 --muon-scale shape_up --muon-nesterov \
		--hs-g rowcol \
		--hs-g-param softplus \
		--post-norm \
		--fixed-layer-scale $INV_LAYERS \
		--upscale-embedding $SQRT_MODELDIM \
		--qk-norm RMSNorm \
		--wd 0 --decay linear --no-warmup \
		--untie-embed \
		--lr $BASE_LR --matrix-lr $matrix_lr --embedding-lr $EMB_LR \
		--extra-name exp8-embed-free \
		$*
done
