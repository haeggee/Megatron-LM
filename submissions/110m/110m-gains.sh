source .env  # export WANDB_API_KEY and HF_TOKEN here.
# 1/sqrt(L)
INV_SQRT_L=0.28  # slightly less to avoid blow ups at initialization.
INV_SQRTMODELDIM=0.044  # approx 1/sqrt(hidden_dim)
INV_LAYERS=0.083  # approx 1/n_layers
SQRT_MODELDIM=22.62  # approx sqrt(hidden_dim)

# with pre norm, post norm and final, hypersphere training muon
# gains mode rowcol
NODES=1
for mlr in 1 2 4 6 8; do
    bash submissions/submit.sh 110 --nodes $NODES \
        --opt master --master-orthogonalize --alpha 0 --hs flat --hs-embed --hs-embed-no-orthogonal \
        --b1 0.95 --mb1 0.9 --muon-scale shape_up \
        --post-norm --post-norm-no-gain \
        --layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
        --upscale-embedding $SQRT_MODELDIM \
        --qk-norm RMSNorm \
        --wd 0 --decay cos --no-warmup \
        --hs-g rowcol \
		--lr 0.003 --mlr $mlr \
		$*
done


# NODES=2
# for mlr in 1 2 4 6 8; do
# 	# linear decay and warmup
#     bash submissions/submit.sh 110 --nodes $NODES \
#         --opt master --master-orthogonalize --alpha 0 --hs flat --hs-embed --hs-embed-no-orthogonal \
#         --b1 0.95 --mb1 0.9 --muon-scale shape_up \
#         --post-norm --post-norm-no-gain \
#         --layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
#         --upscale-embedding $SQRT_MODELDIM \
#         --qk-norm RMSNorm \
#         --wd 0 --decay linear \
#         --hs-g rowcol \
# 		--lr 0.003 --mlr $mlr \
# 		$*
# done