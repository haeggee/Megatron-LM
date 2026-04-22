
# source .env  # export WANDB_API_KEY and HF_TOKEN here.
# # 1/sqrt(L)
# INV_SQRT_L=0.28  # slightly less to avoid blow ups at initialization.
# INV_SQRTMODELDIM=0.044  # approx 1/sqrt(hidden_dim)
# INV_LAYERS=0.083  # approx 1/n_layers
# SQRT_MODELDIM=22.62  # approx sqrt(hidden_dim)

# # Default FOG-swiglu hyper adamw settings.
# NODES=2
# for lr in 0.005; do
# 	bash submissions/submit.sh 110 --nodes $NODES \
# 		--opt master --alpha 0 --hs embed --hs-embed \
# 		--post-norm --no-final-layernorm --post-norm-no-gain \
# 		--layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
# 		--upscale-embedding $SQRT_MODELDIM \
# 		--qk-norm RMSNorm \
# 		--logits-layer-scale 1 --logits-layer-scale-scale 1 \
# 		--wd 0 --no-warmup --decay cos \
# 		--lr $lr \
# 		$*
# done



# source .env  # export WANDB_API_KEY and HF_TOKEN here.
# # 1/sqrt(L)
# INV_SQRT_L=0.28  # slightly less to avoid blow ups at initialization.
# INV_SQRTMODELDIM=0.044  # approx 1/sqrt(hidden_dim)
# INV_LAYERS=0.083  # approx 1/n_layers
# SQRT_MODELDIM=22.62  # approx sqrt(hidden_dim)

# NODES=2
# # with final
# for lr in 0.005; do
# 	bash submissions/submit.sh 110 --nodes $NODES \
# 		--opt master --alpha 0 --hs embed --hs-embed \
# 		--no-pre-norm --post-norm --post-norm-no-gain \
# 		--layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
# 		--upscale-embedding $SQRT_MODELDIM \
# 		--qk-norm RMSNorm \
# 		--logits-layer-scale 1 --logits-layer-scale-scale 1 \
# 		--wd 0 --no-warmup --decay cos \
# 		--lr $lr \
# 		$*
# done


# source .env  # export WANDB_API_KEY and HF_TOKEN here.
# # 1/sqrt(L)
# INV_SQRT_L=0.28  # slightly less to avoid blow ups at initialization.
# INV_SQRTMODELDIM=0.044  # approx 1/sqrt(hidden_dim)
# INV_LAYERS=0.083  # approx 1/n_layers
# SQRT_MODELDIM=22.62  # approx sqrt(hidden_dim)

# # with pre norm, post norm and final
# NODES=2
# for lr in 0.005; do
# 	bash submissions/submit.sh 110 --nodes $NODES \
# 		--opt master --alpha 0 --hs embed --hs-embed \
# 		--post-norm --post-norm-no-gain \
# 		--layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
# 		--upscale-embedding $SQRT_MODELDIM \
# 		--qk-norm RMSNorm \
# 		--logits-layer-scale 1 --logits-layer-scale-scale 1 \
# 		--wd 0 --no-warmup --decay cos \
# 		--lr $lr \
# 		$*
# done



# source .env  # export WANDB_API_KEY and HF_TOKEN here.
# # 1/sqrt(L)
# INV_SQRT_L=0.28  # slightly less to avoid blow ups at initialization.
# INV_SQRTMODELDIM=0.044  # approx 1/sqrt(hidden_dim)
# INV_LAYERS=0.083  # approx 1/n_layers
# SQRT_MODELDIM=22.62  # approx sqrt(hidden_dim)

# # flat one again
# NODES=2
# for lr in 0.005; do
# 	bash submissions/submit.sh 110 --nodes $NODES \
# 		--opt master --alpha 0 --hs flat --hs-embed \
# 		--no-pre-norm --no-final-layernorm \
# 		--post-norm --post-norm-no-gain \
# 		--layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
# 		--upscale-embedding $SQRT_MODELDIM \
# 		--qk-norm RMSNorm \
# 		--logits-layer-scale 1 --logits-layer-scale-scale 1 \
# 		--wd 0 --no-warmup --decay cos \
# 		--lr $lr \
# 		$*
# done



# source .env  # export WANDB_API_KEY and HF_TOKEN here.
# # 1/sqrt(L)
# INV_SQRT_L=0.28  # slightly less to avoid blow ups at initialization.
# INV_SQRTMODELDIM=0.044  # approx 1/sqrt(hidden_dim)
# INV_LAYERS=0.083  # approx 1/n_layers
# SQRT_MODELDIM=22.62  # approx sqrt(hidden_dim)

# # base but with spherical adam
# NODES=2
# for lr in 0.005; do
# 	bash submissions/submit.sh 110 --nodes $NODES \
# 		--opt master --alpha 0 --hs embed --hs-embed \
# 		--wd 0 --no-warmup --decay cos \
# 		--lr $lr \
# 		--init $INV_SQRTMODELDIM \
# 		$*
# done



# source .env  # export WANDB_API_KEY and HF_TOKEN here.
# # 1/sqrt(L)
# INV_SQRT_L=0.28  # slightly less to avoid blow ups at initialization.
# INV_SQRTMODELDIM=0.044  # approx 1/sqrt(hidden_dim)
# INV_LAYERS=0.083  # approx 1/n_layers
# SQRT_MODELDIM=22.62  # approx sqrt(hidden_dim)

# # flat one again
# NODES=2
# for lr in 0.005; do
# 	bash submissions/submit.sh 110 --nodes $NODES \
# 		--opt master --alpha 0 --hs embed --hs-embed \
# 		--no-pre-norm --no-final-layernorm \
# 		--post-norm --post-norm-no-gain \
# 		--layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
# 		--upscale-embedding $SQRT_MODELDIM \
# 		--qk-norm RMSNorm \
# 		--logits-layer-scale 1 --logits-layer-scale-scale 1 \
# 		--wd 0 --no-warmup --decay cos \
# 		--lr $lr \
# 		$*
# done


# source .env  # export WANDB_API_KEY and HF_TOKEN here.
# # 1/sqrt(L)
# INV_SQRT_L=0.28  # slightly less to avoid blow ups at initialization.
# INV_SQRTMODELDIM=0.044  # approx 1/sqrt(hidden_dim)
# INV_LAYERS=0.083  # approx 1/n_layers
# SQRT_MODELDIM=22.62  # approx sqrt(hidden_dim)

# # with pre norm, post norm and final
# NODES=2
# for lr in 0.005; do
# 	bash submissions/submit.sh 110 --nodes $NODES \
# 		--opt master --alpha 0 --hs embed --hs-embed \
# 		--post-norm --post-norm-no-gain \
# 		--layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
# 		--upscale-embedding $SQRT_MODELDIM \
# 		--qk-norm RMSNorm \
# 		--logits-layer-scale 1 --logits-layer-scale-scale 1 \
# 		--wd 0 --no-warmup --decay cos \
# 		--lr $lr \
# 		$*
# done

# source .env  # export WANDB_API_KEY and HF_TOKEN here.
# # 1/sqrt(L)
# INV_SQRT_L=0.28  # slightly less to avoid blow ups at initialization.
# INV_SQRTMODELDIM=0.044  # approx 1/sqrt(hidden_dim)
# INV_LAYERS=0.083  # approx 1/n_layers
# SQRT_MODELDIM=22.62  # approx sqrt(hidden_dim)

# # with pre norm, post norm and final
# NODES=2
# for lr in 0.001 0.002 0.003 0.004 0.005; do
# 	bash submissions/submit.sh 110 --nodes $NODES \
# 		--opt master --alpha 0 --hs flat --hs-embed \
# 		--post-norm --post-norm-no-gain \
# 		--layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
# 		--upscale-embedding $SQRT_MODELDIM \
# 		--qk-norm RMSNorm \
# 		--logits-layer-scale 1 --logits-layer-scale-scale 1 \
# 		--wd 0 --no-warmup --decay cos \
# 		--lr $lr \
# 		$*
# done

# source .env  # export WANDB_API_KEY and HF_TOKEN here.
# # 1/sqrt(L)
# INV_SQRT_L=0.28  # slightly less to avoid blow ups at initialization.
# INV_SQRTMODELDIM=0.044  # approx 1/sqrt(hidden_dim)
# INV_LAYERS=0.083  # approx 1/n_layers
# SQRT_MODELDIM=22.62  # approx sqrt(hidden_dim)

# # with pre norm, post norm and final, row hypersphere
# NODES=2
# for lr in 0.001 0.002 0.003 0.004 0.005; do
# 	bash submissions/submit.sh 110 --nodes $NODES \
# 		--opt master --alpha 0 --hs row --hs-embed \
# 		--no-pre-norm --no-final-layernorm \
# 		--post-norm --post-norm-no-gain \
# 		--layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
# 		--upscale-embedding $SQRT_MODELDIM \
# 		--qk-norm RMSNorm \
# 		--logits-layer-scale 1 --logits-layer-scale-scale 1 \
# 		--wd 0 --no-warmup --decay cos \
# 		--lr $lr \
# 		$*
# done


# source .env  # export WANDB_API_KEY and HF_TOKEN here.
# # 1/sqrt(L)
# INV_SQRT_L=0.28  # slightly less to avoid blow ups at initialization.
# INV_SQRTMODELDIM=0.044  # approx 1/sqrt(hidden_dim)
# INV_LAYERS=0.083  # approx 1/n_layers
# SQRT_MODELDIM=22.62  # approx sqrt(hidden_dim)

# # with pre norm, post norm and final
# NODES=2
# for lr in 0.002 0.003 0.004 0.005; do
# 	bash submissions/submit.sh 110 --nodes $NODES \
# 		--opt master --alpha 0 --hs flat --hs-embed \
# 		--post-norm --post-norm-no-gain \
# 		--layer-scale $INV_LAYERS \
# 		--upscale-embedding $SQRT_MODELDIM \
# 		--qk-norm RMSNorm \
# 		--logits-layer-scale 1 --logits-layer-scale-scale 1 \
# 		--wd 0 --no-warmup --decay cos \
# 		--lr $lr \
# 		$*
# done


# source .env  # export WANDB_API_KEY and HF_TOKEN here.
# # 1/sqrt(L)
# INV_SQRT_L=0.28  # slightly less to avoid blow ups at initialization.
# INV_SQRTMODELDIM=0.044  # approx 1/sqrt(hidden_dim)
# INV_LAYERS=0.083  # approx 1/n_layers
# SQRT_MODELDIM=22.62  # approx sqrt(hidden_dim)

# # with pre norm, post norm and final, NO spherical training
# NODES=1
# for lr in 0.009 0.010; do
# 	bash submissions/submit.sh 110 --nodes $NODES \
# 		--post-norm --post-norm-no-gain \
# 		--layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
# 		--upscale-embedding $SQRT_MODELDIM \
# 		--qk-norm RMSNorm \
# 		--logits-layer-scale 1 --logits-layer-scale-scale 1 \
# 		--decay cos \
# 		--lr $lr \
# 		$*
# done



# source .env  # export WANDB_API_KEY and HF_TOKEN here.
# # 1/sqrt(L)
# INV_SQRT_L=0.28  # slightly less to avoid blow ups at initialization.
# INV_SQRTMODELDIM=0.044  # approx 1/sqrt(hidden_dim)
# INV_LAYERS=0.083  # approx 1/n_layers
# SQRT_MODELDIM=22.62  # approx sqrt(hidden_dim)

# # with pre norm, post norm and final, hypersphere training
# # and muon. one for embed and one for flat.
# NODES=2
# for mlr in 1 2 4 8; do
# 	bash submissions/submit.sh 110 --nodes $NODES \
# 		--opt master --master-orthogonalize --alpha 0 --hs embed --hs-embed --hs-embed-no-orthogonal \
# 		--b1 0.95 --mb1 0.9 --muon-scale shape_up \
# 		--post-norm --post-norm-no-gain \
# 		--layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
# 		--upscale-embedding $SQRT_MODELDIM \
# 		--qk-norm RMSNorm \
# 		--logits-layer-scale 1 --logits-layer-scale-scale 1 \
# 		--wd 0 --no-warmup --decay cos \
# 		--lr 0.003 --mlr $mlr \
# 		$*
# 	bash submissions/submit.sh 110 --nodes $NODES \
# 		--opt master --master-orthogonalize --alpha 0 --hs flat --hs-embed --hs-embed-no-orthogonal \
# 		--b1 0.95 --mb1 0.9 --muon-scale shape_up \
# 		--post-norm --post-norm-no-gain \
# 		--layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
# 		--upscale-embedding $SQRT_MODELDIM \
# 		--qk-norm RMSNorm \
# 		--logits-layer-scale 1 --logits-layer-scale-scale 1 \
# 		--wd 0 --no-warmup --decay cos \
# 		--lr 0.003 --mlr $mlr \
# 		$*
# done



# source .env  # export WANDB_API_KEY and HF_TOKEN here.
# # 1/sqrt(L)
# INV_SQRT_L=0.28  # slightly less to avoid blow ups at initialization.
# INV_SQRTMODELDIM=0.044  # approx 1/sqrt(hidden_dim)
# INV_LAYERS=0.083  # approx 1/n_layers
# SQRT_MODELDIM=22.62  # approx sqrt(hidden_dim)

# # with pre norm, post norm and final
# NODES=2
# for lr in 0.002 0.003 0.004 0.005; do
# 	bash submissions/submit.sh 110 --nodes $NODES \
# 		--opt master --alpha 0 --hs row --hs-embed \
# 		--post-norm --post-norm-no-gain \
# 		--layer-scale $INV_LAYERS \
# 		--upscale-embedding $SQRT_MODELDIM \
# 		--qk-norm RMSNorm \
# 		--logits-layer-scale 1 --logits-layer-scale-scale 1 \
# 		--wd 0 --no-warmup --decay cos \
# 		--lr $lr \
# 		$*
# done



source .env  # export WANDB_API_KEY and HF_TOKEN here.
# 1/sqrt(L)
INV_SQRT_L=0.28  # slightly less to avoid blow ups at initialization.
INV_SQRTMODELDIM=0.044  # approx 1/sqrt(hidden_dim)
INV_LAYERS=0.083  # approx 1/n_layers
SQRT_MODELDIM=22.62  # approx sqrt(hidden_dim)

# with pre norm, post norm and final, NO spherical training
NODES=2
# for lr in 0.002 0.003 0.004 0.005 0.006 0.007 0.008; do
# for lr in 0.010 0.012; do
# 	bash submissions/submit.sh 110 --nodes $NODES \
# 		--post-norm --post-norm-no-gain \
# 		--layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
# 		--upscale-embedding $SQRT_MODELDIM \
# 		--qk-norm RMSNorm \
# 		--logits-layer-scale 1 --logits-layer-scale-scale 1 \
# 		--decay linear \
# 		--lr $lr \
# 		$*
# done



# source .env  # export WANDB_API_KEY and HF_TOKEN here.
# # 1/sqrt(L)
# INV_SQRT_L=0.28  # slightly less to avoid blow ups at initialization.
# INV_SQRTMODELDIM=0.044  # approx 1/sqrt(hidden_dim)
# INV_LAYERS=0.083  # approx 1/n_layers
# SQRT_MODELDIM=22.62  # approx sqrt(hidden_dim)

# # with pre norm, post norm and final
# NODES=2
# for lr in 0.002 0.003 0.004 0.005; do
# 	bash submissions/submit.sh 110 --nodes $NODES \
# 		--opt master --alpha 0 --hs flat --hs-embed \
# 		--post-norm --post-norm-no-gain \
# 		--layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
# 		--upscale-embedding $SQRT_MODELDIM \
# 		--qk-norm RMSNorm \
# 		--logits-layer-scale 1 --logits-layer-scale-scale 1 \
# 		--wd 0 --decay isrwsd \
# 		--lr $lr \
# 		$*
# done

# source .env  # export WANDB_API_KEY and HF_TOKEN here.
# # 1/sqrt(L)
# INV_SQRT_L=0.28  # slightly less to avoid blow ups at initialization.
# INV_SQRTMODELDIM=0.044  # approx 1/sqrt(hidden_dim)
# INV_LAYERS=0.083  # approx 1/n_layers
# SQRT_MODELDIM=22.62  # approx sqrt(hidden_dim)

# # with pre norm, post norm and final, hypersphere training
# # and muon. inverse square root wsd
# NODES=2
# for mlr in 1 2 4 6 8; do
# 	bash submissions/submit.sh 110 --nodes $NODES \
# 		--opt master --master-orthogonalize --alpha 0 --hs flat --hs-embed --hs-embed-no-orthogonal \
# 		--b1 0.95 --mb1 0.9 --muon-scale shape_up \
# 		--post-norm --post-norm-no-gain \
# 		--layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
# 		--upscale-embedding $SQRT_MODELDIM \
# 		--qk-norm RMSNorm \
# 		--logits-layer-scale 1 --logits-layer-scale-scale 1 \
# 		--wd 0 --decay isrwsd \
# 		--lr 0.003 --mlr $mlr \
# 		$*
# done

# source .env  # export WANDB_API_KEY and HF_TOKEN here.
# # 1/sqrt(L)
# INV_SQRT_L=0.28  # slightly less to avoid blow ups at initialization.
# INV_SQRTMODELDIM=0.044  # approx 1/sqrt(hidden_dim)
# INV_LAYERS=0.083  # approx 1/n_layers
# SQRT_MODELDIM=22.62  # approx sqrt(hidden_dim)

# # with pre norm, post norm and final, hypersphere training
# # and muon.
# NODES=2
# for mlr in 1 2 4 6 8; do
# 	bash submissions/submit.sh 110 --nodes $NODES \
# 		--opt master --master-orthogonalize --alpha 0 --hs flat --hs-embed --hs-embed-no-orthogonal \
# 		--b1 0.95 --mb1 0.9 --muon-scale unit_rms_norm \
# 		--post-norm --post-norm-no-gain \
# 		--layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
# 		--upscale-embedding $SQRT_MODELDIM \
# 		--qk-norm RMSNorm \
# 		--logits-layer-scale 1 --logits-layer-scale-scale 1 \
# 		--wd 0 --no-warmup --decay cos \
# 		--lr 0.003 --mlr $mlr \
# 		$*
# done




source .env  # export WANDB_API_KEY and HF_TOKEN here.
# 1/sqrt(L)
INV_SQRT_L=0.28  # slightly less to avoid blow ups at initialization.
INV_SQRTMODELDIM=0.044  # approx 1/sqrt(hidden_dim)
INV_LAYERS=0.083  # approx 1/n_layers
SQRT_MODELDIM=22.62  # approx sqrt(hidden_dim)

# # with pre norm, post norm and final, hypersphere training muon
# # ! warmup with linear decay and NO final layernorm
# NODES=2
# for mlr in 1 2 4 6 8; do
# 	bash submissions/submit.sh 110 --nodes $NODES \
# 		--opt master --master-orthogonalize --alpha 0 --hs flat --hs-embed --hs-embed-no-orthogonal \
# 		--b1 0.95 --mb1 0.9 --muon-scale shape_up \
# 		--post-norm --post-norm-no-gain \
# 		--no-final-layernorm \
# 		--layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
# 		--upscale-embedding $SQRT_MODELDIM \
# 		--logits-layer-scale 1 \
# 		--qk-norm RMSNorm \
# 		--wd 0 --decay linear \
# 		--lr 0.003 --mlr $mlr \
# 		$*
# 	# same thing but with final layernorm and no gain
# 	bash submissions/submit.sh 110 --nodes $NODES \
# 		--opt master --master-orthogonalize --alpha 0 --hs flat --hs-embed --hs-embed-no-orthogonal \
# 		--b1 0.95 --mb1 0.9 --muon-scale shape_up \
# 		--post-norm --post-norm-no-gain \
# 		--final-layernorm-no-gain \
# 		--layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
# 		--upscale-embedding $SQRT_MODELDIM \
# 		--logits-layer-scale 1 \
# 		--qk-norm RMSNorm \
# 		--wd 0 --decay linear \
# 		--lr 0.003 --mlr $mlr \
# 		$*
# done


source .env  # export WANDB_API_KEY and HF_TOKEN here.
# 1/sqrt(L)
INV_SQRT_L=0.28  # slightly less to avoid blow ups at initialization.
INV_SQRTMODELDIM=0.044  # approx 1/sqrt(hidden_dim)
INV_LAYERS=0.083  # approx 1/n_layers
SQRT_MODELDIM=22.62  # approx sqrt(hidden_dim)

# with pre norm, post norm and final, hypersphere training muon
# ! warmup with linear decay, changing layerscale parameters
NODES=2
for mlr in 1 2 4 6 8; do
	# bash submissions/submit.sh 110 --nodes $NODES \
	# 	--opt master --master-orthogonalize --alpha 0 --hs flat --hs-embed --hs-embed-no-orthogonal \
	# 	--b1 0.95 --mb1 0.9 --muon-scale shape_up \
	# 	--post-norm --post-norm-no-gain \
	# 	--layer-scale $INV_LAYERS \
	# 	--upscale-embedding $SQRT_MODELDIM \
	# 	--logits-layer-scale 1 \
	# 	--qk-norm RMSNorm \
	# 	--wd 0 --decay linear \
	# 	--lr 0.003 --mlr $mlr \
	# 	$*
	# bash submissions/submit.sh 110 --nodes $NODES \
	# 	--opt master --master-orthogonalize --alpha 0 --hs flat --hs-embed --hs-embed-no-orthogonal \
	# 	--b1 0.95 --mb1 0.9 --muon-scale shape_up \
	# 	--post-norm --post-norm-no-gain \
	# 	--layer-scale $INV_SQRT_L --layer-scale-scale $INV_SQRTMODELDIM \
	# 	--upscale-embedding $SQRT_MODELDIM \
	# 	--logits-layer-scale 1 \
	# 	--qk-norm RMSNorm \
	# 	--wd 0 --decay linear \
	# 	--lr 0.003 --mlr $mlr \
	# 	$*
	# bash submissions/submit.sh 110 --nodes $NODES \
	# 	--opt master --master-orthogonalize --alpha 0 --hs flat --hs-embed --hs-embed-no-orthogonal \
	# 	--b1 0.95 --mb1 0.9 --muon-scale shape_up \
	# 	--post-norm --post-norm-no-gain \
	# 	--layer-scale 1 \
	# 	--upscale-embedding $SQRT_MODELDIM \
	# 	--logits-layer-scale 1 \
	# 	--qk-norm RMSNorm \
	# 	--wd 0 --decay linear \
	# 	--lr 0.003 --mlr $mlr \
	# 	$*
	bash submissions/submit.sh 110 --nodes $NODES \
		--opt master --master-orthogonalize --alpha 0 --hs flat --hs-embed --hs-embed-no-orthogonal \
		--b1 0.95 --mb1 0.9 --muon-scale shape_up \
		--post-norm --post-norm-no-gain \
		--layer-scale $INV_LAYERS --layer-scale-scale 1 \
		--upscale-embedding $SQRT_MODELDIM \
		--logits-layer-scale 1 \
		--qk-norm RMSNorm \
		--wd 0 --decay linear \
		--lr 0.003 --mlr $mlr \
		$*
	bash submissions/submit.sh 110 --nodes $NODES \
		--opt master --master-orthogonalize --alpha 0 --hs flat --hs-embed --hs-embed-no-orthogonal \
		--b1 0.95 --mb1 0.9 --muon-scale shape_up \
		--post-norm --post-norm-no-gain \
		--layer-scale $INV_SQRT_L --layer-scale-scale 1 \
		--upscale-embedding $SQRT_MODELDIM \
		--logits-layer-scale 1 \
		--qk-norm RMSNorm \
		--wd 0 --decay linear \
		--lr 0.003 --mlr $mlr \
		$*
done