source .env  # export WANDB_API_KEY and HF_TOKEN here.
# Exp 9: muon shape-scaling mode comparison on 110M (master-muon).
# Vary --muon-scale: spectral, shape_scaling, unit_rms_norm, shape_up.
# Sweep absolute matrix LR per mode; all other settings fixed.

INV_LAYERS=0.083
SQRT_MODELDIM=22.62
NODES=2
MODEL_SIZE=110

BASE_LR=0.001
EMB_LR=0.003


# master-muon, no gains
# for scale in shape_scaling unit_rms_norm; do
# 	for k in 3 4 5 6 7 8; do
# 		matrix_lr=$(python3 -c "print(0.001 * 2**($k/2))")
# 		bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
# 			--eval-every 1000 --eval-iters 50 \
# 			--opt master --master-orthogonalize --alpha 0 \
# 			--hs flat --hs-embed row --hs-embed-no-orthogonal \
# 			--b1 0.95 --mb1 0.9 --muon-scale $scale --muon-nesterov \
# 			--post-norm \
# 			--fixed-layer-scale $INV_LAYERS \
# 			--upscale-embedding $SQRT_MODELDIM \
# 			--qk-norm RMSNorm \
# 			--wd 0 --decay linear --no-warmup \
# 			--untie-embed \
# 			--lr $BASE_LR --matrix-lr $matrix_lr --embedding-lr $EMB_LR \
# 			--extra-name exp9-master-scale \
# 			$*
# 	done
# done


# master-muon, no gains
# for k in 3 4 5 6 7 8; do
# 	matrix_lr=$(python3 -c "print(0.001 * 2**($k/2))")
# 	bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
# 		--eval-every 1000 --eval-iters 50 \
# 		--opt master --master-orthogonalize --alpha 0 \
# 		--hs flat --hs-embed row --hs-embed-no-orthogonal \
# 		--b1 0.95 --mb1 0.9 --muon-scale spectral  --muon-extra-scale-factor 0.2 --muon-nesterov \
# 		--post-norm \
# 		--fixed-layer-scale $INV_LAYERS \
# 		--upscale-embedding $SQRT_MODELDIM \
# 		--qk-norm RMSNorm \
# 		--wd 0 --decay linear --no-warmup \
# 		--untie-embed \
# 		--lr $BASE_LR --matrix-lr $matrix_lr --embedding-lr $EMB_LR \
# 		--extra-name exp9-master-scale \
# 		$*
# done

# # ---- muon (master + orthogonalize, no hypersphere) ----
# for scale in shape_scaling unit_rms_norm; do
# 	for k in 3 4 5 6 7 8; do
# 		matrix_lr=$(python3 -c "print(0.001 * 2**($k/2))")
# 		bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
# 			--eval-every 1000 --eval-iters 50 \
# 			--opt master --master-orthogonalize --alpha 0 \
# 			--b1 0.95 --mb1 0.9 --muon-scale $scale --muon-nesterov \
# 			--post-norm \
# 			--fixed-layer-scale $INV_LAYERS \
# 			--upscale-embedding $SQRT_MODELDIM \
# 			--qk-norm RMSNorm \
# 			--wd 0.1 --decay linear \
# 			--warmup-iters 1000 \
# 			--untie-embed \
# 			--lr $BASE_LR --matrix-lr $matrix_lr --embedding-lr $EMB_LR \
# 			--extra-name exp9-muon-scale \
# 			$*
# 	done
# done

# # muon (master + orthogonalize, no hypersphere)
# spectral with warmup
# for k in 3 4 5 6 7 8; do
# 	matrix_lr=$(python3 -c "print(0.001 * 2**($k/2))")
# 	bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
# 		--eval-every 1000 --eval-iters 50 \
# 		--opt master --master-orthogonalize --alpha 0 \
# 		--b1 0.95 --mb1 0.9 --muon-scale spectral  --muon-extra-scale-factor 0.2 --muon-nesterov \
# 		--post-norm \
# 		--fixed-layer-scale $INV_LAYERS \
# 		--upscale-embedding $SQRT_MODELDIM \
# 		--qk-norm RMSNorm \
# 		--wd 0.1 --decay linear --warmup-iters 1000 \
# 		--untie-embed \
# 		--lr $BASE_LR --matrix-lr $matrix_lr --embedding-lr $EMB_LR \
# 		--extra-name exp9-muon-scale \
# 		$*
# done

# muon (master + orthogonalize, no hypersphere)
# spectral no warmup
# for k in 3 4 5 6 7 8; do
# 	matrix_lr=$(python3 -c "print(0.001 * 2**($k/2))")
# 	bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
# 		--eval-every 1000 --eval-iters 50 \
# 		--opt master --master-orthogonalize --alpha 0 \
# 		--b1 0.95 --mb1 0.9 --muon-scale spectral  --muon-extra-scale-factor 0.2 --muon-nesterov \
# 		--post-norm \
# 		--fixed-layer-scale $INV_LAYERS \
# 		--upscale-embedding $SQRT_MODELDIM \
# 		--qk-norm RMSNorm \
# 		--wd 0.1 --decay linear --no-warmup \
# 		--untie-embed \
# 		--lr $BASE_LR --matrix-lr $matrix_lr --embedding-lr $EMB_LR \
# 		--extra-name exp9-muon-scale-nw \
# 		$*
# done


# shape scaling no warmup
for k in 3 4 5 6 7 8; do
	matrix_lr=$(python3 -c "print(0.001 * 2**($k/2))")
	bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
		--eval-every 1000 --eval-iters 50 \
		--opt master --master-orthogonalize --alpha 0 \
		--b1 0.95 --mb1 0.9 --muon-scale shape_scaling  --muon-nesterov \
		--post-norm \
		--fixed-layer-scale $INV_LAYERS \
		--upscale-embedding $SQRT_MODELDIM \
		--qk-norm RMSNorm \
		--wd 0.1 --decay linear --no-warmup \
		--untie-embed \
		--lr $BASE_LR --matrix-lr $matrix_lr --embedding-lr $EMB_LR \
		--extra-name exp9-muon-scale-nw \
		$*
done

