source .env  # export WANDB_API_KEY and HF_TOKEN here.
# 1/sqrt(L)
INV_SQRT_L=0.28  # slightly less to avoid blow ups at initialization.
INV_SQRTMODELDIM=0.044  # approx 1/sqrt(hidden_dim)
INV_LAYERS=0.083  # approx 1/n_layers
SQRT_MODELDIM=22.62  # approx sqrt(hidden_dim)

# with pre norm, post norm and final, hypersphere training muon
# gains mode rowcol
NODES=2

source .env  # export WANDB_API_KEY and HF_TOKEN here.
INV_SQRT_L=0.28  # slightly less to avoid blow ups at initialization.
INV_LAYERS=0.083  # approx 1/n_layers
NODES=4

# # model_size inv_sqrt_modeldim sqrt_modeldim
# configs=(
# 	# "110 0.044  22.62"   # model dim: 512
#     # "193 0.036  27.71"   # model dim: 768
#     # "292 0.031  32.00"   # model dim: 1024
# 	"860 0.022  45.25"   # model dim: 2048
# # 
# )


# rowcol gains
# for config in "${configs[@]}"; do
#     read -r MODEL_SIZE INV_SQRTMODELDIM SQRT_MODELDIM <<< "$config"
# 	for lr in 0.001; do
# 		for k in 3 4 5 6 7 8; do
# 			# mlr as power of sqrt(2)
# 			mlr=$(python3 -c "print(2**($k/2))")
# 			echo "mlr: $mlr"
# 			bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
# 				--eval-every 1000 --eval-iters 50 \
# 				--opt master --master-orthogonalize --alpha 0 --hs flat --hs-embed row \
# 				--hs-embed-no-orthogonal \
# 				--b1 0.95 --mb1 0.9 --muon-scale shape_up \
# 				--post-norm --post-norm-no-gain \
# 				--layer-scale $INV_LAYERS --layer-scale-scale 1 \
# 				--upscale-embedding $SQRT_MODELDIM \
# 				--qk-norm RMSNorm \
# 				--wd 0 --decay linear \
# 				--no-warmup \
# 				--untie-embed \
# 				--hs-g rowcol \
# 				--lr $lr \
# 				--mlr $mlr \
# 				--elr 3 \
# 				$*
# 			# --no-warmup \
# 		done
# 	done
# done

configs=(
	"2 110 0.044  22.62"   # model dim: 512
    # "2 193 0.036  27.71"   # model dim: 768
    # "2 292 0.031  32.00"   # model dim: 1024
	# "4 860 0.022  45.25"   # model dim: 2048
)

# # no gains
# for config in "${configs[@]}"; do
#     read -r NODES MODEL_SIZE INV_SQRTMODELDIM SQRT_MODELDIM <<< "$config"
# 	for lr in 0.001; do
# 		for k in 3 4 5 6 7 8; do
# 			# mlr as power of sqrt(2)
# 			mlr=$(python3 -c "print(2**($k/2))")
# 			echo "mlr: $mlr"
# 			bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
# 				--eval-every 1000 --eval-iters 50 \
# 				--opt master --master-orthogonalize --alpha 0 --hs flat --hs-embed row \
# 				--hs-embed-no-orthogonal \
# 				--b1 0.95 --mb1 0.9 --muon-scale shape_up \
# 				--muon-nesterov \
# 				--post-norm \
# 				--fixed-layer-scale $INV_LAYERS \
# 				--upscale-embedding $SQRT_MODELDIM \
# 				--qk-norm RMSNorm \
# 				--wd 0 --decay linear \
# 				--no-warmup \
# 				--untie-embed \
# 				--lr $lr \
# 				--mlr $mlr \
# 				--elr 3 \
# 				$*
# 		done
# 	done
# done

# gains
for config in "${configs[@]}"; do
    read -r NODES MODEL_SIZE INV_SQRTMODELDIM SQRT_MODELDIM <<< "$config"
	for lr in 0.001; do
		for k in 3 4 5 6 7 8; do
			# mlr as power of sqrt(2)
			mlr=$(python3 -c "print(2**($k/2))")
			echo "mlr: $mlr"
			bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
				--eval-every 1000 --eval-iters 50 \
				--opt master --master-orthogonalize --alpha 0 --hs flat --hs-embed row \
				--hs-embed-no-orthogonal \
				--b1 0.95 --mb1 0.9 --muon-scale shape_up \
				--hs-g rowcol \
				--hs-g-embed none \
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
				$*
		done
	done
done


# gains with preserve init
for config in "${configs[@]}"; do
    read -r NODES MODEL_SIZE INV_SQRTMODELDIM SQRT_MODELDIM <<< "$config"
	for lr in 0.001; do
		for k in 3 4 5 6 7 8; do
			# mlr as power of sqrt(2)
			mlr=$(python3 -c "print(2**($k/2))")
			echo "mlr: $mlr"
			bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
				--eval-every 1000 --eval-iters 50 \
				--opt master --master-orthogonalize --alpha 0 --hs flat --hs-embed row \
				--hs-embed-no-orthogonal \
				--b1 0.95 --mb1 0.9 --muon-scale shape_up \
				--hs-g rowcol \
				--hs-g-embed none \
				--untie-embed \
				--lr $lr \
				--mlr $mlr \
				--elr 3 \
				--hs-preserve-init \
				--muon-nesterov \
				--post-norm \
				--fixed-layer-scale $INV_LAYERS \
				--upscale-embedding $SQRT_MODELDIM \
				--qk-norm RMSNorm \
				--wd 0 --decay linear \
				--no-warmup \
				$*
		done
	done
done

# # muon
# for config in "${configs[@]}"; do
#     read -r NODES MODEL_SIZE INV_SQRTMODELDIM SQRT_MODELDIM <<< "$config"
# 	for lr in 0.001; do
# 		for k in 3 4 5 6 7 8; do
# 			# mlr as power of sqrt(2)
# 			mlr=$(python3 -c "print(2**($k/2))")
# 			echo "mlr: $mlr"
# 			bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
# 				--eval-every 1000 --eval-iters 50 \
# 				--opt master --master-orthogonalize --alpha 0 \
# 				--b1 0.95 --mb1 0.9 --muon-scale unit_rms_norm \
# 				--muon-nesterov \
# 				--post-norm \
# 				--fixed-layer-scale $INV_LAYERS \
# 				--upscale-embedding $SQRT_MODELDIM \
# 				--qk-norm RMSNorm \
# 				--wd 0.1 --decay linear \
# 				--warmup-iters 1000 \
# 				--untie-embed \
# 				--lr $lr \
# 				--mlr $mlr \
# 				--elr 3 \
# 				$*
# 		done
# 	done
# done

# # adam
# for config in "${configs[@]}"; do
#     read -r NODES MODEL_SIZE INV_SQRTMODELDIM SQRT_MODELDIM <<< "$config"
# 	for lr in 0.001; do
# 		for k in 3 4 5 6 7 8; do
# 			# mlr as power of sqrt(2)
# 			mlr=$(python3 -c "print(2**($k/2))")
# 			echo "mlr: $mlr"
# 			bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
# 				--eval-every 1000 --eval-iters 50 \
# 				--opt master --alpha 0 \
# 				--post-norm \
# 				--fixed-layer-scale $INV_LAYERS \
# 				--upscale-embedding $SQRT_MODELDIM \
# 				--qk-norm RMSNorm \
# 				--wd 0.1 --decay linear \
# 				--warmup-iters 1000 \
# 				--untie-embed \
# 				--lr $lr \
# 				--mlr $mlr \
# 				--elr 3 \
# 				$*
# 		done
# 	done
# done



# # rowcol gains, output layer none
# for config in "${configs[@]}"; do
#     read -r MODEL_SIZE INV_SQRTMODELDIM SQRT_MODELDIM <<< "$config"
# 	for lr in 0.001; do
# 		for k in 3 4 5 6 7 8; do
# 			# mlr as power of sqrt(2)
# 			mlr=$(python3 -c "print(2**($k/2))")
# 			echo "mlr: $mlr"
# 			bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
# 				--eval-every 1000 --eval-iters 50 \
# 				--opt master --master-orthogonalize --alpha 0 --hs flat --hs-embed row \
# 				--hs-embed-no-orthogonal \
# 				--b1 0.95 --mb1 0.9 --muon-scale shape_up \
# 				--post-norm --post-norm-no-gain \
# 				--layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
# 				--upscale-embedding $SQRT_MODELDIM \
# 				--qk-norm RMSNorm \
# 				--wd 0 --decay linear \
# 				--no-warmup \
# 				--untie-embed \
# 				--hs-g rowcol \
# 				--hs-g-output none \
# 				--lr $lr \
# 				--mlr $mlr \
# 				--elr 3 \
# 				$*
# 			# --no-warmup \
# 		done
# 	done
# done

# rowcol gains, output layer none
# for config in "${configs[@]}"; do
#     read -r MODEL_SIZE INV_SQRTMODELDIM SQRT_MODELDIM <<< "$config"
# 	for lr in 0.001; do
# 		for k in 3 4 5 6 7 8; do
# 			# mlr as power of sqrt(2)
# 			mlr=$(python3 -c "print(2**($k/2))")
# 			echo "mlr: $mlr"
# 			bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
# 				--eval-every 1000 --eval-iters 50 \
# 				--opt master --master-orthogonalize --alpha 0 --hs flat --hs-embed row \
# 				--hs-embed-no-orthogonal \
# 				--b1 0.95 --mb1 0.9 --muon-scale shape_up \
# 				--post-norm --post-norm-no-gain \
# 				--layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
# 				--upscale-embedding $SQRT_MODELDIM \
# 				--qk-norm RMSNorm \
# 				--wd 0 --decay linear \
# 				--no-warmup \
# 				--untie-embed \
# 				--hs-g rowcol \
# 				--hs-g-output row \
# 				--lr $lr \
# 				--mlr $mlr \
# 				--elr 3 \
# 				$*
# 			# --no-warmup \
# 		done
# 	done
# done


# pre norm no gain
# for config in "${configs[@]}"; do
#     read -r MODEL_SIZE INV_SQRTMODELDIM SQRT_MODELDIM <<< "$config"
# 	for lr in 0.001; do
# 		for k in 3 4 5 6 7 8; do
# 			# mlr as power of sqrt(2)
# 			mlr=$(python3 -c "print(2**($k/2))")
# 			echo "mlr: $mlr"
# 			bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
# 				--eval-every 1000 --eval-iters 50 \
# 				--opt master --master-orthogonalize --alpha 0 --hs flat --hs-embed row \
# 				--hs-embed-no-orthogonal \
# 				--b1 0.95 --mb1 0.9 --muon-scale shape_up \
# 				--post-norm --post-norm-no-gain \
# 				--layer-scale $INV_LAYERS --layer-scale-scale 1 \
# 				--upscale-embedding $SQRT_MODELDIM \
# 				--qk-norm RMSNorm \
# 				--wd 0 --decay linear \
# 				--no-warmup \
# 				--untie-embed \
# 				--hs-g rowcol \
# 				--pre-norm-no-gain \
# 				--lr $lr \
# 				--mlr $mlr \
# 				--elr 3 \
# 				$*
# 			# --no-warmup \
# 		done
# 	done
# done

# embed
# for config in "${configs[@]}"; do
#     read -r MODEL_SIZE INV_SQRTMODELDIM SQRT_MODELDIM <<< "$config"
# 	for lr in 0.001; do
# 		for k in 3 4 5 6 7 8; do
# 			# mlr as power of sqrt(2)
# 			mlr=$(python3 -c "print(2**($k/2))")
# 			echo "mlr: $mlr"
# 			bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
# 				--eval-every 1000 --eval-iters 50 \
# 				--opt master --master-orthogonalize --alpha 0 --hs embed --hs-embed row \
# 				--hs-embed-no-orthogonal \
# 				--b1 0.95 --mb1 0.9 --muon-scale shape_up \
# 				--post-norm --post-norm-no-gain \
# 				--layer-scale $INV_LAYERS --layer-scale-scale 1 \
# 				--upscale-embedding $SQRT_MODELDIM \
# 				--qk-norm RMSNorm \
# 				--wd 0 --decay linear \
# 				--no-warmup \
# 				--untie-embed \
# 				--hs-g embed \
# 				--lr $lr \
# 				--mlr $mlr \
# 				--elr 3 \
# 				$*
# 			# --no-warmup \
# 		done
# 	done
# done


# # no postnorm
# for config in "${configs[@]}"; do
#     read -r MODEL_SIZE INV_SQRTMODELDIM SQRT_MODELDIM <<< "$config"
# 	for lr in 0.001; do
# 		for k in 3 4 5 6 7 8; do
# 			# mlr as power of sqrt(2)
# 			mlr=$(python3 -c "print(2**($k/2))")
# 			echo "mlr: $mlr"
# 			bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
# 				--eval-every 1000 --eval-iters 50 \
# 				--opt master --master-orthogonalize --alpha 0 --hs flat --hs-embed row \
# 				--hs-embed-no-orthogonal \
# 				--b1 0.95 --mb1 0.9 --muon-scale shape_up \
# 				--layer-scale $INV_LAYERS --layer-scale-scale 1 \
# 				--upscale-embedding $SQRT_MODELDIM \
# 				--qk-norm RMSNorm \
# 				--wd 0 --decay linear \
# 				--no-warmup \
# 				--untie-embed \
# 				--hs-g rowcol \
# 				--lr $lr \
# 				--mlr $mlr \
# 				--elr 3 \
# 				$*
# 			# --no-warmup \
# 		done
# 	done
# done



# no prenorm
# for config in "${configs[@]}"; do
#     read -r MODEL_SIZE INV_SQRTMODELDIM SQRT_MODELDIM <<< "$config"
# 	for lr in 0.001; do
# 		for k in 3 4 5 6 7 8; do
# 			# mlr as power of sqrt(2)
# 			mlr=$(python3 -c "print(2**($k/2))")
# 			echo "mlr: $mlr"
# 			bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
# 				--eval-every 1000 --eval-iters 50 \
# 				--opt master --master-orthogonalize --alpha 0 --hs flat --hs-embed row \
# 				--hs-embed-no-orthogonal \
# 				--b1 0.95 --mb1 0.9 --muon-scale shape_up \
# 				--layer-scale $INV_LAYERS --layer-scale-scale 1 \
# 				--upscale-embedding $SQRT_MODELDIM \
# 				--qk-norm RMSNorm \
# 				--post-norm --post-norm-no-gain \
# 				--no-pre-norm \
# 				--wd 0 --decay linear \
# 				--no-warmup \
# 				--untie-embed \
# 				--hs-g rowcol \
# 				--lr $lr \
# 				--mlr $mlr \
# 				--elr 3 \
# 				$*
# 			# --no-warmup \
# 		done
# 	done
# done

# for config in "${configs[@]}"; do
#     read -r MODEL_SIZE INV_SQRTMODELDIM SQRT_MODELDIM <<< "$config"
# 	for lr in 0.001; do
# 		for k in 3 4 5 6 7 8; do
# 			# mlr as power of sqrt(2)
# 			mlr=$(python3 -c "print(2**($k/2))")
# 			echo "mlr: $mlr"
# 			bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
# 				--eval-every 1000 --eval-iters 50 \
# 				--opt master --master-orthogonalize --alpha 0 --hs flat --hs-embed row \
# 				--hs-embed-no-orthogonal \
# 				--b1 0.95 --mb1 0.9 --muon-scale shape_up \
# 				--layer-scale $INV_LAYERS --layer-scale-scale 1 \
# 				--upscale-embedding $SQRT_MODELDIM \
# 				--qk-norm RMSNorm \
# 				--post-norm --post-norm-no-gain \
# 				--wd 0 --decay linear \
# 				--no-warmup \
# 				--untie-embed \
# 				--hs-g row \
# 				--lr $lr \
# 				--mlr $mlr \
# 				--elr 3 \
# 				$*
# 			# --no-warmup \
# 		done
# 	done
# done

# # no clipping
# for config in "${configs[@]}"; do
#     read -r MODEL_SIZE INV_SQRTMODELDIM SQRT_MODELDIM <<< "$config"
# 	for lr in 0.001; do
# 		for k in 3 4 5 6 7 8; do
# 			# mlr as power of sqrt(2)
# 			mlr=$(python3 -c "print(2**($k/2))")
# 			echo "mlr: $mlr"
# 			bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
# 				--eval-every 1000 --eval-iters 50 \
# 				--opt master --master-orthogonalize --alpha 0 --hs flat --hs-embed row \
# 				--hs-embed-no-orthogonal \
# 				--b1 0.95 --mb1 0.9 --muon-scale shape_up \
# 				--layer-scale $INV_LAYERS --layer-scale-scale 1 \
# 				--upscale-embedding $SQRT_MODELDIM \
# 				--qk-norm RMSNorm \
# 				--post-norm --post-norm-no-gain \
# 				--wd 0 --decay linear \
# 				--no-warmup \
# 				--untie-embed \
# 				--hs-g rowcol \
# 				--clip-grad 0 \
# 				--lr $lr \
# 				--mlr $mlr \
# 				--elr 3 \
# 				$*
# 			# --no-warmup \
# 		done
# 	done
# done



# no embedding gains
# for config in "${configs[@]}"; do
#     read -r MODEL_SIZE INV_SQRTMODELDIM SQRT_MODELDIM <<< "$config"
# 	for lr in 0.001; do
# 		for k in 3 4 5 6 7 8; do
# 			# mlr as power of sqrt(2)
# 			mlr=$(python3 -c "print(2**($k/2))")
# 			echo "mlr: $mlr"
# 			bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
# 				--eval-every 1000 --eval-iters 50 \
# 				--opt master --master-orthogonalize --alpha 0 --hs flat --hs-embed row \
# 				--hs-embed-no-orthogonal \
# 				--b1 0.95 --mb1 0.9 --muon-scale shape_up \
# 				--layer-scale $INV_LAYERS --layer-scale-scale 1 \
# 				--upscale-embedding $SQRT_MODELDIM \
# 				--qk-norm RMSNorm \
# 				--post-norm --post-norm-no-gain \
# 				--wd 0 --decay linear \
# 				--no-warmup \
# 				--untie-embed \
# 				--hs-g rowcol \
# 				--hs-g-embed none \
# 				--lr $lr \
# 				--mlr $mlr \
# 				--elr 3 \
# 				$*
# 			# --no-warmup \
# 		done
# 	done
# done



# for config in "${configs[@]}"; do
#     read -r MODEL_SIZE INV_SQRTMODELDIM SQRT_MODELDIM <<< "$config"
# 	for lr in 0.001; do
# 		for k in 3 4 5 6 7 8; do
# 			# mlr as power of sqrt(2)
# 			mlr=$(python3 -c "print(2**($k/2))")
# 			echo "mlr: $mlr"
# 			bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
# 				--eval-every 1000 --eval-iters 50 \
# 				--opt master --master-orthogonalize --alpha 0 --hs flat --hs-embed row \
# 				--hs-embed-no-orthogonal \
# 				--b1 0.95 --mb1 0.9 --muon-scale shape_up \
# 				--layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
# 				--upscale-embedding $SQRT_MODELDIM \
# 				--qk-norm RMSNorm \
# 				--post-norm --post-norm-no-gain \
# 				--wd 0 --decay wsd \
# 				--no-warmup \
# 				--untie-embed \
# 				--hs-g rowcol \
# 				--hs-g-embed none \
# 				--lr $lr \
# 				--mlr $mlr \
# 				--elr 3 \
# 				$*
# 			# --no-warmup \
# 		done
# 	done
# done


# for config in "${configs[@]}"; do
#     read -r MODEL_SIZE INV_SQRTMODELDIM SQRT_MODELDIM <<< "$config"
# 	for lr in 0.001; do
# 		for k in 3 4 5 6 7 8; do
# 		# for k in 7 8; do
# 			# mlr as power of sqrt(2)
# 			mlr=$(python3 -c "print(2**($k/2))")
# 			echo "mlr: $mlr"
# 			bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
# 				--eval-every 1000 --eval-iters 50 \
# 				--opt master --master-orthogonalize --alpha 0 --hs flat --hs-embed row \
# 				--hs-embed-no-orthogonal \
# 				--b1 0.95 --mb1 0.9 --muon-scale shape_up \
# 				--layer-scale $INV_LAYERS --layer-scale-scale 0.03 \
# 				--upscale-embedding $SQRT_MODELDIM \
# 				--qk-norm RMSNorm \
# 				--post-norm --post-norm-no-gain \
# 				--wd 0 --decay linear \
# 				--no-warmup \
# 				--untie-embed \
# 				--hs-g rowcol \
# 				--hs-g-embed none \
# 				--clip-grad 0 \
# 				--lr $lr \
# 				--mlr $mlr \
# 				--elr 3 \
# 				$*
# 			# --no-warmup \
# 		done
# 	done
# done

