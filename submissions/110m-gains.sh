source .env  # export WANDB_API_KEY and HF_TOKEN here.
# 1/sqrt(L)
INV_SQRT_L=0.28  # slightly less to avoid blow ups at initialization.
INV_SQRTMODELDIM=0.044  # approx 1/sqrt(hidden_dim)
INV_LAYERS=0.083  # approx 1/n_layers
SQRT_MODELDIM=22.62  # approx sqrt(hidden_dim)

# with pre norm, post norm and final, hypersphere training muon
# gains mode rowcol
NODES=2
# for mlr in 1 2 4 6 8; do
#     bash submissions/submit.sh 110 --nodes $NODES \
#         --opt master --master-orthogonalize --alpha 0 --hs flat --hs-embed --hs-embed-no-orthogonal \
#         --b1 0.95 --mb1 0.9 --muon-scale shape_up \
#         --post-norm --post-norm-no-gain \
#         --layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
#         --upscale-embedding $SQRT_MODELDIM \
#         --qk-norm RMSNorm \
#         --wd 0 --decay cos --no-warmup \
#         --hs-g rowcol \
# 		--lr 0.003 --mlr $mlr \
# 		$*
# done


# warmup 1000
# for lr in 0.001 0.002 0.003 0.004; do
# 	for mlr in 1 2 4 6 8; do
# 		bash submissions/submit.sh 110 --nodes $NODES \
# 			--opt master --master-orthogonalize --alpha 0 --hs flat --hs-embed row \
# 			--hs-embed-no-orthogonal \
# 			--b1 0.95 --mb1 0.9 --muon-scale shape_up \
# 			--post-norm --post-norm-no-gain \
# 			--layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
# 			--upscale-embedding $SQRT_MODELDIM \
# 			--qk-norm RMSNorm \
# 			--wd 0 --decay linear \
# 			--warmup-iters 1000 \
#             --hs-g rowcol \
# 			--lr $lr \
# 			--mlr $mlr \
# 			$*
# 	done
# done


# flat gain, ie just one scaling factor
# for lr in 0.003; do
# 	for mlr in 1 2 4 6 8; do
# 		bash submissions/submit.sh 110 --nodes $NODES \
# 			--opt master --master-orthogonalize --alpha 0 --hs flat --hs-embed row \
# 			--hs-embed-no-orthogonal \
# 			--b1 0.95 --mb1 0.9 --muon-scale shape_up \
# 			--post-norm --post-norm-no-gain \
# 			--layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
# 			--upscale-embedding $SQRT_MODELDIM \
# 			--qk-norm RMSNorm \
# 			--wd 0 --decay linear \
# 			--warmup-iters 1000 \
#             --hs-g flat \
# 			--lr $lr \
# 			--mlr $mlr \
# 			$*
# 	done
# done


# pre norm no gain
# for lr in 0.003; do
# 	for mlr in 1 2 4 6 8; do
# 		bash submissions/submit.sh 110 --nodes $NODES \
# 			--opt master --master-orthogonalize --alpha 0 --hs flat --hs-embed row \
# 			--hs-embed-no-orthogonal \
# 			--b1 0.95 --mb1 0.9 --muon-scale shape_up \
# 			--post-norm --post-norm-no-gain \
# 			--pre-norm-no-gain \
# 			--layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
# 			--upscale-embedding $SQRT_MODELDIM \
# 			--qk-norm RMSNorm \
# 			--wd 0 --decay linear \
# 			--warmup-iters 1000 \
#             --hs-g rowcol \
# 			--lr $lr \
# 			--mlr $mlr \
# 			$*
# 	done
# done


# # split qkv gains, pre norm no gain
# for lr in 0.003; do
# 	for mlr in 1 2 4 6 8; do
# 		bash submissions/submit.sh 110 --nodes $NODES \
# 			--opt master --master-orthogonalize --alpha 0 --hs flat --hs-embed row \
# 			--hs-embed-no-orthogonal \
# 			--b1 0.95 --mb1 0.9 --muon-scale shape_up \
# 			--post-norm --post-norm-no-gain \
# 			--pre-norm-no-gain \
# 			--layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
# 			--upscale-embedding $SQRT_MODELDIM \
# 			--qk-norm RMSNorm \
# 			--wd 0 --decay linear \
# 			--warmup-iters 1000 \
#             --hs-g rowcol \
# 			--lr $lr \
# 			--mlr $mlr \
# 			--split-qkv-gains \
# 			$*
# 	done
# done

#  row gains
# for lr in 0.003; do
# 	for mlr in 1 2 4 6 8; do
# 		bash submissions/submit.sh 110 --nodes $NODES \
# 			--opt master --master-orthogonalize --alpha 0 --hs flat --hs-embed row \
# 			--hs-embed-no-orthogonal \
# 			--b1 0.95 --mb1 0.9 --muon-scale shape_up \
# 			--post-norm --post-norm-no-gain \
# 			--layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
# 			--upscale-embedding $SQRT_MODELDIM \
# 			--qk-norm RMSNorm \
# 			--wd 0 --decay linear \
# 			--warmup-iters 1000 \
#             --hs-g row \
# 			--lr $lr \
# 			--mlr $mlr \
# 			$*
# 	done
# done


#  row gains: go down with gains LR by a factor of 3x, keep embedding LR same
# for lr in 0.001; do
# 	for mlr in 1 2 4 6 8; do
# 		bash submissions/submit.sh 110 --nodes $NODES \
# 			--opt master --master-orthogonalize --alpha 0 --hs flat --hs-embed row \
# 			--hs-embed-no-orthogonal \
# 			--b1 0.95 --mb1 0.9 --muon-scale shape_up \
# 			--post-norm --post-norm-no-gain \
# 			--layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
# 			--upscale-embedding $SQRT_MODELDIM \
# 			--qk-norm RMSNorm \
# 			--wd 0 --decay linear \
# 			--warmup-iters 1000 \
#             --hs-g row \
# 			--lr $lr \
# 			--mlr $mlr \
# 			--elr 3.0 \
# 			$*
# 	done
# done


# same just redo with higher mlr
# for lr in 0.001; do
# 	for mlr in 12 16; do
# 		bash submissions/submit.sh 110 --nodes $NODES \
# 			--opt master --master-orthogonalize --alpha 0 --hs flat --hs-embed row \
# 			--hs-embed-no-orthogonal \
# 			--b1 0.95 --mb1 0.9 --muon-scale shape_up \
# 			--post-norm --post-norm-no-gain \
# 			--layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
# 			--upscale-embedding $SQRT_MODELDIM \
# 			--qk-norm RMSNorm \
# 			--wd 0 --decay linear \
# 			--warmup-iters 1000 \
#             --hs-g row \
# 			--lr $lr \
# 			--mlr $mlr \
# 			--elr 3.0 \
# 			$*
# 	done
# done


# #  row gains without clipping?
# for lr in 0.003; do
# 	for mlr in 1 8; do
# 		bash submissions/submit.sh 110 --nodes $NODES \
# 			--opt master --master-orthogonalize --alpha 0 --hs flat --hs-embed row \
# 			--hs-embed-no-orthogonal \
# 			--b1 0.95 --mb1 0.9 --muon-scale shape_up \
# 			--post-norm --post-norm-no-gain \
# 			--layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
# 			--upscale-embedding $SQRT_MODELDIM \
# 			--qk-norm RMSNorm \
# 			--wd 0 --decay linear \
# 			--warmup-iters 1000 \
#             --hs-g row \
# 			--lr $lr \
# 			--mlr $mlr \
# 			--clip-grad 0 \
# 			$*
# 	done
# done

# row gains with adam
# for lr in 0.002 0.003 0.004; do
# 	for mlr in 1; do
# 		bash submissions/submit.sh 110 --nodes $NODES \
# 			--opt master --alpha 0 --hs flat --hs-embed row \
# 			--hs-embed-no-orthogonal \
# 			--post-norm --post-norm-no-gain \
# 			--layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
# 			--upscale-embedding $SQRT_MODELDIM \
# 			--qk-norm RMSNorm \
# 			--wd 0 --decay linear \
# 			--warmup-iters 1000 \
#             --hs-g row \
# 			--lr $lr \
# 			$*
# 	done
# done


#  go down with gains LR by a factor of 3x, keep embedding LR same. rowcol gains!
# for lr in 0.001; do
# 	for k in 3 4 5 6 7 8; do
# 		# mlr as power of sqrt(2)
# 		mlr=$(python3 -c "print(2**($k/2))")
# 		echo "mlr: $mlr"
# 		bash submissions/submit.sh 110 --nodes $NODES \
# 			--eval-every 1000 --eval-iters 50 \
# 			--opt master --master-orthogonalize --alpha 0 --hs flat --hs-embed row \
# 			--hs-embed-no-orthogonal \
# 			--b1 0.95 --mb1 0.9 --muon-scale shape_up \
# 			--post-norm --post-norm-no-gain \
# 			--layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
# 			--upscale-embedding $SQRT_MODELDIM \
# 			--qk-norm RMSNorm \
# 			--wd 0 --decay linear \
# 			--no-warmup \
# 			--untie-embed \
#             --hs-g rowcol \
# 			--lr $lr \
# 			--mlr $mlr \
# 			--elr 3 \
# 			$*
# 	done
# done

# no gains
# for lr in 0.001; do
# 	for k in 3 4 5 6 7 8; do
# 		# mlr as power of sqrt(2)
# 		mlr=$(python3 -c "print(2**($k/2))")
# 		echo "mlr: $mlr"
# 		bash submissions/submit.sh 110 --nodes $NODES \
# 			--eval-every 1000 --eval-iters 50 \
# 			--opt master --master-orthogonalize --alpha 0 --hs flat --hs-embed row \
# 			--hs-embed-no-orthogonal \
# 			--b1 0.95 --mb1 0.9 --muon-scale shape_up \
# 			--post-norm --post-norm-no-gain \
# 			--layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
# 			--upscale-embedding $SQRT_MODELDIM \
# 			--qk-norm RMSNorm \
# 			--wd 0 --decay linear \
# 			--no-warmup \
# 			--untie-embed \
# 			--lr $lr \
# 			--mlr $mlr \
# 			--elr 3 \
# 			$*
# 	done
# done


# base muon
for lr in 0.001; do
	for k in 3 4 5 6 7 8; do
		# mlr as power of sqrt(2)
		mlr=$(python3 -c "print(2**($k/2))")
		echo "mlr: $mlr"
		bash submissions/submit.sh 110 --nodes $NODES \
			--eval-every 1000 --eval-iters 50 \
			--opt master --master-orthogonalize --alpha 0 \
			--b1 0.95 --mb1 0.9 --muon-scale unit_rms_norm \
			--post-norm --post-norm-no-gain \
			--layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
			--upscale-embedding $SQRT_MODELDIM \
			--qk-norm RMSNorm \
			--wd 0.1 --decay linear \
			--warmup-iters 1000 \
			--untie-embed \
			--lr $lr \
			--mlr $mlr \
			--elr 3 \
			$*
	done
done


# base adam
# for lr in 0.001; do
# 	for k in 3 4 5 6 7 8; do
# 		# mlr as power of sqrt(2)
# 		mlr=$(python3 -c "print(2**($k/2))")
# 		echo "mlr: $mlr"
# 		bash submissions/submit.sh 110 --nodes $NODES \
# 			--eval-every 1000 --eval-iters 50 \
# 			--opt master --alpha 0 \
# 			--post-norm --post-norm-no-gain \
# 			--layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
# 			--upscale-embedding $SQRT_MODELDIM \
# 			--qk-norm RMSNorm \
# 			--wd 0.1 --decay linear \
# 			--warmup-iters 1000 \
# 			--untie-embed \
# 			--lr $lr \
# 			--mlr $mlr \
# 			--elr 3 \
# 			$*
# 	done
# done