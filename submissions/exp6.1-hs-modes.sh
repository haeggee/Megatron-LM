source .env  # export WANDB_API_KEY and HF_TOKEN here.
# Exp 6: hypersphere modes comparison on 110M (master-muon).
# Vary the parameter-sphere mode (--hs) across {flat, row, col, rowcol, invrowcol}.
# Two loops: WITH gains (our method) and WITHOUT gains. Comment one out if needed.

INV_LAYERS=0.083
SQRT_MODELDIM=22.62
NODES=2
MODEL_SIZE=110

BASE_LR=0.001
EMB_LR=0.003

# ---- with gains ----
# for hsmode in flat row col rowcol invrowcol; do
# 	for k in 3 4 5 6 7 8; do
# 		matrix_lr=$(python3 -c "print(0.001 * 2**($k/2))")
# 		bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
# 			--eval-every 1000 --eval-iters 50 \
# 			--opt master --master-orthogonalize --alpha 0 \
# 			--hs $hsmode --hs-embed row --hs-embed-no-orthogonal \
# 			--b1 0.95 --mb1 0.9 --muon-scale shape_up --muon-nesterov \
# 			--hs-g rowcol --hs-g-embed none \
# 			--post-norm \
# 			--fixed-layer-scale $INV_LAYERS \
# 			--upscale-embedding $SQRT_MODELDIM \
# 			--qk-norm RMSNorm \
# 			--wd 0 --decay linear --no-warmup \
# 			--untie-embed \
# 			--lr $BASE_LR --matrix-lr $matrix_lr --embedding-lr $EMB_LR \
# 			--extra-name exp6-hs-${hsmode}-gains \
# 			$*
# 	done
# done

# ---- without gains ----
for hsmode in row embed; do
	for k in 3 4 5 6 7 8; do
		matrix_lr=$(python3 -c "print(0.001 * 2**($k/2))")
		bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
			--eval-every 1000 --eval-iters 50 \
			--opt master --master-orthogonalize --alpha 0 \
			--hs $hsmode --hs-embed row --hs-embed-no-orthogonal \
			--b1 0.95 --mb1 0.9 --muon-scale shape_up --muon-nesterov \
			--post-norm \
			--fixed-layer-scale $INV_LAYERS \
			--upscale-embedding $SQRT_MODELDIM \
			--qk-norm RMSNorm \
			--wd 0 --decay linear --no-warmup \
			--untie-embed \
			--lr $BASE_LR --matrix-lr $matrix_lr --embedding-lr $EMB_LR \
			--extra-name exp6.1-hs-${hsmode}-nogains \
			$*
	done
done
