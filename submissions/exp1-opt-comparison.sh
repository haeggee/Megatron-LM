source .env  # export WANDB_API_KEY and HF_TOKEN here.
# Exp 1: optimizer comparison on 110M.
# adam vs muon vs master-muon (no gains) vs master-muon + gains.
# All four go through the master-opt code path; matrix LR is swept absolutely.
# Conventional per-opt training: adam/muon use wd=0.1 + warmup=1000,
# master variants use wd=0 + --no-warmup. Embedding LR is matched across
# arms (now that the chained_adam path respects --embedding-lr).

INV_LAYERS=0.083
SQRT_MODELDIM=22.62
NODES=2
MODEL_SIZE=110
INV_SQRT_MODELDIM=0.044

BASE_LR=0.001
EMB_LR=0.003

# matrix LR sweep: 2^(k/2) * 1e-3 for k in 3..8 (same effective range as before).

---- adam (master, no orthogonalize, no hypersphere) ----
for k in 3 4 5 6 7 8; do
	matrix_lr=$(python3 -c "print(0.001 * 2**($k/2))")
	bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
		--eval-every 1000 --eval-iters 50 \
		--opt master --alpha 0 \
		--post-norm \
		--fixed-layer-scale $INV_LAYERS \
		--upscale-embedding $SQRT_MODELDIM \
		--qk-norm RMSNorm \
		--wd 0.1 --decay linear \
		--warmup-iters 1000 \
		--untie-embed \
		--lr $BASE_LR --matrix-lr $matrix_lr --embedding-lr $EMB_LR \
		--extra-name exp1.1-adam \
		$*
done

# ---- muon (master + orthogonalize, no hypersphere) ----
for k in 3 4 5 6 7 8; do
	matrix_lr=$(python3 -c "print(0.001 * 2**($k/2))")
	bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
		--eval-every 1000 --eval-iters 50 \
		--opt master --master-orthogonalize --alpha 0 \
		--b1 0.95 --mb1 0.9 --muon-scale shape_up --muon-nesterov \
		--post-norm \
		--fixed-layer-scale $INV_LAYERS \
		--upscale-embedding $SQRT_MODELDIM \
		--qk-norm RMSNorm \
		--wd 0.1 --decay linear \
		--warmup-iters 1000 \
		--untie-embed \
		--lr $BASE_LR --matrix-lr $matrix_lr --embedding-lr $EMB_LR \
		--extra-name exp1.1-muon \
		$*
done

# # ---- master-muon, no gains ----
for k in 3 4 5 6 7 8; do
	matrix_lr=$(python3 -c "print(0.001 * 2**($k/2))")
	bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
		--eval-every 1000 --eval-iters 50 \
		--opt master --master-orthogonalize --alpha 0 \
		--hs flat --hs-embed row --hs-embed-no-orthogonal \
		--b1 0.95 --mb1 0.9 --muon-scale shape_up --muon-nesterov \
		--post-norm \
		--fixed-layer-scale $INV_LAYERS \
		--upscale-embedding $SQRT_MODELDIM \
		--qk-norm RMSNorm \
		--wd 0 --decay linear --no-warmup \
		--untie-embed \
		--lr $BASE_LR --matrix-lr $matrix_lr --embedding-lr $EMB_LR \
		--extra-name exp1-master-nogains \
		$*
done

# # ---- master-muon + gains (our method) ----
for k in 3 4 5 6 7 8; do
	matrix_lr=$(python3 -c "print(0.001 * 2**($k/2))")
	bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
		--eval-every 1000 --eval-iters 50 \
		--opt master --master-orthogonalize --alpha 0 \
		--hs flat --hs-embed row --hs-embed-no-orthogonal \
		--b1 0.95 --mb1 0.9 --muon-scale shape_up --muon-nesterov \
		--hs-g rowcol --hs-g-embed none \
		--post-norm \
		--hs-g-param softplus \
		--fixed-layer-scale $INV_LAYERS \
		--upscale-embedding $SQRT_MODELDIM \
		--qk-norm RMSNorm \
		--wd 0 --decay linear --no-warmup \
		--untie-embed \
		--lr $BASE_LR --matrix-lr $matrix_lr --embedding-lr $EMB_LR \
		--extra-name exp1-master-gains \
		$*
done

# ---- master + adam + hypersphere (no orthogonalize, no gains); warmup kept ----
for k in 3 4 5 6 7 8; do
	matrix_lr=$(python3 -c "print(0.001 * 2**($k/2))")
	bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
		--eval-every 1000 --eval-iters 50 \
		--opt master --alpha 0 \
		--hs flat --hs-embed row --hs-embed-no-orthogonal \
		--post-norm \
		--fixed-layer-scale $INV_LAYERS \
		--upscale-embedding $SQRT_MODELDIM \
		--qk-norm RMSNorm \
		--wd 0 --decay linear \
		--warmup-iters 1000 \
		--untie-embed \
		--lr $BASE_LR --matrix-lr $matrix_lr --embedding-lr $EMB_LR \
		--extra-name exp1-master-adam-nogains \
		$*
done

# ---- master + adam + hypersphere + gains (no orthogonalize); warmup kept ----
for k in 3 4 5 6 7 8; do
	matrix_lr=$(python3 -c "print(0.001 * 2**($k/2))")
	bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
		--eval-every 1000 --eval-iters 50 \
		--opt master --alpha 0 \
		--hs flat --hs-embed row --hs-embed-no-orthogonal \
		--hs-g rowcol --hs-g-embed none \
		--post-norm \
		--hs-g-param softplus \
		--fixed-layer-scale $INV_LAYERS \
		--upscale-embedding $SQRT_MODELDIM \
		--qk-norm RMSNorm \
		--wd 0 --decay linear \
		--warmup-iters 1000 \
		--untie-embed \
		--lr $BASE_LR --matrix-lr $matrix_lr --embedding-lr $EMB_LR \
		--extra-name exp1-master-adam-gains \
		$*
done
