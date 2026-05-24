source .env  # export WANDB_API_KEY and HF_TOKEN here.
# Exp 5: decoupled 2D grid over (embedding LR, matrix LR) for
# master-muon + gains on 110M.
#
# All three LRs are absolute now (no multiplicative coupling):
#   --lr            base LR (norms, biases, 1D params)
#   --matrix-lr     matrix (2D linear) LR via master
#   --embedding-lr  embedding LR (works whether or not --hs-embed is set,
#                   after the chained_adam fix in master.py)
# We hold base LR fixed at the master-typical 1e-3 and sweep the two
# parameter classes whose LRs actually matter for this method, in sqrt(2)
# steps -- same spacing as the matrix-LR sweep used everywhere else.
#
# Both axes: 0.001 * 2^(k/2), 6 values.
# matrix_lr k in 3..8 -> 0.00283 .. 0.016 (matches exp 1 sweep).
# emb_lr    k in 1..6 -> 0.00141 .. 0.008  (brackets the 3e-3 used in exp 1).

INV_LAYERS=0.083
SQRT_MODELDIM=22.62
NODES=2
MODEL_SIZE=110

BASE_LR=0.001

for ke in 2 3 4 5; do
	emb_lr=$(python3 -c "print(0.001 * 2**($ke/2))")
	for km in 4 5 6 7 8; do
		matrix_lr=$(python3 -c "print(0.001 * 2**($km/2))")
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
			--lr $BASE_LR \
			--matrix-lr $matrix_lr \
			--embedding-lr $emb_lr \
			--extra-name exp5-grid \
			$*
	done
done
