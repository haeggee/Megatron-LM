source .env  # export WANDB_API_KEY and HF_TOKEN here.
# Exp 7a: WSD cooldown-shape comparison on 110M (master-muon + gains).
# Branch from the exp7 Mlr=0.008 WSD run at its 40k checkpoint and finish each
# variant to 50k (the same 40k->50k window the original cooled over), varying only
# the cooldown shape. All variants share the exact pretraining optimizer state, so
# differences are purely due to the anneal shape.
#
# --branch-from seeds each (fresh) run dir with a symlink to the source iter_0040000
# and adds --override-opt_param-scheduler so the new --wsd shape takes effect while
# resuming at step 40000. minus_sqrt reproduces the original cooldown (a control).

INV_LAYERS=0.083
SQRT_MODELDIM=22.62
NODES=2
MODEL_SIZE=110

BASE_LR=0.001
EMB_LR=0.003
MATRIX_LR=0.008  # = 0.001 * 2**(6/2), i.e. k=6 in exp7-lr-decay.sh

# Source run to branch from (its 40k checkpoint).
PRETRAIN=$SCRATCH/opts/logs/vf/110M1-mas_o_shup_nest-wd0-HSflat1_l2_embrowNO_growcol_gpsp-Mlr0.008-Elr0.003-fls0.083-up22.62-nw-exp7-decay-wsd

for shape in minus_cbrt minus_cbcrt sqrt_pow2 power2 power3; do
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
		--wd 0 --decay wsd --cooldown 0.2 --no-warmup \
		--untie-embed \
		--lr $BASE_LR --matrix-lr $MATRIX_LR --embedding-lr $EMB_LR \
		--branch-from "$PRETRAIN" --branch-iter 40000 \
		--wsd $shape \
		--apertus-reservation \
		--extra-name exp7a \
		$*
done
