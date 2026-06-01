source .env  # export WANDB_API_KEY and HF_TOKEN here.
# Exp 12: long-horizon overtraining on 110M and 232M.
# Three arms from exp1 (adam, muon, master-muon + gains), trained well past the 50k
# default: to 75k and 100k steps (both do 2000 iters/BT, so 37.5BT and 50BT).
# Each arm keeps its exp1 conventions: adam/muon use wd=0.1 + warmup=1000, the
# master+gains arm uses wd=0 + --no-warmup. Decay is linear to each full horizon.
#
# LR per length follows a length-scaling rule: multiplier = k^{-p}, where
# k = length/50k is the length multiplier and p the exponent. For 110M we compare
# the two common rules (p=0.3, p=0.5) plus an unscaled control (p=0, keep the 50k
# LR); for 232M we run p=0.5 only for now. Matrix centers are each arm's best-at-50k.
#
# SWEEP_ALL_GROUPS=true:  the whole LR vector (base, emb, matrix) follows the rule,
#                         keeping base:emb:matrix ratios fixed.
# SWEEP_ALL_GROUPS=false: only matrix LR follows the rule; base/emb stay at centers.
SWEEP_ALL_GROUPS=false

# 50k-step reference (length multiplier k = length/50k). base/emb LRs are shared.
REF_STEPS=50000
CENTER_BASE=0.001
CENTER_EMB=0.003

# steps -> tokens (in B). 2000 iters/BT: 75k=37.5BT, 100k=50BT.
declare -A STEP_TOKENS=( [75000]=37.5 [100000]=50 )

# Per-model: nodes, arch (fixed-layer-scale=1/L, upscale-embedding=sqrt(d)), the
# per-arm matrix-LR centers (best-at-50k), and which scaling exponents to run.
for MODEL_SIZE in 232; do
	if [[ $MODEL_SIZE -eq 110 ]]; then  # 12L, hidden 512
		NODES=2
		INV_LAYERS=0.083; SQRT_MODELDIM=22.62
		CENTER_ADAM=0.008; CENTER_MUON=0.005657; CENTER_MASTER=0.01131
		EXPONENTS=(0 0.3 0.5)
	elif [[ $MODEL_SIZE -eq 232 ]]; then  # 18L, hidden 768
		NODES=4
		INV_LAYERS=0.056; SQRT_MODELDIM=27.71
		CENTER_ADAM=0.005657; CENTER_MUON=0.005657; CENTER_MASTER=0.008
		EXPONENTS=(0.5)
	fi

for STEPS in 75000 ; do
	TOKENS=${STEP_TOKENS[$STEPS]}
	K=$(python3 -c "print($STEPS/$REF_STEPS)")

	# Apply each length-scaling rule: LR multiplier = k^{-p}.
	for p in "${EXPONENTS[@]}"; do
		FACTOR=$(python3 -c "print($K**(-$p))")
		TAG=p$p

		if [[ $SWEEP_ALL_GROUPS = true ]]; then
			BASE_LR=$(python3 -c "print($CENTER_BASE*$FACTOR)")
			EMB_LR=$(python3 -c "print($CENTER_EMB*$FACTOR)")
		else
			BASE_LR=$CENTER_BASE
			EMB_LR=$CENTER_EMB
		fi
		matrix_lr_adam=$(python3 -c "print($CENTER_ADAM*$FACTOR)")
		matrix_lr_muon=$(python3 -c "print($CENTER_MUON*$FACTOR)")
		matrix_lr_master=$(python3 -c "print($CENTER_MASTER*$FACTOR)")

		# ---- adam (master, no orthogonalize, no hypersphere) ----
		bash submissions/submit.sh $MODEL_SIZE --nodes $NODES --tokens $TOKENS \
			--eval-every 1000 --eval-iters 50 \
			--opt master --alpha 0 \
			--post-norm \
			--fixed-layer-scale $INV_LAYERS \
			--upscale-embedding $SQRT_MODELDIM \
			--qk-norm RMSNorm \
			--wd 0.1 --decay linear \
			--warmup-iters 1000 \
			--untie-embed \
			--lr $BASE_LR --matrix-lr $matrix_lr_adam --embedding-lr $EMB_LR \
			--extra-name exp12-$TAG \
			$*

		# ---- muon (master + orthogonalize, no hypersphere) ----
		bash submissions/submit.sh $MODEL_SIZE --nodes $NODES --tokens $TOKENS \
			--eval-every 1000 --eval-iters 50 \
			--opt master --master-orthogonalize --alpha 0 \
			--b1 0.95 --mb1 0.9 --muon-scale unit_rms_norm --muon-nesterov \
			--post-norm \
			--fixed-layer-scale $INV_LAYERS \
			--upscale-embedding $SQRT_MODELDIM \
			--qk-norm RMSNorm \
			--wd 0.1 --decay linear \
			--warmup-iters 1000 \
			--untie-embed \
			--lr $BASE_LR --matrix-lr $matrix_lr_muon --embedding-lr $EMB_LR \
			--extra-name exp12-$TAG \
			$*

		# ---- master-muon + gains (our method) ----
		bash submissions/submit.sh $MODEL_SIZE --nodes $NODES --tokens $TOKENS \
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
			--lr $BASE_LR --matrix-lr $matrix_lr_master --embedding-lr $EMB_LR \
			--extra-name exp12-$TAG \
			$*
	done
done
done
