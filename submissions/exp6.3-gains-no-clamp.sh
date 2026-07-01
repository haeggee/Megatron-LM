source .env  # export WANDB_API_KEY and HF_TOKEN here.
# Exp 6.3: gains undo-clamp ablation on 110M (master-muon).
#
# Background: in _preprocess_gains the undo division p / phi(g) used to floor
# phi(g) at 1e-8, while _apply_gains re-multiplies by the *true* (unclamped,
# possibly negative or sub-eps) phi(g). With direct parametrization and no
# gains_min, a gain can run below the floor — even negative — and the undo then
# divides by a clamped positive value while the re-apply multiplies by the true
# value, so the round-trip stops inverting and p drifts (complete mismatch).
#
# --gains-no-clamp uses the true phi(g) on undo so it exactly inverts the
# re-apply. This sweep compares the no-clamp fix against the previous defaults
# (plain clamp, and the gmin=1e-5 projected-gradient floor from exp6.2), all
# with the direct parametrization where the mismatch actually bites.

INV_LAYERS=0.083
SQRT_MODELDIM=22.62
NODES=2
MODEL_SIZE=110

BASE_LR=0.001
EMB_LR=0.003

# ---- no-clamp undo vs. clamped baselines, direct parametrization ----
for k in 3 4 5 6 7 8; do
	matrix_lr=$(python3 -c "print(0.001 * 2**($k/2))")
	bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
		--eval-every 1000 --eval-iters 50 \
		--opt master --master-orthogonalize --alpha 0 \
		--hs flat --hs-embed row --hs-embed-no-orthogonal \
		--b1 0.95 --mb1 0.9 --muon-scale shape_up --muon-nesterov \
		--hs-g rowcol --hs-g-embed none \
		--hs-g-param direct \
		--post-norm \
		--fixed-layer-scale $INV_LAYERS \
		--upscale-embedding $SQRT_MODELDIM \
		--qk-norm RMSNorm \
		--wd 0 --decay linear --no-warmup \
		--untie-embed \
		--lr $BASE_LR --matrix-lr $matrix_lr --embedding-lr $EMB_LR \
		--extra-name exp6.3-hs-direct-noclamp \
		--gains-no-clamp \
		$*
done
