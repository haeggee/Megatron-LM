source .env  # export WANDB_API_KEY and HF_TOKEN here.
INV_SQRT_L=0.28  # slightly less to avoid blow ups at initialization.
INV_LAYERS=0.083  # approx 1/n_layers
NODES=2

# model_size inv_sqrt_modeldim sqrt_modeldim
configs=(
    "47  0.0625 16.00"   # model dim: 256
    "110 0.044  22.62"   # model dim: 512
    "193 0.036  27.71"   # model dim: 768
    "292 0.031  32.00"   # model dim: 1024
)

# flat, but with preln + final layernorm + warmup
for config in "${configs[@]}"; do
    read -r MODEL_SIZE INV_SQRTMODELDIM SQRT_MODELDIM <<< "$config"
    for lr in 0.002 0.003 0.004 0.005 0.006; do
        bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
            --opt master --alpha 0 --hs flat --hs-embed \
            --post-norm --post-norm-no-gain \
            --layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
            --upscale-embedding $SQRT_MODELDIM \
            --qk-norm RMSNorm \
            --logits-layer-scale 1 --logits-layer-scale-scale 1 \
            --wd 0 --decay linear \
            --lr $lr \
            $*
    done
done


for config in "${configs[@]}"; do
    read -r MODEL_SIZE INV_SQRTMODELDIM SQRT_MODELDIM <<< "$config"
    for mlr in 1 2 4 6 8; do
        # same with muon
        bash submissions/submit.sh $MODEL_SIZE --nodes $NODES \
            --opt master --master-orthogonalize --alpha 0 --hs flat --hs-embed --hs-embed-no-orthogonal \
            --b1 0.95 --mb1 0.9 --muon-scale shape_up \
            --post-norm --post-norm-no-gain \
            --layer-scale $INV_LAYERS --layer-scale-scale $INV_SQRTMODELDIM \
            --upscale-embedding $SQRT_MODELDIM \
            --qk-norm RMSNorm \
            --logits-layer-scale 1 --logits-layer-scale-scale 1 \
            --wd 0 --decay linear \
            --lr 0.003 --mlr $mlr \
            $*
    done
done
