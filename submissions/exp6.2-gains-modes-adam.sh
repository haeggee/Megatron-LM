source .env  # export WANDB_API_KEY and HF_TOKEN here.
# Exp 6.2: gains modes comparison on 110M (master-muon).
# Vary the gains mode (--hs-g) across {flat, embed, row}, with --hs fixed to flat.

INV_LAYERS=0.083
SQRT_MODELDIM=22.62
NODES=2
MODEL_SIZE=110

BASE_LR=0.001
EMB_LR=0.003


# with different eps
# bash submissions/submit.sh 110 --nodes 2 \
#     --eval-every 1000 --eval-iters 50 \
#     --opt master --master-orthogonalize --alpha 0 \
#     --hs flat --hs-embed row --hs-embed-no-orthogonal \
#     --b1 0.95 --mb1 0.9 --muon-scale shape_up --muon-nesterov \
#     --hs-g rowcol --hs-g-embed none \
#     --post-norm \
#     --fixed-layer-scale 0.083 --upscale-embedding 22.62 \
#     --qk-norm RMSNorm \
#     --wd 0 --decay linear --no-warmup \
#     --untie-embed \
#     --lr 0.001 --matrix-lr 8e-3 --embedding-lr 0.003 \
#     --extra-name exp1-master-gains --geps 1e-12


# bash submissions/submit.sh 110 --nodes 2 \
#     --eval-every 1000 --eval-iters 50 \
#     --opt master --master-orthogonalize --alpha 0 \
#     --hs flat --hs-embed row --hs-embed-no-orthogonal \
#     --b1 0.95 --mb1 0.9 --muon-scale shape_up --muon-nesterov \
#     --hs-g rowcol --hs-g-embed none \
#     --post-norm \
#     --fixed-layer-scale 0.083 --upscale-embedding 22.62 \
#     --qk-norm RMSNorm \
#     --wd 0 --decay linear --no-warmup \
#     --untie-embed \
#     --lr 0.001 --matrix-lr 8e-3 --embedding-lr 0.003 \
#     --extra-name exp1-master-gains --gb2 0.95


# remove bias correction for gains
# bash submissions/submit.sh 110 --nodes 2 \
#     --eval-every 1000 --eval-iters 50 \
#     --opt master --master-orthogonalize --alpha 0 \
#     --hs flat --hs-embed row --hs-embed-no-orthogonal \
#     --b1 0.95 --mb1 0.9 --muon-scale shape_up --muon-nesterov \
#     --hs-g rowcol --hs-g-embed none \
#     --post-norm \
#     --fixed-layer-scale 0.083 --upscale-embedding 22.62 \
#     --qk-norm RMSNorm \
#     --wd 0 --decay linear --warmup-iters 100 \
#     --untie-embed \
#     --lr 0.001 --matrix-lr 8e-3 --embedding-lr 0.003 \
#     --extra-name exp1-master-gains --g-no-bc


# remove bias correction for gains
bash submissions/submit.sh 110 --nodes 2 \
    --eval-every 1000 --eval-iters 50 \
    --opt master --master-orthogonalize --alpha 0 \
    --hs flat --hs-embed row --hs-embed-no-orthogonal \
    --b1 0.95 --mb1 0.9 --muon-scale shape_up --muon-nesterov \
    --hs-g rowcol --hs-g-embed none \
    --post-norm \
    --fixed-layer-scale 0.083 --upscale-embedding 22.62 \
    --qk-norm RMSNorm \
    --wd 0 --decay linear --no-warmup \
    --untie-embed \
    --lr 0.001 --matrix-lr 8e-3 --embedding-lr 0.003 \
    --extra-name exp1-master-gains --gmin 1e-5