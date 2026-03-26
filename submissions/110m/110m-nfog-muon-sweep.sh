
# cuz I think we tried this at some point and it didn't improve things but good to try
# bash submissions/110m/110m-nfog-muon.sh --muon-scale none

# bash submissions/110m/110m-nfog-muon.sh --muon-scale none --hs-embed-no-orthogonal

# I think this is the _correct_ hypermuon setting that we would want to try with update normalization no? i.e. without ademamix (lol), adam embeddings, but still normalization of lm head weights
# bash submissions/110m/110m-nfog-muon.sh --hs-embed-no-orthogonal
# bash submissions/110m/110m-nfog-muon.sh --hs-embed-no-orthogonal --lr 0.001
# bash submissions/110m/110m-nfog-muon.sh --hs-embed-no-orthogonal --lr 0.002
# bash submissions/110m/110m-nfog-muon.sh --hs-embed-no-orthogonal --lr 0.008

# bash submissions/110m/110m-nfog-muon.sh --hs-u --hs-embed-no-orthogonal
# bash submissions/110m/110m-nfog-muon.sh --hs-u --hs-embed-no-orthogonal --lr 0.001
# bash submissions/110m/110m-nfog-muon.sh --hs-u --hs-embed-no-orthogonal --lr 0.002
# bash submissions/110m/110m-nfog-muon.sh --hs-u --hs-embed-no-orthogonal --lr 0.008

# # new
# bash submissions/110m/110m-nfog-muon.sh --hs-embed-no-orthogonal --lr 0.012
# bash submissions/110m/110m-nfog-muon.sh --hs-u --hs-embed-no-orthogonal --lr 0.012


# bash submissions/110m/110m-nfog-muon.sh --hs-embed-no-orthogonal --lr 0.016
# bash submissions/110m/110m-nfog-muon.sh --hs-u --hs-embed-no-orthogonal --lr 0.016


bash submissions/110m/110m-nfog-muon-embNO.sh --lr 0.002 --mlr 2
bash submissions/110m/110m-nfog-muon-embNO.sh --lr 0.004 --mlr 2
bash submissions/110m/110m-nfog-muon-embNO.sh --lr 0.002 --mlr 4
bash submissions/110m/110m-nfog-muon-embNO.sh --lr 0.004 --mlr 4
# bash submissions/110m/110m-nfog-muon-embNO.sh --lr 0.004 --mlr 4 --logits-layer-scale-scale 1

bash submissions/110m/110m-nfog-muon-no-emb.sh --lr 0.002 --mlr 2
bash submissions/110m/110m-nfog-muon-no-emb.sh --lr 0.004 --mlr 2
bash submissions/110m/110m-nfog-muon-no-emb.sh --lr 0.002 --mlr 4
bash submissions/110m/110m-nfog-muon-no-emb.sh --lr 0.004 --mlr 4


bash submissions/110m/110m-nfog-muon-embNO.sh --lr 0.002 --mlr 2  --hs-u  
bash submissions/110m/110m-nfog-muon-embNO.sh --lr 0.004 --mlr 2  --hs-u  
bash submissions/110m/110m-nfog-muon-embNO.sh --lr 0.002 --mlr 4  --hs-u  
bash submissions/110m/110m-nfog-muon-embNO.sh --lr 0.004 --mlr 4  --hs-u  


bash submissions/110m/110m-nfog-muon-no-emb.sh --lr 0.002 --mlr 2 --hs-u
bash submissions/110m/110m-nfog-muon-no-emb.sh --lr 0.004 --mlr 2 --hs-u
bash submissions/110m/110m-nfog-muon-no-emb.sh --lr 0.002 --mlr 4 --hs-u
bash submissions/110m/110m-nfog-muon-no-emb.sh --lr 0.004 --mlr 4 --hs-u


bash submissions/110m/110m-nfog-muon-embNO.sh --lr 0.004 --mlr 2  --muon-scale none
bash submissions/110m/110m-nfog-muon-embNO.sh --lr 0.002 --mlr 4  --muon-scale none
bash submissions/110m/110m-nfog-muon-no-emb.sh --lr 0.004 --mlr 2  --muon-scale none
bash submissions/110m/110m-nfog-muon-no-emb.sh --lr 0.002 --mlr 4  --muon-scale none
bash submissions/110m/110m-nfog-muon-no-emb.sh --lr 0.002 --mlr 8  --muon-scale none
bash submissions/110m/110m-nfog-muon-no-emb.sh --lr 0.004 --mlr 4  --muon-scale none


bash submissions/110m/110m-nfog-muon-no-emb.sh --lr 0.002 --mlr 4 --hs-u --decay wsd
bash submissions/110m/110m-nfog-muon-no-emb.sh --lr 0.001 --mlr 4 --hs-u --decay wsd

bash submissions/110m/110m-nfog-muon-no-emb.sh --lr 0.002 --mlr 8 
bash submissions/110m/110m-nfog-muon-no-emb.sh --lr 0.002
bash submissions/110m/110m-nfog-muon-embNO.sh --lr 0.002 --mlr 8 
bash submissions/110m/110m-nfog-muon-embNO.sh --lr 0.002

bash submissions/110m/110m-nfog-muon-embNO.sh --muon-scale shape_scaling --lr 0.002 --mlr 4 
bash submissions/110m/110m-nfog-muon-embNO.sh --muon-scale shape_scaling --lr 0.002 --mlr 8 
bash submissions/110m/110m-nfog-muon-embNO.sh --muon-scale shape_scaling --lr 0.004 --mlr 2 
bash submissions/110m/110m-nfog-muon-embNO.sh --muon-scale shape_scaling --lr 0.004 --mlr 4  
bash submissions/110m/110m-nfog-muon-no-emb.sh --muon-scale shape_scaling --lr 0.002 --mlr 4 
bash submissions/110m/110m-nfog-muon-no-emb.sh --muon-scale shape_scaling --lr 0.002 --mlr 8
bash submissions/110m/110m-nfog-muon-no-emb.sh --muon-scale shape_scaling --lr 0.004 --mlr 2 
bash submissions/110m/110m-nfog-muon-no-emb.sh --muon-scale shape_scaling --lr 0.004 --mlr 4 


bash submissions/110m/110m-nfog-muon-no-emb.sh --lr 0.004 --mlr 8  --muon-scale none


# # Then with best base LR sweep muon LR?

# torchrun --nproc_per_node=4 --nnodes=256 --node_rank=68 --master_addr=nid006029 --master_port=29591 pretrain_gpt.py --num-layers 45 --hidden-size 2048 --ffn-hidden-size 6144 --num-attention-heads 32 --group-query-attention --num-query-groups 4 --kv-channels 128 --seq-length 4096 --max-position-embeddings 4096 --make-vocab-size-divisible-by 128 --normalization RMSNorm --norm-epsilon 1e-6 --squared-relu --qk-layernorm --position-embedding-type rope --rotary-base 10000 --no-rope-fusion --disable-bias-linear --untie-embeddings-and-output-weights --hidden-dropout 0.0 --attention-dropout 0.0 --no-bias-dropout-fusion --num-experts 128 --moe-ffn-hidden-size 768 --moe-router-topk 7 --moe-shared-expert-intermediate-size 768 --moe-layer-freq '([0]+[1]*44)' --moe-router-score-function sigmoid --moe-router-load-balancing-type seq_aux_loss --moe-aux-loss-coeff 0.0001 --moe-router-enable-expert-bias --moe-router-topk-scaling-factor 2.5 --moe-router-dtype fp32 --moe-grouped-gemm --moe-token-dispatcher-type alltoall --moe-per-layer-logging --overlap-moe-expert-parallel-comm --tensor-model-parallel-size 1 --pipeline-model-parallel-size 4 --expert-model-parallel-size 4 --pipeline-model-parallel-layout 'Et|(tt|)*22L' --window-size '(4096, 0)' --window-attn-skip-freq 4 --attention-backend flash --transformer-impl transformer_engine --ckpt-format torch_dist --cross-entropy-loss-fusion --cross-entropy-fusion-impl te --bf16 --no-check-for-nan-in-loss-and-grad --disable-symmetric-registration --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather --recompute-activations --global-batch-size 4096 --micro-batch-size 2 --train-iters 298023 --eval-interval 999999999 --eval-iters 0 --lr 1e-3 --min-lr 1e-5 --adam-beta1 0.9 --adam-beta2 0.95 --adam-eps 1e-8 --weight-decay 0.1 --lr-warmup-iters 2000 --lr-decay-style WSD --clip-grad 1.0 --init-method-std 0.008 --no-load-optim --no-load-rng --save-interval 200 --save /capstor/scratch/cscs/ntazi/checkpoints/256n_gbs4096_localattn --log-interval 1 --log-throughput --log-timers-to-tensorboard --log-validation-ppl-to-tensorboard --tensorboard-queue-size 1 --tensorboard-dir /capstor/scratch/cscs/ntazi/checkpoints/256n_gbs4096_localattn/tensorboard --wandb-project qwen3-moe --wandb-exp-name 256n_gbs4096_localattn --tokenizer-type HuggingFaceTokenizer --tokenizer-model alehc/swissai-tokenizer --data-cache-path /iopsstor/scratch/cscs/ntazi/datasets/cache --dataloader-type single --num-workers 2 --num-dataset-builder-threads 8 --distributed-timeout-minutes 600 --per-split-data-args-path /iopsstor/scratch/cscs/ntazi/projects/Megatron-Bridge/prod/data_configs/bmessmer_multilingual_moe_stage1.json --seed 41