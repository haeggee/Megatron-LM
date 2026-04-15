#!/bin/bash
# Minimal debug script for testing model internals logging
# Usage: 
#   1. Get interactive job:
# srun --account=infra01 --time=00:30:00 --nodes=1 --ntasks-per-node=1 \
#      --gpus-per-node=4 --cpus-per-task=72 --mem=460000 --partition=debug \
#      --environment=/capstor/store/cscs/swissai/a06/containers/NGC-PyTorch/ngc_pt_jan.toml \
#      --pty bash
#   2. Run: bash scripts/debug.sh

set -e
ulimit -c 0

MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/Megatron-LM-personal
cd $MEGATRON_LM_DIR

# Trigger directory for exit/save signals
TRIGGER_DIR=/tmp/megatron-debug-triggers-$$
mkdir -p $TRIGGER_DIR

# Dataset (using a single shard for simplicity)
DATASET="/capstor/store/cscs/swissai/a06/datasets_tokenized/megatron/sai/swissai-fineweb-edu-score-2-filterrobots-merge/dump-0-merged"
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/$USER/datasets/cache

# Environment setup
export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR=localhost
export MASTER_PORT=8963

# Number of GPUs (data parallel size)
NGPUS=4

# Run weight heatmap
# torchrun --nproc_per_node=1 tools/checkpoint/weight_heatmap.py /iopsstor/scratch/cscs/ahernnde/opts/logs/v1/110M/checkpoints/iter_0050000

python tools/checkpoint/heatmap.py \
    --checkpoint-path /iopsstor/scratch/cscs/ahernnde/opts/logs/v1/110M/checkpoints/iter_0050000 \
    --layers 0,2,4,6,8,11 \
    --subsample -1 \
    --output adamw_baseline.png

python tools/checkpoint/heatmap.py \
    --checkpoint-path /iopsstor/scratch/cscs/ahernnde/opts/logs/v1/110M-master_a0-wd0-HSrow1_l2_emb_sh-lr0.004-std0.044-ngpt-nw-cos//checkpoints/iter_0050000 \
    --layers 0,2,4,6,8,11 \
    --subsample -1 \
    --output ngpt.png


python tools/checkpoint/heatmap.py \
    --checkpoint-path /iopsstor/scratch/cscs/ahgele/opts/logs/v1/110M-master_h_a0-wd0-HSembed1_l2_it0_emb_sh-std0.044-L2Norm-fz-nPre-nFin-pst-ppst-usmr-ss11.31-ls0.083S0.044-qkls1S0.044-mlpls1G22.62-lgsls1S0.044-nw-cos/checkpoints/iter_0050000 \
    --layers 0,2,4,6,8,11 \
    --subsample -1 \
    --output ngpt_fix.png


python tools/checkpoint/heatmap.py \
    --checkpoint-path /iopsstor/scratch/cscs/ahgele/opts/logs/v1/110M-master_h_a0-wd0-HSembed1_l2_it0_emb_sh-qkRMS-nPre-nFin-pst-ls0.083S0.044-lgsls1S0.044-up22.62-nw-cos/checkpoints/iter_0050000 \
    --layers 0,2,4,6,8,11 \
    --subsample -1 \
    --output nfog.png