MEGATRON_LM_DIR=/iopsstor/scratch/cscs/dfan/copies/Megatron-LM-convert
TORCH_NODIST_PATH=$(mktemp -d -p $SCRATCH/.tmp)
# CHECKPOINT_PATH=/iopsstor/scratch/cscs/dfan/Megatron-LM/logs/Meg-Runs/robotstxt_ablation_runs/apertus-1b-21n-4096sl-504gbsz-fw-edu-robots-filtered/checkpoints/ 
# CHECKPOINT_PATH=/iopsstor/scratch/cscs/dfan/Megatron-LM/logs/Meg-Runs/robotstxt_ablation_runs/apertus-1b-21n-4096sl-504gbsz-fw-edu-robots-kept/checkpoints/
CHECKPOINT_PATH=/iopsstor/scratch/cscs/dfan/copies/Megatron-LM-debug/logs/Meg-Runs/meta_data_conditioning_masking_out/apertus-1b-21n-4096sl-504gbsz-fw-edu-url_suffix_v2/checkpoints
ITERATIONS=$(cat $CHECKPOINT_PATH/latest_checkpointed_iteration.txt)
MODEL_NAME=url_suffix_correct
HF_SAVE_DIR=/iopsstor/scratch/cscs/$USER/Meg-Checkpoints/hf-checkpoints
HF_CKPT_PATH=$HF_SAVE_DIR/$MODEL_NAME
TOKENIZER=dyfan/swissai-tokenizer-wcontext
REPOS_PATH=$(mktemp -d -p $SCRATCH/.tmp)

export CUDA_DEVICE_MAX_CONNECTIONS=1

export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH

REPOS_PATH=$(mktemp -d -p $SCRATCH/.tmp)
cd $REPOS_PATH
git clone https://github.com/swiss-ai/transformers.git
cd transformers
git checkout swissai-model
python -m pip install -e .

cd $MEGATRON_LM_DIR
torchrun scripts/conversion/torchdist_2_torch.py --bf16 --load=$CHECKPOINT_PATH --ckpt-step=$ITERATIONS --ckpt-convert-save=$TORCH_NODIST_PATH

python tools/checkpoint/convert.py --model-type=GPT --loader=core --saver=llama_hf --load-dir=$TORCH_NODIST_PATH/torch --save-dir=$HF_CKPT_PATH --hf-tokenizer=$TOKENIZER
