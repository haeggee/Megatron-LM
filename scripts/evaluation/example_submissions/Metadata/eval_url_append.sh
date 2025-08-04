# Make sure to set your WANDB_API_KEY.
export LOGS_ROOT=$SCRATCH/eval-logs/main-v1
# export MODEL=/iopsstor/scratch/cscs/dfan/Megatron-LM/logs/Meg-Runs/robotstxt_ablation_runs/apertus-1b-21n-4096sl-504gbsz-fw-edu-robots-filtered/checkpoints/ 
# export MODEL=/iopsstor/scratch/cscs/dfan/Megatron-LM/logs/Meg-Runs/robotstxt_ablation_runs/apertus-1b-21n-4096sl-504gbsz-fw-edu-robots-filtered-plus-Top1-domains/checkpoints/
# export MODEL=/iopsstor/scratch/cscs/dfan/Megatron-LM/logs/Meg-Runs/robotstxt_ablation_runs/apertus-1b-21n-4096sl-504gbsz-fw-edu-robots-filtered-plus-Top5-domains/checkpoints/
# export MODEL=/iopsstor/scratch/cscs/dfan/Megatron-LM/logs/Meg-Runs/robotstxt_ablation_runs/apertus-1b-21n-4096sl-504gbsz-fw-edu-robots-filtered-plus-Top10-domains/checkpoints/
export MODEL=/iopsstor/scratch/cscs/dfan/Megatron-LM/logs/Meg-Runs/meta_data_conditioning_masking_out/apertus-1b-21n-4096sl-504gbsz-fw-edu-url-0.9_non_url-0.1-APPEND/checkpoints/
export NAME=URL-append-5shots-$(date '+%Y-%m-%d_%H-%M-%S')
export ARGS="--convert-to-hf --size 1 --wandb-entity meta-robots --wandb-project meta_eval_nips_Apr26 --wandb-id $NAME --bs 32 --tokens-per-iter 2064384 --tasks scripts/evaluation/english_eval"

bash scripts/evaluation/submit_evaluation.sh $MODEL $ARGS --iterations "10000,20000,30000,40000,48441"
