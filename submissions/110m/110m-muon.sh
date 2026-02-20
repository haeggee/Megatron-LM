source .env  # export WANDB_API_KEY and HF_TOKEN here.
bash submissions/submit.sh 110 --opt muon --b1 0.95 $*
