#= Prelude =#
# General settings.
CONTAINER=/iopsstor/scratch/cscs/ahernnde/ncg_new_v3.toml

SCRIPT_VERSION=vf
SEQ_LEN=4096
TOKENIZER=mistralai/Mistral-Nemo-Base-2407
TOKENIZED_DATA_PATH=/iopsstor/scratch/cscs/jpcoles/a06/swissai-fineweb-edu-filterrobots-merge
DATA_PATHS=$(find $TOKENIZED_DATA_PATH -type f | sed -E 's/\.[^.]+$//' | sort -u | tr '\n' ' ')
VALID_DATA_PATH=$(find $TOKENIZED_DATA_PATH -type f | sed -E 's/\.[^.]+$//' | sort -u | head -1)
CODE_PATH=$PWD
EMERGING_OPTIMIZERS_PATH=/iopsstor/scratch/cscs/ahgele/opts/Emerging-Optimizers
TRAIN_ROOT=$SCRATCH/opts/logs

# tmps.
TRITON_HOME_DIR=/tmp/.triton
TORCH_INDUCTOR_CACHE_DIR=/tmp/.torch_inductor
PYTHON_CACHE_DIR=/tmp/.python_cache

# Hardware defaults.
NODES=1
TIME=12:00:00

# Arch defaults.
NORMALIZATION=RMSNorm

# Opt defaults.
WEIGHT_DECAY=0.1
WEIGHT_DECAY_METHOD=decoupled
MIN_LR=1e-8
OPT=adam

BETA1=0.9
BETA2=0.99
BETA3=0.999
MUON_MOMENTUM=0.95
ADAMBETA1=0.95
ALPHA=5
MUON_SCALE_MODE=none
MUON_EXTRA_SCALE_FACTOR=1.0
MUON_NUM_NS_STEPS=5
EMBEDDING_LR_MULTIPLIER=1.0

HYPERBALL=false
HS_KIND=l2
HS_R=1
HS_UPDATE=false
HS_EMBED=false
HS_SPLIT_HEADS=false
HS_SPLIT_HEADS_UPDATE=false

ACTIVATION=swiglu
CLIP_GRAD=1.0
NO_WARMUP=false
WARMUP_ITERS=5000
DECAY=wsd
COOLDOWN=0.2
WSD=minus_sqrt

# Validation defaults.
EVAL_INTERVAL=0
EVAL_ITERS=50

# Misc. defaults.
EXTRA_LOG=true
NO_SAVE=false

# Usage function.
usage () {
	echo "Usage: submit.sh <size> [options...]"
	echo "<size>: 110/390/1"
	echo "Options:"
	# Misc settings.
	echo " --nodes <nodes>: How many nodes to use"
	echo " --debug: Enable debug mode"
	echo " --extra-name <name>: Add a suffix to the name"
	echo " --time (default=$TIME): Change the sbatch time limit"
	echo " --no-extra-logs: Disable costly logging"
	echo " --no-save: Disable saving checkpoint"
	echo " --apertus-reservation: Add #SBATCH --reservation=SD-69241-apertus-1-5-0"
	# FP8 settings.
	echo " --fp8: Enables fp8"
	echo " --fp8dpa: Enables fp8dpa"
	# Training settings..
	echo " --tokens <int>: Amount of tokens to train with (in B)."
	echo " --lr <float>: Learning rate."
	echo " --no-warmup: Deactivates learning rate warmup"
	echo " --warmup-iters <int> (default=$WARMUP_ITERS): LR warmup steps (\`--lr-warmup-iters\`)"
	echo " --decay <wsd/cos/linear/inverse-square-root/isrwsd>"
	echo " --cooldown <float>: Fraction to do cooldown"
	echo " --branch-from <dir>: Seed run from another run's checkpoint (its root or checkpoints/ dir); for cooldown-shape experiments from a fixed ckpt"
	echo " --branch-iter <int>: Iteration to branch from (default: that run's latest checkpoint)"
	echo " --clip-grad <float>: Gradient clipping"
	# Architecture settings.
	echo " --init <float>: Change init std."
	echo " --scaled-output-init: Enable the 1/sqrt(2*L) scaling of the output layer init (off by default)"
	echo " --activation (default=$ACTIVATION): MLP activation. Choices=[swiglu, gelu]."
	echo " --no-pre-norm"
	echo " --no-final-layernorm"
	echo " --normalization <RMSNorm/L2Norm>"
	echo " --no-learnable-norms"
	echo " --post-norm"
	echo " --pre-norm-no-gain"
	echo " --post-norm-no-gain"
	echo " --final-layernorm-no-gain"
	echo " --post-block-norm"
	echo " --use-stream-minus-residual"
	echo " --layer-scale <float>"
	echo " --layer-scale-scale <float>"
	echo " --residual-layer-scale <float>"
	echo " --residual-layer-scale-scale <float>"
	echo " --softmax-scale <float>"
	echo " --qk-norm <RMSNorm/L2Norm>"
	echo " --mlp-layer-scale <float>"
	echo " --mlp-layer-scale-gate-scale <float>"
	echo " --mlp-out-scale <float>"
	echo " --fixed-layer-scale <float>"
	echo " --logits-layer-scale <float>"
	echo " --logits-layer-scale-scale <float>"
	echo " --upscale-embedding <float>"
	echo " --untie-embed: Untie embeddings and output weights"
	# Optimizer settings.
	echo " --opt <adam/dmuon/muon/dmaster/master/ademamix> (default=$OPT)"
	echo " --master-orthogonalize"
	echo " --poor-mans-ortho: Use _normalize instead of Newton-Schulz in the Muon branch"
	echo " --b1: beta1 (master&adam&muon&ademamix)"
	echo " --b2: beta2 (master&adam&ademamix)"
	echo " --mb1: beta1 (master&muon)"
	echo " --b3: beta3 (master&ademamix)"
	echo " --alpha: ademamix alpha"
	echo " --muon-scale <spectral/shape_scaling/unit_rms_norm/none>"
	echo " --muon-extra-scale-factor <float>: extra multiplicative factor on the muon scale (default=$MUON_EXTRA_SCALE_FACTOR)"
	echo " --muon-nesterov: Enables muon nesterov momentum"
	echo " --muon-num-ns-steps <int>: Number of Newton-Schulz steps for the Muon optimizer"
	echo " --mlr: muon learning rate factor"
	echo " --matrix-lr <float>: absolute matrix LR (overrides mlr * lr)"
	echo " --elr <float>: embedding/output LR multiplier (final LR = elr * lr)"
	echo " --embedding-lr <float>: absolute embedding LR (overrides --elr; applies whether or not --hs-embed is set)"
	echo " --output-lr <float>: absolute LM-head output LR (when untied)"
	echo " --scale-min-lr: scale min_lr proportionally to each group's max_lr"
	echo " --wd: weight decay"
	echo " --wd-method (decoupled/independent): weight decay method"
	echo " --wsd: wsd decay method"
	echo " --hs <row/col/rowcol/invrowcol/flat>: Enables hypersphere training"
	echo " --hs-kind <l2/standard/spectral>: hypersphere kind"
	echo " --hs-r <learnable/float>: hypersphere radius"
	echo " --hs-u: hypersphere normalize update"
	echo " --hs-embed [<row/col/rowcol/invrowcol/flat>]: hypersphere normalize embeddings (optionally with a different HS mode)"
	echo " --hs-emb-no-orthogonal: Don't use muon update on embeddings, but keep fixed to the sphere (if --hs-embed is also set, otherwise no effect)"
	echo " --hs-split-heads: hypersphere normalize q,k,v heads separately"
	echo " --hs-g <flat/embed/row/col/rowcol>: hypersphere gains mode"
	echo " --hs-g-output <flat/row/col/rowcol/none>: gains mode override for LM head output layer"
	echo " --hs-g-embed <flat/row/col/rowcol/none>: gains mode override for embedding input layer"
	echo " --glr <float>: absolute gains LR (default: follows param group LR)"
	echo " --hs-preserve-init: skip init projection; absorb init norms into gains"
	echo " --split-qkv-gains: separate column gains for Q, K, V"
	echo " --hs-g-param <direct/offset/softplus/exp>: gain parametrization (phi). direct=g, offset=1+g, softplus=softplus(g), exp=exp(g)."
	echo " --hs-p: project gradient to tangent space"
	echo " --hs-s: soft hyperball norm clipping."
	# Validation.
	echo " --eval-every <int>: Run validation every N steps (0 = disabled, default)"
	echo " --eval-iters <int> (default=$EVAL_ITERS): Number of batches per validation run"
	# Logs.
	echo " --wandb-name <str>: Specify wandb name."
}

if [[ $# -eq 0 ]]; then
	>&2 echo Invalid argument count: $#
	usage
	exit 1
fi

# Define some variables depending on size.
EXTRA_ARGS=()
SCALE=B  # M or B.
TP=1
PP=1
UNTIE=true
if [[ $1 -eq 110 ]]; then 
	# batch_size: ~0.52M.
	LAYERS=12
	HIDDEN_SIZE=512
	FFN_SIZE=2048
	NUM_HEADS=4
	NUM_QUERY_GROUPS=2
	MBS="${MBS:-8}"
	GBS=128
	ITERS_PER_BT=2000
	LR=0.001
	SIZE=110
	SAVE_FREQ=10000
	DEF_TOKENS=25
	INTERMEDIATE_METRICS_INTERVAL=10
	SCALE=M
	UNTIE=false
	LOG_FREQ=50
elif [[ $1 -eq 47 ]]; then # 0.5x=256 dim from 110m
	# batch_size: ~0.52M.
	LAYERS=12
	HIDDEN_SIZE=256
	FFN_SIZE=1024
	NUM_HEADS=2
	NUM_QUERY_GROUPS=1
	MBS="${MBS:-8}"
	GBS=128
	ITERS_PER_BT=2000
	LR=0.001
	SIZE=47
	SAVE_FREQ=10000
	DEF_TOKENS=25
	INTERMEDIATE_METRICS_INTERVAL=10
	SCALE=M
	UNTIE=false
	LOG_FREQ=50
elif [[ $1 -eq 70 ]]; then # 0.75x=384 dim, 9L (ratio-preserving with 110m)
	# batch_size: ~0.52M.
	LAYERS=9
	HIDDEN_SIZE=384
	FFN_SIZE=1536
	NUM_HEADS=6
	NUM_QUERY_GROUPS=3
	MBS="${MBS:-8}"
	GBS=128
	ITERS_PER_BT=2000
	LR=0.001
	SIZE=70
	SAVE_FREQ=10000
	DEF_TOKENS=25
	INTERMEDIATE_METRICS_INTERVAL=10
	SCALE=M
	UNTIE=false
	LOG_FREQ=50
elif [[ $1 -eq 90 ]]; then # 0.5x layers from 110m
	# batch_size: ~0.52M.
	LAYERS=6
	HIDDEN_SIZE=512
	FFN_SIZE=2048
	NUM_HEADS=4
	NUM_QUERY_GROUPS=2
	MBS="${MBS:-8}"
	GBS=128
	ITERS_PER_BT=2000
	LR=0.001
	SIZE=90
	SAVE_FREQ=10000
	DEF_TOKENS=25
	INTERMEDIATE_METRICS_INTERVAL=10
	SCALE=M
	UNTIE=false
	LOG_FREQ=50
elif [[ $1 -eq 130 ]]; then # 1.5x layers from 110m
	# batch_size: ~0.52M.
	LAYERS=18
	HIDDEN_SIZE=512
	FFN_SIZE=2048
	NUM_HEADS=4
	NUM_QUERY_GROUPS=2
	MBS="${MBS:-8}"
	GBS=128
	ITERS_PER_BT=2000
	LR=0.001
	SIZE=130
	SAVE_FREQ=10000
	DEF_TOKENS=25
	INTERMEDIATE_METRICS_INTERVAL=10
	SCALE=M
	UNTIE=false
	LOG_FREQ=50
elif [[ $1 -eq 150 ]]; then # 2x layers from 110m
	# batch_size: ~0.52M.
	LAYERS=24
	HIDDEN_SIZE=512
	FFN_SIZE=2048
	NUM_HEADS=4
	NUM_QUERY_GROUPS=2
	MBS="${MBS:-8}"
	GBS=128
	ITERS_PER_BT=2000
	LR=0.001
	SIZE=150
	SAVE_FREQ=10000
	DEF_TOKENS=25
	INTERMEDIATE_METRICS_INTERVAL=10
	SCALE=M
	UNTIE=false
	LOG_FREQ=50
elif [[ $1 -eq 170 ]]; then # 2.5x layers from 110m
	# batch_size: ~0.52M.
	LAYERS=30
	HIDDEN_SIZE=512
	FFN_SIZE=2048
	NUM_HEADS=4
	NUM_QUERY_GROUPS=2
	MBS="${MBS:-4}"
	GBS=128
	ITERS_PER_BT=2000
	LR=0.001
	SIZE=170
	SAVE_FREQ=10000
	DEF_TOKENS=25
	INTERMEDIATE_METRICS_INTERVAL=10
	SCALE=M
	UNTIE=false
	LOG_FREQ=50
elif [[ $1 -eq 190 ]]; then # 1.5x=768 dim, but WRONG heads/query groups
	# batch_size: ~0.52M.
	LAYERS=12
	HIDDEN_SIZE=768
	FFN_SIZE=3072
	NUM_HEADS=4
	NUM_QUERY_GROUPS=2
	MBS="${MBS:-8}"
	GBS=128
	ITERS_PER_BT=2000
	LR=0.001
	SIZE=190
	SAVE_FREQ=10000
	DEF_TOKENS=25
	INTERMEDIATE_METRICS_INTERVAL=10
	SCALE=M
	UNTIE=false
	LOG_FREQ=50
elif [[ $1 -eq 193 ]]; then # 1.5x=768 dim, correct heads/query groups
	# batch_size: ~0.52M.
	LAYERS=12
	HIDDEN_SIZE=768
	FFN_SIZE=3072
	NUM_HEADS=6
	NUM_QUERY_GROUPS=3
	MBS="${MBS:-8}"
	GBS=128
	ITERS_PER_BT=2000
	LR=0.001
	SIZE=193
	SAVE_FREQ=10000
	DEF_TOKENS=25
	INTERMEDIATE_METRICS_INTERVAL=10
	SCALE=M
	UNTIE=false
	LOG_FREQ=50
elif [[ $1 -eq 232 ]]; then # 1.5x=768 dim, 18L (ratio-preserving with 110m)
	# batch_size: ~0.52M.
	LAYERS=18
	HIDDEN_SIZE=768
	FFN_SIZE=3072
	NUM_HEADS=6
	NUM_QUERY_GROUPS=3
	MBS="${MBS:-8}"
	GBS=128
	ITERS_PER_BT=2000
	LR=0.001
	SIZE=232
	SAVE_FREQ=10000
	DEF_TOKENS=25
	INTERMEDIATE_METRICS_INTERVAL=10
	SCALE=M
	UNTIE=false
	LOG_FREQ=50
elif [[ $1 -eq 290 ]]; then # 2x=1024 dim, but wrong heads/query groups
	# batch_size: ~0.52M.
	LAYERS=12
	HIDDEN_SIZE=1024
	FFN_SIZE=4096
	NUM_HEADS=4
	NUM_QUERY_GROUPS=2
	MBS="${MBS:-8}"
	GBS=128
	ITERS_PER_BT=2000
	LR=0.001
	SIZE=290
	SAVE_FREQ=10000
	DEF_TOKENS=25
	INTERMEDIATE_METRICS_INTERVAL=10
	SCALE=M
	UNTIE=false
	LOG_FREQ=50
elif [[ $1 -eq 292 ]]; then # 2x=1024 dim, correct heads/query groups
	# batch_size: ~0.52M.
	LAYERS=12
	HIDDEN_SIZE=1024
	FFN_SIZE=4096
	NUM_HEADS=8
	NUM_QUERY_GROUPS=4
	MBS="${MBS:-8}"
	GBS=128
	ITERS_PER_BT=2000
	LR=0.001
	SIZE=292
	SAVE_FREQ=10000
	DEF_TOKENS=25
	INTERMEDIATE_METRICS_INTERVAL=10
	SCALE=M
	UNTIE=false
	LOG_FREQ=50
elif [[ $1 -eq 543 ]]; then # 3x=1536 dim, correct heads/query groups
	# batch_size: ~0.52M.
	LAYERS=12
	HIDDEN_SIZE=1536
	FFN_SIZE=6144
	NUM_HEADS=12
	NUM_QUERY_GROUPS=6
	MBS="${MBS:-4}"
	GBS=128
	ITERS_PER_BT=2000
	LR=0.001
	SIZE=543
	SAVE_FREQ=10000
	DEF_TOKENS=25
	INTERMEDIATE_METRICS_INTERVAL=10
	SCALE=M
	UNTIE=false
	LOG_FREQ=50
elif [[ $1 -eq 430 ]]; then # 2x=1024 dim, 24L (ratio-preserving with 110m)
	# batch_size: ~0.52M.
	LAYERS=24
	HIDDEN_SIZE=1024
	FFN_SIZE=4096
	NUM_HEADS=8
	NUM_QUERY_GROUPS=4
	MBS="${MBS:-4}"
	GBS=128
	ITERS_PER_BT=2000
	LR=0.001
	SIZE=430
	SAVE_FREQ=10000
	DEF_TOKENS=25
	INTERMEDIATE_METRICS_INTERVAL=10
	SCALE=M
	UNTIE=false
	LOG_FREQ=50
elif [[ $1 -eq 860 ]]; then # 4x=2048 dim, correct heads/query groups
	# batch_size: ~0.52M.
	LAYERS=12
	HIDDEN_SIZE=2048
	FFN_SIZE=8192
	NUM_HEADS=16
	NUM_QUERY_GROUPS=8
	MBS="${MBS:-4}"
	GBS=128
	ITERS_PER_BT=2000
	LR=0.001
	SIZE=860
	SAVE_FREQ=10000
	DEF_TOKENS=25
	INTERMEDIATE_METRICS_INTERVAL=10
	SCALE=M
	UNTIE=false
	LOG_FREQ=50
elif [[ $1 -eq 390 ]]; then 
	# batch_size: ~0.52M.
	LAYERS=16
	HIDDEN_SIZE=1024
	FFN_SIZE=4096
	NUM_HEADS=8
	NUM_QUERY_GROUPS=4
	MBS="${MBS:-4}"
	GBS=128
	ITERS_PER_BT=2000
	LR=0.001
	SIZE=390
	SAVE_FREQ=10000
	DEF_TOKENS=25
	INTERMEDIATE_METRICS_INTERVAL=10
	SCALE=M
	UNTIE=false
	LOG_FREQ=50
elif [[ $1 -eq 1 ]]; then 
	# batch_size: ~1.05M.
	LAYERS=16
	HIDDEN_SIZE=2048
	FFN_SIZE=8192
	NUM_HEADS=16
	NUM_QUERY_GROUPS=8
	MBS="${MBS:-4}"
	GBS=256
	ITERS_PER_BT=1000
	LR=0.00025
	SIZE=1.5
	SAVE_FREQ=5000
	DEF_TOKENS=125
	INTERMEDIATE_METRICS_INTERVAL=100
	LOG_FREQ=200
else
	>&2 echo "Invalid model size: $1"
	usage
	exit 1
fi
shift

# Default init std: 1/sqrt(hidden_size). Can be overridden by --init.
INIT_STD=$(python3 -c "import math; print(1/math.sqrt($HIDDEN_SIZE))")

# Now get the general options.
TOKENS=$DEF_TOKENS
SUFFIX=""
while [[ $# -gt 0 ]]; do
	case $1 in
		# Misc settings.
		--nodes) NODES=$2; shift 2;;
		--debug)
			SCRIPT_VERSION=$SCRIPT_VERSION-debug
			DEBUG=true
			shift;;
		--extra-name)
			EXTRA_NAME="-$2"; shift 2;;
		--time)
			TIME=$2; shift 2;;
		--no-extra-logs)
			EXTRA_LOG=false; shift;;
		--no-save)
			NO_SAVE=true; shift;;
		--apertus-reservation)
			APERTUS_RESERVATION=true; shift;;
		# FP8 settings.
		--fp8)
			FP8=true; shift;;
		--fp8dpa)
			FP8DPA=true; shift;;
		# Training settings.
		--tokens)
			TOKENS=$2; shift 2;;
		--lr)
			LR=$2; shift 2;;
		--no-warmup)
			NO_WARMUP=true; shift;;
		--warmup-iters)
			WARMUP_ITERS=$2; shift 2;;
		--decay)
			DECAY=$2; shift 2;;
		--cooldown)
			COOLDOWN=$2; shift 2;;
		--branch-from)
			BRANCH_FROM=$2; shift 2;;
		--branch-iter)
			BRANCH_ITER=$2; shift 2;;
		--clip-grad)
			CLIP_GRAD=$2; shift 2;;
		# Architecture settings.
		--init)
			NEW_INIT_STD=$2; shift 2;;
		--scaled-output-init)
			SCALED_OUTPUT_INIT=true; shift;;
		--activation)
			ACTIVATION=$2; shift 2;;
		--no-pre-norm)
			NO_PRE_NORM=true; shift;;
		--no-final-layernorm)
			NO_FINAL_LAYERNORM=true; shift;;
		--normalization)
			NORMALIZATION=$2; shift 2;;
		--no-learnable-norms)
			NO_LEARNABLE_NORMS=true; shift;;
		--post-norm)
			POST_NORM=true; shift;;
		--pre-norm-no-gain)
			PRE_NORM_NO_GAIN=true; shift;;
		--post-norm-no-gain)
			POST_NORM_NO_GAIN=true; shift;;
		--final-layernorm-no-gain)
			FINAL_LAYERNORM_NO_GAIN=true; shift;;
		--post-block-norm)
			POST_BLOCK_NORM=true; shift;;
		--use-stream-minus-residual)
			USE_STREAM_MINUS_RESIDUAL=true; shift;;
		--layer-scale)
			LAYER_SCALE=$2; shift 2;;
		--layer-scale-scale)
			LAYER_SCALE_SCALE=$2; shift 2;;
		--residual-layer-scale)
			RESIDUAL_LAYER_SCALE=$2; shift 2;;
		--residual-layer-scale-scale)
			RESIDUAL_LAYER_SCALE_SCALE=$2; shift 2;;
		--softmax-scale)
			SOFT_MAX_SCALE=$2; shift 2;;
		--qk-norm)
			QK_NORM=$2; shift 2;;
		--qk-frozen)
			QK_FROZEN=true; shift;;
		--qk-layer-scale)
			QK_LAYER_SCALE=$2; shift 2;;
		--qk-layer-scale-scale)
			QK_LAYER_SCALE_SCALE=$2; shift 2;;
		--mlp-layer-scale)
			MLP_LAYER_SCALE=$2; shift 2;;
		--mlp-layer-scale-gate-scale)
			MLP_LAYER_SCALE_GATE_SCALE=$2; shift 2;;
		--mlp-out-scale)
			MLP_OUT_SCALE=$2; shift 2;;
		--fixed-layer-scale)
			FIXED_LAYER_SCALE=$2; shift 2;;
		--logits-layer-scale)
			LOGITS_LAYER_SCALE=$2; shift 2;;
		--logits-layer-scale-scale)
			LOGITS_LAYER_SCALE_SCALE=$2; shift 2;;
		--upscale-embedding)
			UPSCALE_EMBEDDING=$2; shift 2;;
		--untie-embed)
			UNTIE=true; FORCE_UNTIE=true; shift;;
		# Opt settings.
		--opt)
			OPT=$2; shift 2;;
		--master-orthogonalize)
			MASTER_ORTHOGONALIZE=true; shift;;
		--poor-mans-ortho)
			POOR_MANS_ORTHO=true; shift;;
		--b1)
			BETA1=$2; shift 2;;
		--b2)
			BETA2=$2; shift 2;;
		--b3)
			BETA3=$2; shift 2;;
		--mb1)
			ADAMBETA1=$2; shift 2;;
		--mlr)
			MUON_LR_FACTOR=$2; shift 2;;
		--matrix-lr)
			MATRIX_LR=$2; shift 2;;
		--elr)
			EMBEDDING_LR_MULTIPLIER=$2; shift 2;;
		--embedding-lr)
			EMBEDDING_LR=$2; shift 2;;
		--output-lr)
			OUTPUT_LR=$2; shift 2;;
		--scale-min-lr)
			SCALE_MIN_LR=true; shift;;
		--alpha)
			ALPHA=$2; shift 2;;
		--muon-scale)
			MUON_SCALE_MODE=$2; shift 2;;
		--muon-extra-scale-factor)
			MUON_EXTRA_SCALE_FACTOR=$2; shift 2;;
		--muon-nesterov)
			MUON_NESTEROV=true; shift;;
		--muon-num-ns-steps)
			MUON_NUM_NS_STEPS=$2; shift 2;;
		--wd)
			WEIGHT_DECAY=$2; shift 2;;
		--wd-method)
			WEIGHT_DECAY_METHOD=$2; shift 2;;
		--wsd)
			WSD=$2; shift 2;;
		--hs)
			HYPERBALL=$2; shift 2;;
		--hs-kind)
			HS_KIND=$2; shift 2;;
		--hs-r)
			HS_R=$2; shift 2;;
		--hs-u)
			HS_UPDATE=true; shift;;
		--hs-embed)
			HS_EMBED=true
			if [[ $# -ge 2 ]] && [[ $2 != --* ]]; then
				HS_EMBED_MODE=$2; shift
			fi
			shift;;
		--hs-embed-no-orthogonal)
			HS_EMBED_NO_ORTHOGONAL=true; shift;;
		--hs-split-heads)
			HS_SPLIT_HEADS=true; shift;;
		--hs-split-heads-update)
			HS_SPLIT_HEADS_UPDATE=true; shift;;
		--hs-g)
			HS_GAINS_MODE=$2; shift 2;;
		--hs-g-output)
			HS_GAINS_MODE_OUTPUT=$2; shift 2;;
		--hs-g-embed)
			HS_GAINS_MODE_EMBEDDING=$2; shift 2;;
		--glr)
			GAINS_LR=$2; shift 2;;
		--hs-preserve-init)
			HS_PRESERVE_INIT=true; shift;;
		--split-qkv-gains)
			SPLIT_QKV_GAINS=true; shift;;
		--hs-g-param)
			HS_GAINS_PARAM=$2; shift 2;;
		--hs-p)
			HS_PROJECT=true; shift;;
		--hs-s)
			HS_SOFT=true; shift;;
		# Validation.
		--eval-every)
			EVAL_INTERVAL=$2; shift 2;;
		--eval-iters)
			EVAL_ITERS=$2; shift 2;;
		# Logs.
		--wandb-name)
			WANDB_NAME=$2; shift 2;;
		*)
			echo "Unexpected argument $1"
			usage
			exit 1
	esac
done

# Resolve branch checkpoint (cooldown-shape experiments from a fixed ckpt).
if [[ ! -z "${BRANCH_FROM+xxx}" ]]; then
	BRANCH_FROM=${BRANCH_FROM%/}
	if [[ -d "$BRANCH_FROM/checkpoints" ]]; then
		BRANCH_FROM=$BRANCH_FROM/checkpoints
	fi
	if [[ -z "${BRANCH_ITER+xxx}" ]]; then
		BRANCH_ITER=$(cat "$BRANCH_FROM/latest_checkpointed_iteration.txt")
	fi
	BRANCH_ITER_DIR=$(printf "iter_%07d" "$BRANCH_ITER")
	if [[ ! -d "$BRANCH_FROM/$BRANCH_ITER_DIR" ]]; then
		>&2 echo "Branch checkpoint not found: $BRANCH_FROM/$BRANCH_ITER_DIR"
		exit 1
	fi
fi

#= MIDDLE: Set up arguments. =#
# Opt settings.
OPT_ARGS=()
if [[ $OPT = adam ]]; then
	OPT_ARGS+=(--overlap-grad-reduce --use-distributed-optimizer)
	if [[ $BETA1 != 0.9 ]] || [[ $BETA2 != 0.99 ]]; then
		SUFFIX=${SUFFIX}-b${BETA1}_$BETA2
	fi
elif [[ $OPT = muon ]] || [[ $OPT = dmuon ]]; then
	SUFFIX=$SUFFIX-${OPT}_h
	if [[ $BETA1 != 0.9 ]]; then
		SUFFIX=${SUFFIX}_m$BETA1
	fi
	if [[ $ADAMBETA1 != 0.95 ]]; then
		SUFFIX=${SUFFIX}_b$ADAMBETA1
	fi
	if [[ ! -z "${MUON_LR_FACTOR+xxx}" ]]; then
		SUFFIX=${SUFFIX}_mlr$(printf "%.1f" "$MUON_LR_FACTOR")
		OPT_ARGS+=(--muon-lr-factor $MUON_LR_FACTOR)
	fi
	if [[ $MUON_NUM_NS_STEPS != 5 ]]; then
		SUFFIX=${SUFFIX}_mns$MUON_NUM_NS_STEPS
	fi
	if [[ $OPT = dmuon ]]; then
		OPT=dist_muon
	fi
	if [[ $MUON_SCALE_MODE != spectral ]]; then
		if [[ $MUON_SCALE_MODE = unit_rms_norm ]]; then
			SUFFIX=${SUFFIX}_urm
		elif [[ $MUON_SCALE_MODE = shape_scaling ]]; then
			SUFFIX=${SUFFIX}_shsc
		elif [[ $MUON_SCALE_MODE = shape_up ]]; then
			SUFFIX=${SUFFIX}_shup
		else
			SUFFIX=${SUFFIX}_$MUON_SCALE_MODE
		fi
	fi
	if [[ $MUON_NESTEROV = true ]]; then
		# SUFFIX=${SUFFIX}_nest
		OPT_ARGS+=(--muon-use-nesterov)
	fi
	MUON_MOMENTUM=$BETA1
	BETA1=$ADAMBETA1
elif [[ $OPT = dmaster ]] || [[ $OPT = master ]]; then
	# SUFFIX=$SUFFIX-${OPT}_h
	SUFFIX=$SUFFIX-mas
	IS_MASTER_OPT=true
	if [[ $MASTER_ORTHOGONALIZE = true ]]; then
		SUFFIX=${SUFFIX}_o
		if [[ $BETA1 != 0.95 ]]; then
			SUFFIX=${SUFFIX}_m$BETA1
		fi
		if [[ $ADAMBETA1 != 0.9 ]]; then
			SUFFIX=${SUFFIX}_b$ADAMBETA1
		fi
		if [[ ! -z "${MUON_LR_FACTOR+xxx}" ]]; then
			SUFFIX=${SUFFIX}_mlr$(printf "%.1f" "$MUON_LR_FACTOR")
			OPT_ARGS+=(--muon-lr-factor $MUON_LR_FACTOR)
		fi
		if [[ $MUON_NUM_NS_STEPS != 5 ]]; then
			SUFFIX=${SUFFIX}_mns$MUON_NUM_NS_STEPS
		fi
		OPT_ARGS+=(--use-orthogonal-updates)
		if [[ $POOR_MANS_ORTHO = true ]]; then
			SUFFIX=${SUFFIX}_pmo1
			OPT_ARGS+=(--poor-mans-ortho)
		fi
		MUON_MOMENTUM=$BETA1
		BETA1=$ADAMBETA1
		if [[ $MUON_SCALE_MODE != spectral ]]; then
			if [[ $MUON_SCALE_MODE = unit_rms_norm ]]; then
				SUFFIX=${SUFFIX}_urm
			elif [[ $MUON_SCALE_MODE = shape_scaling ]]; then
				SUFFIX=${SUFFIX}_shsc
			elif [[ $MUON_SCALE_MODE = shape_up ]]; then
				SUFFIX=${SUFFIX}
			else
				SUFFIX=${SUFFIX}_$MUON_SCALE_MODE
			fi
		fi
		if [[ $MUON_NESTEROV = true ]]; then
			SUFFIX=${SUFFIX}_nest
			OPT_ARGS+=(--muon-use-nesterov)
		fi
	else
		if [[ $BETA1 != 0.9 ]] || [[ $BETA2 != 0.99 ]] || [[ $BETA3 != 0.999 ]]; then
			SUFFIX=${SUFFIX}_b${BETA1}_${BETA2}_$BETA3
		fi
		if [[ ! -z "${MUON_LR_FACTOR+xxx}" ]]; then
			SUFFIX=${SUFFIX}_mlr$(printf "%.1f" "$MUON_LR_FACTOR")
			OPT_ARGS+=(--muon-lr-factor $MUON_LR_FACTOR)
		fi
	fi
	# if [[ $ALPHA != 5 ]]; then
	# 	SUFFIX=${SUFFIX}_a$ALPHA
	# fi
	if [[ $OPT = dmaster ]]; then
		OPT=dist_master
	fi
elif [[ $OPT = ademamix ]]; then
	SUFFIX=$SUFFIX-amix
	OPT_ARGS+=(--overlap-grad-reduce)
	if [[ $BETA1 != 0.9 ]] || [[ $BETA2 != 0.95 ]] || [[ $BETA3 != 0.999 ]]; then
		SUFFIX=${SUFFIX}_b${BETA1}_${BETA2}_$BETA3
	fi
	# if [[ $ALPHA != 5 ]]; then
	# 	SUFFIX=${SUFFIX}_a$ALPHA
	# fi
	if [[ $HYPERBALL = false ]]; then
		OPT_ARGS+=(--use-distributed-optimizer)
	fi
fi

if [[ $CLIP_GRAD != 1.0 ]]; then
	SUFFIX=$SUFFIX-cg$CLIP_GRAD
	ARCH_ARGS+=(--clip-grad $CLIP_GRAD)
fi

if [[ $WEIGHT_DECAY != 0.0 ]]; then
	SUFFIX=$SUFFIX-wd$WEIGHT_DECAY
fi
if [[ $WEIGHT_DECAY_METHOD != decoupled ]]; then
	if [[ $IS_MASTER_OPT != true ]]; then
		echo "different weight decay method only implemented in the master opt"
		exit 1
	fi
	SUFFIX=$SUFFIX-$WEIGHT_DECAY_METHOD
fi

if [[ $HYPERBALL != false ]]; then
	if [[ $IS_MASTER_OPT != true ]]; then
		echo "hypersphere only implemented for master optimizer"
		exit 1
	fi
	SUFFIX=$SUFFIX-HS${HYPERBALL}${HS_R}_${HS_KIND}
	OPT_ARGS+=(--hypersphere-mode $HYPERBALL --hypersphere-kind $HS_KIND --hypersphere-radius $HS_R)
	if [[ $HS_UPDATE = true ]]; then
		SUFFIX=${SUFFIX}_u
	else
		OPT_ARGS+=(--hypersphere-no-update)
	fi
	if [[ $HS_EMBED = true ]]; then
		if [[ ! -z "${HS_EMBED_MODE+xxx}" ]]; then
			SUFFIX=${SUFFIX}_emb${HS_EMBED_MODE}
			OPT_ARGS+=(--hypersphere-embedding-mode $HS_EMBED_MODE)
		else
			SUFFIX=${SUFFIX}_emb
			OPT_ARGS+=(--hypersphere-embeddings)
		fi
		if [[ $HS_EMBED_NO_ORTHOGONAL = true ]]; then
			SUFFIX=${SUFFIX}NO
			OPT_ARGS+=(--no-use-orthogonal-embeddings)
		fi
	fi
	if [[ $HS_SPLIT_HEADS = true ]]; then
		SUFFIX=${SUFFIX}_sh
		OPT_ARGS+=(--hypersphere-split-heads)
	fi
	if [[ $HS_SPLIT_HEADS_UPDATE = true ]]; then
		SUFFIX=${SUFFIX}_shu
		OPT_ARGS+=(--hypersphere-split-heads-update)
	fi
	if [[ $HS_PROJECT = true ]]; then
		SUFFIX=${SUFFIX}_p
		OPT_ARGS+=(--hypersphere-project)
	fi
	if [[ $HS_SOFT = true ]]; then
		SUFFIX=${SUFFIX}_s
		OPT_ARGS+=(--hypersphere-soft)
	fi
	if [[ ! -z "${HS_GAINS_MODE+xxx}" ]]; then
		SUFFIX=${SUFFIX}_g${HS_GAINS_MODE}
		OPT_ARGS+=(--hypersphere-gains-mode $HS_GAINS_MODE)
		if [[ ! -z "${HS_GAINS_MODE_OUTPUT+xxx}" ]]; then
			SUFFIX=${SUFFIX}_go${HS_GAINS_MODE_OUTPUT}
			OPT_ARGS+=(--hypersphere-gains-mode-output $HS_GAINS_MODE_OUTPUT)
		fi
		if [[ ! -z "${HS_GAINS_MODE_EMBEDDING+xxx}" ]]; then
			# SUFFIX=${SUFFIX}_ge${HS_GAINS_MODE_EMBEDDING}
			OPT_ARGS+=(--hypersphere-gains-mode-embedding $HS_GAINS_MODE_EMBEDDING)
		fi
		if [[ $SPLIT_QKV_GAINS = true ]]; then
			SUFFIX=${SUFFIX}_sqg
			OPT_ARGS+=(--split-qkv-gains)
		fi
		if [[ ! -z "${GAINS_LR+xxx}" ]]; then
			SUFFIX=${SUFFIX}_glr$(printf "%.1e" $GAINS_LR)
			OPT_ARGS+=(--gains-lr $GAINS_LR)
		fi
		if [[ ! -z "${HS_GAINS_PARAM+xxx}" ]] && [[ $HS_GAINS_PARAM != direct ]]; then
			if [[ $HS_GAINS_PARAM = softplus ]]; then
				SUFFIX=${SUFFIX}_gpsp
			elif [[ $HS_GAINS_PARAM = offset ]]; then
				SUFFIX=${SUFFIX}_gpof
			else
				SUFFIX=${SUFFIX}_gp$HS_GAINS_PARAM
			fi
			OPT_ARGS+=(--gain-parametrization $HS_GAINS_PARAM)
		fi
		if [[ $HS_PRESERVE_INIT = true ]]; then
			SUFFIX=${SUFFIX}_pi
			OPT_ARGS+=(--hypersphere-preserve-init)
		fi
	fi
fi

if [[ $LR != 0.001 ]]; then
	SUFFIX=$SUFFIX-lr$(printf "%.1e" $LR)
fi

if [[ ! -z "${MATRIX_LR+xxx}" ]]; then
	SUFFIX=${SUFFIX}-Mlr$(printf "%.1e" $MATRIX_LR)
	OPT_ARGS+=(--matrix-lr $MATRIX_LR)
fi
if [[ $SCALE_MIN_LR = true ]]; then
	SUFFIX=${SUFFIX}-sml
	OPT_ARGS+=(--scale-min-lr)
fi
if [[ ! -z "${EMBEDDING_LR_MULTIPLIER+xxx}" ]]; then
	# SUFFIX=${SUFFIX}-elr
	if [[ $EMBEDDING_LR_MULTIPLIER != 1.0 ]]; then
		SUFFIX=${SUFFIX}$EMBEDDING_LR_MULTIPLIER
	fi
	OPT_ARGS+=(--embedding-lr-multiplier $EMBEDDING_LR_MULTIPLIER)
fi
if [[ ! -z "${EMBEDDING_LR+xxx}" ]]; then
	# if [[ $EMBEDDING_LR != 0.003 ]]; then
	SUFFIX=${SUFFIX}-Elr$(printf "%.1e" $EMBEDDING_LR)
	# fi
	OPT_ARGS+=(--embedding-lr $EMBEDDING_LR)
fi
if [[ ! -z "${OUTPUT_LR+xxx}" ]]; then
	SUFFIX=${SUFFIX}-Olr$(printf "%.1e" $OUTPUT_LR)
	OPT_ARGS+=(--output-lr $OUTPUT_LR)
fi

# FP8 settings.
FP8_ARGS=()
if [[ $FP8 = true ]]; then
	SUFFIX=$SUFFIX-fp8
	FP8_ARGS+=(--fp8-margin 0 --fp8-format hybrid --fp8-amax-history-len 1024 --fp8-amax-compute-algo max --fp8-recipe delayed)
fi
if [[ $FP8DPA = true ]]; then
	SUFFIX=$SUFFIX-fp8dpa
	FP8_ARGS+=(--fp8-margin 0 --fp8-format hybrid --fp8-amax-history-len 1024 --fp8-amax-compute-algo max --fp8-recipe delayed --fp8-dot-product-attention)
fi

# Arch settings.
ARCH_ARGS=()
if [[ ! -z ${NEW_INIT_STD+x} ]]; then
	SUFFIX=$SUFFIX-std$NEW_INIT_STD
	INIT_STD=$NEW_INIT_STD
fi
if [[ $SCALED_OUTPUT_INIT = true ]]; then
	SUFFIX=$SUFFIX-soi
else
	ARCH_ARGS+=(--no-scaled-output-layer-init)
fi

if [[ $ACTIVATION = gelu ]]; then
	FFN_SIZE=$((3*$FFN_SIZE/2))
	SUFFIX=$SUFFIX-$ACTIVATION
elif [[ $ACTIVATION = swiglu ]]; then
	ARCH_ARGS+=(--swiglu)
else
	>&2 echo Unknown activation: $ACTIVATION
	exit 1
fi

if [[ $NORMALIZATION != RMSNorm ]]; then
	SUFFIX=$SUFFIX-$NORMALIZATION
fi
if [[ $NO_LEARNABLE_NORMS = true ]]; then
	SUFFIX=$SUFFIX-fz
	ARCH_ARGS+=(--no-learnable-norms)
fi
if [[ ! -z "${QK_NORM+xxx}" ]]; then
	if [[ $QK_NORM = RMSNorm ]]; then
		if [[ $NORMALIZATION != RMSNorm ]]; then
			echo When using qkRMSNorm you must also use --normalization RMSNorm
			exit 1
		fi
		ARCH_ARGS+=(--qk-layernorm)
		if [[ $QK_FROZEN = true ]]; then
			ARCH_ARGS+=(--qk-layernorm-frozen)
		fi
	elif [[ $QK_NORM = L2Norm ]]; then
		SUFFIX=$SUFFIX-qkL2
		ARCH_ARGS+=(--qk-l2-norm)
	else
		# If not RMSNorm, must indicate in suffix
		SUFFIX=$SUFFIX-qk$QK_NORM
	fi
elif [[ $NORMALIZATION = RMSNorm ]]; then
	# no QK_NORM specified, nothing to add
	:
else
	# If neither/QK_NORM set to something else, reflect it
	SUFFIX=$SUFFIX-nqkRMS
fi

if [[ $NO_PRE_NORM = true ]]; then
	SUFFIX=$SUFFIX-nPre
	ARCH_ARGS+=(--no-pre-norm)
fi
if [[ $NO_FINAL_LAYERNORM = true ]]; then
	SUFFIX=$SUFFIX-nFin
	ARCH_ARGS+=(--no-final-layernorm)
fi
if [[ $POST_NORM = true ]]; then
	# SUFFIX=$SUFFIX-pst
	ARCH_ARGS+=(--post-norm)
fi
if [[ $PRE_NORM_NO_GAIN = true ]]; then
	SUFFIX=$SUFFIX-prng
	ARCH_ARGS+=(--pre-norm-no-gain)
fi
if [[ $POST_NORM_NO_GAIN = true ]]; then
	SUFFIX=$SUFFIX-png
	ARCH_ARGS+=(--post-norm-no-gain)
fi
if [[ $FINAL_LAYERNORM_NO_GAIN = true ]]; then
	SUFFIX=$SUFFIX-fng
	ARCH_ARGS+=(--final-layernorm-no-gain)
fi
if [[ $POST_BLOCK_NORM = true ]]; then
	SUFFIX=$SUFFIX-ppst
	ARCH_ARGS+=(--post-block-norm)
fi
if [[ $USE_STREAM_MINUS_RESIDUAL = true ]]; then
	SUFFIX=$SUFFIX-usmr
	ARCH_ARGS+=(--use-stream-minus-residual)
fi
# if [[ $FORCE_UNTIE = true ]]; then
# 	# SUFFIX=${SUFFIX}-unt
# fi
LONG_SUFFIX=$SUFFIX  # To save checkpoints haha.
if [[ ! -z "${SOFT_MAX_SCALE+xxx}" ]]; then
	SUFFIX=$SUFFIX-ss
	LONG_SUFFIX=$LONG_SUFFIX-ss$SOFT_MAX_SCALE
	ARCH_ARGS+=(--softmax-scale $SOFT_MAX_SCALE)
fi

if [[ ! -z "${LAYER_SCALE+xxx}" ]]; then
	SUFFIX=$SUFFIX-ls${LAYER_SCALE}
	LONG_SUFFIX=$LONG_SUFFIX-ls$LAYER_SCALE
	ARCH_ARGS+=(--layer-scale $LAYER_SCALE)
	if [[ ! -z "${LAYER_SCALE_SCALE+xxx}" ]]; then
		SUFFIX=${SUFFIX}1S$LAYER_SCALE_SCALE
		LONG_SUFFIX=${LONG_SUFFIX}S$LAYER_SCALE_SCALE
		ARCH_ARGS+=(--layer-scale-scale $LAYER_SCALE_SCALE)
	fi
fi

if [[ ! -z "${RESIDUAL_LAYER_SCALE+xxx}" ]]; then
	SUFFIX=$SUFFIX-rls
	LONG_SUFFIX=$LONG_SUFFIX-rls$RESIDUAL_LAYER_SCALE
	ARCH_ARGS+=(--residual-layer-scale $RESIDUAL_LAYER_SCALE)
	if [[ ! -z "${RESIDUAL_LAYER_SCALE_SCALE+xxx}" ]]; then
		SUFFIX=${SUFFIX}S
		LONG_SUFFIX=${LONG_SUFFIX}S$RESIDUAL_LAYER_SCALE_SCALE
		ARCH_ARGS+=(--residual-layer-scale-scale $RESIDUAL_LAYER_SCALE_SCALE)
	fi
fi

if [[ ! -z "${QK_LAYER_SCALE+xxx}" ]]; then
	SUFFIX=$SUFFIX-qkls
	LONG_SUFFIX=$LONG_SUFFIX-qkls$QK_LAYER_SCALE
	ARCH_ARGS+=(--qk-layer-scale $QK_LAYER_SCALE)
	if [[ ! -z "${QK_LAYER_SCALE_SCALE+xxx}" ]]; then
		LONG_SUFFIX=${LONG_SUFFIX}S$QK_LAYER_SCALE_SCALE
		SUFFIX=${SUFFIX}S
		ARCH_ARGS+=(--qk-layer-scale-scale $QK_LAYER_SCALE_SCALE)
	fi
fi
if [[ ! -z "${MLP_LAYER_SCALE+xxx}" ]]; then
	SUFFIX=$SUFFIX-mlpls
	LONG_SUFFIX=$LONG_SUFFIX-mlpls$MLP_LAYER_SCALE
	ARCH_ARGS+=(--mlp-layer-scale $MLP_LAYER_SCALE)
	if [[ ! -z "${MLP_LAYER_SCALE_GATE_SCALE+xxx}" ]]; then
		SUFFIX=${SUFFIX}G
		LONG_SUFFIX=${LONG_SUFFIX}G$MLP_LAYER_SCALE_GATE_SCALE
		ARCH_ARGS+=(--mlp-layer-scale-gate-scale $MLP_LAYER_SCALE_GATE_SCALE --no-bias-swiglu-fusion)
	fi
else
	if [[ ! -z "${MLP_LAYER_SCALE_GATE_SCALE+xxx}" ]]; then
		SUFFIX=${SUFFIX}-mlpG
		LONG_SUFFIX=${LONG_SUFFIX}-mlpG$MLP_LAYER_SCALE_GATE_SCALE
		ARCH_ARGS+=(--mlp-layer-scale-gate-scale $MLP_LAYER_SCALE_GATE_SCALE --no-bias-swiglu-fusion)
	fi
fi
if [[ ! -z "${MLP_OUT_SCALE+xxx}" ]]; then
	SUFFIX=$SUFFIX-mlpO
	LONG_SUFFIX=$LONG_SUFFIX-mlpO$MLP_OUT_SCALE
	ARCH_ARGS+=(--mlp-out-scale $MLP_OUT_SCALE)
fi
if [[ ! -z "${FIXED_LAYER_SCALE+xxx}" ]]; then
	SUFFIX=$SUFFIX-fls${FIXED_LAYER_SCALE}
	LONG_SUFFIX=$LONG_SUFFIX-fls$FIXED_LAYER_SCALE
	ARCH_ARGS+=(--fixed-layer-scale $FIXED_LAYER_SCALE)
fi
if [[ ! -z "${LOGITS_LAYER_SCALE+xxx}" ]]; then
	SUFFIX=$SUFFIX-lg$LOGITS_LAYER_SCALE
	LONG_SUFFIX=$LONG_SUFFIX-lgsls$LOGITS_LAYER_SCALE
	ARCH_ARGS+=(--logits-layer-scale $LOGITS_LAYER_SCALE)
	if [[ ! -z "${LOGITS_LAYER_SCALE_SCALE+xxx}" ]]; then
		SUFFIX=${SUFFIX}S
		LONG_SUFFIX=${LONG_SUFFIX}S$LOGITS_LAYER_SCALE_SCALE
		ARCH_ARGS+=(--logits-layer-scale-scale $LOGITS_LAYER_SCALE_SCALE)
	fi
fi
if [[ ! -z "${UPSCALE_EMBEDDING+xxx}" ]]; then
	# SUFFIX=$SUFFIX-ue
	LONG_SUFFIX=$LONG_SUFFIX-up$UPSCALE_EMBEDDING
	ARCH_ARGS+=(--upscale-embedding $UPSCALE_EMBEDDING)
fi


# Training settings.
if [[ $NO_WARMUP = true ]]; then
	# SUFFIX=$SUFFIX-nw
	LONG_SUFFIX=$LONG_SUFFIX-nw
	WARMUP=0
else
	WARMUP=$WARMUP_ITERS
	if [[ $WARMUP_ITERS -ne 1000 ]]; then
		SUFFIX=$SUFFIX-wu$WARMUP_ITERS
		LONG_SUFFIX=$LONG_SUFFIX-wu$WARMUP_ITERS
	fi
fi
if [[ $TOKENS != $DEF_TOKENS ]]; then
	SUFFIX=$SUFFIX-${TOKENS}BT
	LONG_SUFFIX=$LONG_SUFFIX-${TOKENS}BT
fi
ITERS=$(python3 -c "print(int($ITERS_PER_BT*$TOKENS))")  # python so --tokens can be fractional (e.g. 37.5 -> 75k iters)

DECAY_ARGS=()
if [[ $DECAY = wsd ]]; then
	if [[ $COOLDOWN != 0.2 ]]; then
		SUFFIX=$SUFFIX-cd$COOLDOWN
		LONG_SUFFIX=$LONG_SUFFIX-cd${COOLDOWN}
	fi
	if [[ $WSD != minus_sqrt ]]; then
		SUFFIX=$SUFFIX-WSD$WSD
		LONG_SUFFIX=$LONG_SUFFIX-WSD$WSD
	fi
	DECAY_ITERS=$(python3 -c "print(int($ITERS * $COOLDOWN))")
	DECAY_ARGS+=(
		--lr-decay-style WSD
		--lr-wsd-decay-style $WSD
		--lr-wsd-decay-iters $DECAY_ITERS
	)
elif [[ $DECAY = cos ]]; then
	SUFFIX=$SUFFIX-cos
	LONG_SUFFIX=$LONG_SUFFIX-cos
	DECAY_ARGS+=(
		--lr-decay-style cosine
		--lr-decay-iters $ITERS
	)
elif [[ $DECAY = linear ]]; then
	SUFFIX=$SUFFIX-lin
	LONG_SUFFIX=$LONG_SUFFIX-lin
	DECAY_ARGS+=(
		--lr-decay-style linear
		--lr-decay-iters $ITERS
	)
elif [[ $DECAY = inverse-square-root ]]; then
	SUFFIX=$SUFFIX-invsq
	LONG_SUFFIX=$LONG_SUFFIX-invsq
	DECAY_ARGS+=(
		--lr-decay-style inverse-square-root
		--lr-decay-iters $ITERS
	)
elif [[ $DECAY = isrwsd ]]; then
	SUFFIX=$SUFFIX-isrwsd
	LONG_SUFFIX=$LONG_SUFFIX-isrwsd
	if [[ $COOLDOWN != 0.2 ]]; then
		SUFFIX=$SUFFIX-cd$COOLDOWN
		LONG_SUFFIX=$LONG_SUFFIX-cd${COOLDOWN}
	fi
	if [[ $WSD != minus_sqrt ]]; then
		SUFFIX=$SUFFIX-$WSD
		LONG_SUFFIX=$LONG_SUFFIX-$WSD
	fi
	DECAY_ITERS=$(python3 -c "print(int($ITERS * $COOLDOWN))")
	DECAY_ARGS+=(
		--lr-decay-style inverse-square-root-WSD
		--lr-wsd-decay-style $WSD
		--lr-wsd-decay-iters $DECAY_ITERS
	)
else
	echo "Unknown decay method $DECAY"
	exit 1
fi

# In order to make the name shorter, we will alias all the suffixes that correspond to
# ngpt architecture to just show ngpt.
NGPT_SUBSTRING=-L2Norm-fz-nPre-nFin-pst-ppst-usmr-ss-lsS-qklsS-mlplsG-lgsls
SUFFIX="${SUFFIX/$NGPT_SUBSTRING/-ngpt}"

SUFFIX=$SUFFIX$EXTRA_NAME
LONG_SUFFIX=$LONG_SUFFIX$EXTRA_NAME
EXTRA_LOGS=()
if [[ $EXTRA_LOG = true ]]; then
	EXTRA_LOGS+=(
		--log-validation-ppl-to-tensorboard
		--log-params-norm-per-param
		--log-num-zeros-in-grad
		--log-params-norm
		--log-progress
		--log-timers-to-tensorboard
		--log-model-internals
		--log-activation-stats
		--log-gradient-stats
		--log-angular-metrics
		--log-relative-updates
		--log-delta-y
		--log-update-step-stats
		--internals-log-interval $LOG_FREQ
	)
fi

# Final preparations.
if [ "$DEBUG" = true ]; then
	WANDB_ENTITY=alehc
else
	WANDB_ENTITY=epfl-relay
fi
WANDB_PROJECT=megatron_opt_$SCRIPT_VERSION
EXP_NAME=$SIZE$SCALE"1"$SUFFIX
ROOT_PATH=$TRAIN_ROOT/$SCRIPT_VERSION/$SIZE$SCALE"1"$LONG_SUFFIX
DEBUG_ROOT=$ROOT_PATH/debug
SAVE_PATH=$ROOT_PATH/checkpoints
DIFFS_PATH=$ROOT_PATH/diffs
TRIGGER_DIR=$ROOT_PATH/triggerdir
TRIGGER_LOCK=$TRIGGER_DIR/training_lock
mkdir -p $SAVE_PATH
mkdir -p $DIFFS_PATH
mkdir -p $TRIGGER_DIR

# For branch runs: seed the (new) save dir once with the source checkpoint, then
# load/save from this dir as usual so the run stays preemption-safe.
if [[ ! -z "${BRANCH_FROM+xxx}" ]]; then
	if [[ ! -f $SAVE_PATH/latest_checkpointed_iteration.txt ]]; then
		ln -s "$BRANCH_FROM/$BRANCH_ITER_DIR" "$SAVE_PATH/$BRANCH_ITER_DIR"
		echo "$BRANCH_ITER" > "$SAVE_PATH/latest_checkpointed_iteration.txt"
		echo "Seeded $SAVE_PATH <- $BRANCH_FROM/$BRANCH_ITER_DIR (iter $BRANCH_ITER)"
	else
		echo "$SAVE_PATH already has checkpoints; not re-seeding (resume-safe)."
	fi
fi

if [[ -z ${WANDB_NAME+x} ]]; then
	WANDB_NAME=$EXP_NAME
fi

if [[ ${#WANDB_NAME} -gt 117 ]]; then
	# We compare against 117 because WANDB_NAME gets appended -n$NODE_COUNT-$JOB_ID in the final sbatch.
	CHARS=${#WANDB_NAME}
	>&2 echo "WANDB_NAME is too long (it has $CHARS characters, shouldn't exceed 117 characters): $WANDB_NAME"
	exit 1
fi

#= WRAPPING UP: Set up the _ARGS variables that are going to be used in the end =#

LLAMA_ARGS=(
	--tensor-model-parallel-size $TP
	--pipeline-model-parallel-size $PP
	$MAYBE_VIRTUAL
	--seq-length $SEQ_LEN
	--max-position-embeddings $SEQ_LEN
	--tokenizer-type HuggingFaceTokenizer
	--tokenizer-model $TOKENIZER
	--normalization $NORMALIZATION
	--position-embedding-type rope
	--attention-softmax-in-fp32
	--disable-bias-linear
	--transformer-impl transformer_engine
	--num-layers $LAYERS
	--hidden-size $HIDDEN_SIZE
	--group-query-attention
	--num-query-groups $NUM_QUERY_GROUPS
	--ffn-hidden-size $FFN_SIZE
	--num-attention-heads $NUM_HEADS
	--attention-dropout 0.0
	--hidden-dropout 0.0
	--rotary-base 500000
	--rotary-percent 1.0
	--use-rope-scaling
	--bf16
	--optimizer $OPT
	--adam-eps 0.00000001
	--norm-epsilon 0.00001
)
if [[ $UNTIE = true ]]; then
	LLAMA_ARGS+=(--untie-embeddings-and-output-weights)
fi

TRAINING_ARGS=(
	--micro-batch-size $MBS
	--global-batch-size $GBS
	--train-iters $ITERS
	--eval-interval $([[ $EVAL_INTERVAL -gt 0 ]] && echo $EVAL_INTERVAL || echo $((ITERS + 1)))
	--eval-iters $EVAL_ITERS
	--weight-decay $WEIGHT_DECAY
	--weight-decay-method $WEIGHT_DECAY_METHOD
	--adam-beta1 $BETA1
	--muon-momentum $MUON_MOMENTUM
	--muon-scale-mode $MUON_SCALE_MODE
	--muon-extra-scale-factor $MUON_EXTRA_SCALE_FACTOR
	--muon-num-ns-steps $MUON_NUM_NS_STEPS
	--adam-beta2 $BETA2
	--ademamix-beta3 $BETA3
	--ademamix-alpha $ALPHA
	--ademamix-beta3-warmup $ITERS
	--ademamix-alpha-warmup $ITERS
	--init-method-std $INIT_STD
	--clip-grad $CLIP_GRAD
	--lr $LR
	--min-lr $MIN_LR
	--trigger-path $TRIGGER_DIR
	--exit-signal-handler
)

DATA_ARGS=(
	--data-cache-path $SCRATCH/data/cache_opt
)
if [[ $EVAL_INTERVAL -gt 0 ]]; then
	DATA_ARGS+=(
		--train-data-path $DATA_PATHS
		--valid-data-path $VALID_DATA_PATH
		--deterministic-validation
	)
else
	DATA_ARGS+=(
		--data-path $DATA_PATHS
		--split 100,0,0
	)
fi
	#--mock-data

LOGGING=(
	--log-interval 1
	--load $SAVE_PATH
	--tensorboard-dir $ROOT_PATH/tensorboard
	--wandb-project $WANDB_PROJECT
	--wandb-save-dir $ROOT_PATH/wandb
	--wandb-entity $WANDB_ENTITY
	--timing-log-level 1
	--tensorboard-log-interval 1
	--log-throughput
)
if [[ $NO_SAVE = false ]]; then
	LOGGING+=(
		--save-interval $SAVE_FREQ
		--save $SAVE_PATH
	)
fi
LOGGING=(${LOGGING[@]} ${EXTRA_LOGS[@]})

SCHEDULER_ARGS=(
	--lr-warmup-iters $WARMUP
)

EXTRA_ARGS+=(
	--ckpt-format torch
)
if [[ ! -z "${BRANCH_FROM+xxx}" ]]; then
	# Apply the new (cooldown) schedule from CLI args instead of the checkpoint's.
	EXTRA_ARGS+=(--override-opt_param-scheduler)
fi

ARGS="${LLAMA_ARGS[@]} ${TRAINING_ARGS[@]} ${SCHEDULER_ARGS[@]} ${DATA_ARGS[@]} ${LOGGING[@]} ${EXTRA_ARGS[@]} ${FP8_ARGS[@]} ${ARCH_ARGS[@]} ${OPT_ARGS[@]} ${DECAY_ARGS[@]}"

#= RUNNING: Prepare and launch a slurm script =#
CMD="numactl --membind=0-3 env python3 pretrain_gpt.py $ARGS"

RESERVATION_DIRECTIVE=""
if [[ $APERTUS_RESERVATION = true ]]; then
	RESERVATION_DIRECTIVE="#SBATCH --reservation=SD-69241-apertus-1-5-0"
fi

mkdir -p $ROOT_PATH
cat > $ROOT_PATH/submission.sbatch <<- EOM
#!/bin/bash
#SBATCH --time=$TIME
#SBATCH --job-name=$EXP_NAME
#SBATCH --output=$ROOT_PATH/slurmlogs/%j.out
#SBATCH --error=$ROOT_PATH/slurmlogs/%j.err
#SBATCH --nodes=$NODES
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=36
#SBATCH --mem=460000
#SBATCH --exclusive
#SBATCH --account=infra01
#SBATCH --partition=${PARTITION:-normal}
#SBATCH --signal=SIGTERM@180
#SBATCH --dependency=singleton
$RESERVATION_DIRECTIVE


# Wake up.
echo [\$(date)] Starting job
echo [\$(date)] Using nodes: \$SLURM_JOB_NODELIST
srun --environment=$CONTAINER -l --mpi=pmix --network=disable_rdzv_get bash -c 'echo \$(hostname) \$(nvidia-smi | grep -o "|\\s*[0-9]*MiB")'


# Log git status.
cd $CODE_PATH
echo ---------
echo git status:
git status
echo git log:
git log -n 1
echo ---------
git diff > $DIFFS_PATH/\$SLURM_JOBID.diff

export MASTER_ADDR=\$(hostname)
export WORLD_SIZE=\$SLURM_NPROCS
export MASTER_PORT=25678

export WANDB__FILE_STREAM_RETRY_MAX=10
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK

export TRITON_HOME=$TRITON_HOME_DIR
export TRITON_CACHE_DIR=$TRITON_HOME_DIR/cache

# debug!
DEBUG_DIR=$DEBUG_ROOT/\$SLURM_JOB_ID
mkdir -p \$DEBUG_DIR
cp \$0 \$DEBUG_DIR/submit.sbatch
cat \$SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment > \$DEBUG_DIR/env.toml
echo "\\nMegatron path: $CODE_PATH (\$(git -C $CODE_PATH rev-parse --verify HEAD))" > \$DEBUG_DIR/git
git diff > \$DEBUG_DIR/git.diff
pip list > \$DEBUG_DIR/pip.txt
nvidia-smi > \$DEBUG_DIR/cuda
printenv > \$DEBUG_DIR/env.sh

srun --environment=$CONTAINER -lu --cpus-per-task \$SLURM_CPUS_PER_TASK --mpi=pmix --network=disable_rdzv_get bash -c "
	cd $CODE_PATH
	export PYTHONPATH=\$PWD:$EMERGING_OPTIMIZERS_PATH
	export RANK=\\\$SLURM_PROCID
	export LOCAL_RANK=\\\$SLURM_LOCALID
	mkdir -p \\\$TRITON_CACHE_DIR
	$CMD --wandb-exp-name $EXP_NAME-n$NODES-\\\$SLURM_JOBID
"

# Goodbye lol.
echo [\$(date)] Goodbye
EOM
echo "Saved sbatch to $ROOT_PATH/submission.sbatch"

sbatch $ROOT_PATH/submission.sbatch
