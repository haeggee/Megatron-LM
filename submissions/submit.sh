# TODO: fp8dpa
# TODO: ademamix parameters
# TODO: extra logs

#= Prelude =#
# General settings.
#CONTAINER=/iopsstor/scratch/cscs/ahernnde/ngc_25-12-alps1.toml
CONTAINER=/iopsstor/scratch/cscs/ahernnde/ncg_new_v2.toml

SCRIPT_VERSION=v0.1
SEQ_LEN=4096
TOKENIZER=mistralai/Mistral-Nemo-Base-2407
TOKENIZED_DATA_PATH=/iopsstor/scratch/cscs/jpcoles/a06/swissai-fineweb-edu-filterrobots-merge
DATA_PATHS=$(find $TOKENIZED_DATA_PATH -type f | sed -E 's/\.[^.]+$//' | sort -u | tr '\n' ' ')
CODE_PATH=$PWD
EMERGING_OPTIMIZERS_PATH=/iopsstor/scratch/cscs/ahernnde/opts/Emerging-Optimizers
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
MIN_LR=1e-8
OPT=adam

BETA1=0.9
BETA2=0.95
BETA3=0.999
ALPHA=5

HYPERBALL=false
HB_KIND=l2
HB_R=1
HB_UPDATE=false
HB_EMBED=false
HB_SPLIT_HEADS=false

NO_WARMUP=false

# Misc. defaults.
EXTRA_LOG=true

# Usage function.
usage () {
	echo "Usage: submit.sh <size> [options...]"
	echo "<size>: 110/390/1"
	echo "Options:"
	# Misc settings.
	echo " --nodes <nodes>: How many nodes to use"
	echo " --extra-name <name>: Add a suffix to the name"
	echo " --time (default=$TIME): Change the sbatch time limit"
	# FP8 settings.
	echo " --fp8: Enables fp8"
	# Training settings..
	echo " --tokens <int>: Amount of tokens to train with (in B)."
	echo " --lr <float>: Learning rate."
	echo " --no-warmup: Deactivates learning rate warmup"
	# Architecture settings.
	echo " --init <float>: Change init std."
	echo " --no-pre-norm"
	echo " --no-final-layernorm"
	echo " --normalization <RMSNorm/L2Norm>"
	echo " --no-learnable-norms"
	echo " --post-norm"
	echo " --post-block-norm"
	echo " --use-stream-minus-residual"
	echo " --layer-scale <float>"
	echo " --layer-scale-scale <float>"
	echo " --softmax-scale <float>"
	echo " --qk-norm <RMSNorm/L2Norm>"
	echo " --mlp-layer-scale <float>"
	echo " --mlp-layer-scale-gate-scale <float>"
	echo " --logits-layer-scale <float>"
	echo " --logits-layer-scale-scale <float>"
	# Optimizer settings.
	echo " --opt <adam/dmuon/muon/dmaster/master/ademamix> (default=$OPT)"
	echo " --master-orthogonalize"
	echo " --b1: beta1 (master&adam&muon&ademamix)"
	echo " --b2: beta2 (master&adam&ademamix)"
	echo " --mb2: beta2 (master)"
	echo " --b3: beta3 (master&ademamix)"
	echo " --alpha: ademamix alpha"
	echo " --wd: weight decay"
	echo " --hb <row/col/rowcol/flat>: Enables hyperball training"
	echo " --hb-kind <l2/standard/spectral>: hyperball kind"
	echo " --hb-r <learnable/float>: hyperball radius"
	echo " --hb-u: hyperball normalize update"
	echo " --hb-embed: hyperball normalize embeddings"
	echo " --hb-split-heads: hyperball normalize q,k,v heads separately"
	# Logs.
	echo " --wandb-name <str>: Specify wandb name."
	echo " --no-extra-log"
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
INIT_STD=0.02
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
	LR=0.002
	SIZE=110
	SAVE_FREQ=10000
	DEF_TOKENS=25
	INTERMEDIATE_METRICS_INTERVAL=10
	SCALE=M
	UNTIE=false
elif [[ $1 -eq 390 ]]; then 
	# batch_size: ~0.52M.
	LAYERS=16
	HIDDEN_SIZE=1024
	FFN_SIZE=4096
	NUM_HEADS=8
	NUM_QUERY_GROUPS=4
	MBS="${MBS:-8}"
	GBS=128
	ITERS_PER_BT=2000
	LR=0.001
	SIZE=390
	SAVE_FREQ=10000
	DEF_TOKENS=50
	INTERMEDIATE_METRICS_INTERVAL=10
	SCALE=M
	UNTIE=false
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
else
	>&2 echo "Invalid model size: $1"
	usage
	exit 1
fi
shift

# Now get the general options.
TOKENS=$DEF_TOKENS
SUFFIX=""
while [[ $# -gt 0 ]]; do
	case $1 in
		# Misc settings.
		--nodes) NODES=$2; shift 2;;
		--extra-name)
			EXTRA_NAME="-$2"; shift 2;;
		--time)
			TIME=$2; shift 2;;
		# FP8 settings.
		--fp8)
			FP8=true; shift;;
		# Training settings.
		--tokens)
			TOKENS=$2; shift 2;;
		--lr)
			LR=$2; 
			CHANGED_LR=true
			shift 2;;
		--no-warmup)
			NO_WARMUP=true; shift;;
		# Architecture settings.
		--init)
			NEW_INIT_STD=$2; shift 2;;
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
		--post-block-norm)
			POST_BLOCK_NORM=true; shift;;
		--use-stream-minus-residual)
			USE_STREAM_MINUS_RESIDUAL=true; shift;;
		--layer-scale)
			LAYER_SCALE=$2; shift 2;;
		--layer-scale-scale)
			LAYER_SCALE_SCALE=$2; shift 2;;
		--softmax-scale)
			SOFT_MAX_SCALE=$2; shift 2;;
		--qk-norm)
			QK_NORM=$2; shift 2;;
		--qk-layer-scale)
			QK_LAYER_SCALE=$2; shift 2;;
		--qk-layer-scale-scale)
			QK_LAYER_SCALE_SCALE=$2; shift 2;;
		--mlp-layer-scale)
			MLP_LAYER_SCALE=$2; shift 2;;
		--mlp-layer-scale-gate-scale)
			MLP_LAYER_SCALE_GATE_SCALE=$2; shift 2;;
		--logits-layer-scale)
			LOGITS_LAYER_SCALE=$2; shift 2;;
		--logits-layer-scale-scale)
			LOGITS_LAYER_SCALE_SCALE=$2; shift 2;;
		# Opt settings.
		--opt)
			OPT=$2; shift 2;;
		--master-orthogonalize)
			MASTER_ORTHOGONALIZE=true; shift;;
		--b1)
			BETA1=$2; shift 2;;
		--b2)
			BETA2=$2; shift 2;;
		--b3)
			BETA3=$2; shift 2;;
		--alpha)
			ALPHA=$2; shift 2;;
		--wd)
			WEIGHT_DECAY=$2; shift 2;;
		--hb)
			HYPERBALL=$2; shift 2;;
		--hb-kind)
			HB_KIND=$2; shift 2;;
		--hb-r)
			HB_R=$2; shift 2;;
		--hb-u)
			HB_UPDATE=true; shift;;
		--hb-embed)
			HB_EMBED=true; shift;;
		--hb-split-heads)
			HB_SPLIT_HEADS=true; shift;;
		# Logs.
		--wandb-name)
			WANDB_NAME=$2; shift 2;;
		--no-extra-log)
			EXTRA_LOG=false; shift;;
		*)
			echo "Unexpected argument $1"
			usage
			exit 1
	esac
done

#= MIDDLE: Set up arguments. =#
# Opt settings.
OPT_ARGS=()
if [[ $OPT = adam ]]; then
	OPT_ARGS+=(--overlap-grad-reduce --use-distributed-optimizer)
	if [[ $HYPERBALL != false ]]; then
		echo "Adam hypersphere NYI"
		exit 1
	fi
	if [[ $BETA1 != 0.9 ]] || [[ $BETA2 != 0.95 ]]; then
		SUFFIX=${SUFFIX}-b${BETA1}_$BETA2
	fi
elif [[ $OPT = muon ]] || [[ $OPT = dmuon ]]; then
	SUFFIX=$SUFFIX-$OPT
	if [[ $BETA1 != 0.9 ]]; then
		SUFFIX=${SUFFIX}_m$BETA1
	fi
	if [[ $OPT = dmuon ]]; then
		OPT=dist_muon
	fi
elif [[ $OPT = dmaster ]] || [[ $OPT = master ]]; then
	SUFFIX=$SUFFIX-$OPT
	if [[ $BETA1 != 0.9 ]] || [[ $BETA2 != 0.95 ]] || [[ $BETA3 != 0.999 ]]; then
		SUFFIX=${SUFFIX}_b${BETA1}_${BETA2}_$BETA3
	fi
	if [[ $ALPHA != 5 ]]; then
		SUFFIX=${SUFFIX}_a$ALPHA
	fi
	if [[ $MASTER_ORTHOGONALIZE = true ]]; then
		SUFFIX=${SUFFIX}_o
		OPT_ARGS+=(--use-orthogonal-updates)
	fi
	if [[ $OPT = dmaster ]]; then
		OPT=dist_master
	fi
elif [[ $OPT = ademamix ]]; then
	SUFFIX=$SUFFIX-amix
	OPT_ARGS+=(--overlap-grad-reduce)
	if [[ $BETA1 != 0.9 ]] || [[ $BETA2 != 0.95 ]] || [[ $BETA3 != 0.999 ]]; then
		SUFFIX=${SUFFIX}_b${BETA1}_${BETA2}_$BETA3
	fi
	if [[ $ALPHA != 5 ]]; then
		SUFFIX=${SUFFIX}_a$ALPHA
	fi
	if [[ $HYPERBALL = false ]]; then
		OPT_ARGS+=(--use-distributed-optimizer)
	fi
fi

if [[ $WEIGHT_DECAY != 0.1 ]]; then
	SUFFIX=$SUFFIX-wd$WEIGHT_DECAY
fi

if [[ $HYPERBALL != false ]]; then
	SUFFIX=$SUFFIX-HB${HYPERBALL}${HB_R}_$HB_KIND
	OPT_ARGS+=(--hyperball-mode $HYPERBALL --hyperball-kind $HB_KIND --hyperball-radius $HB_R)
	if [[ $HB_UPDATE = true ]]; then
		SUFFIX=${SUFFIX}_u
	else
		OPT_ARGS+=(--hyperball-no-update)
	fi
	if [[ $HB_EMBED = true ]]; then
		SUFFIX=${SUFFIX}_emb
		OPT_ARGS+=(--hyperball-embeddings)
	fi
	if [[ $HB_SPLIT_HEADS = true ]]; then
		SUFFIX=${SUFFIX}_sh
		OPT_ARGS+=(--hyperball-split-heads)
	fi
fi

if [[ $CHANGED_LR = true ]]; then
	SUFFIX=$SUFFIX-lr$LR
fi

# FP8 settings.
FP8_ARGS=()
if [[ $FP8 = true ]]; then
	SUFFIX=$SUFFIX-fp8
	FP8_ARGS+=(--fp8-margin 0 --fp8-format hybrid --fp8-amax-history-len 1024 --fp8-amax-compute-algo max --fp8-recipe delayed)
fi

# Arch settings.
ARCH_ARGS=()
if [[ ! -z ${NEW_INIT_STD+x} ]]; then
	SUFFIX=$SUFFIX-std$NEW_INIT_STD
	INIT_STD=$NEW_INIT_STD
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
		SUFFIX=$SUFFIX-qkRMS
		ARCH_ARGS+=(--qk-layernorm)
	elif [[ $QK_NORM = L2Norm ]]; then
		SUFFIX=$SUFFIX-qkL2
		ARCH_ARGS+=(--qk-l2-norm)
	fi
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
	SUFFIX=$SUFFIX-pst
	ARCH_ARGS+=(--post-norm)
fi
if [[ $POST_BLOCK_NORM = true ]]; then
	SUFFIX=$SUFFIX-ppst
	ARCH_ARGS+=(--post-block-norm)
fi
if [[ $USE_STREAM_MINUS_RESIDUAL = true ]]; then
	SUFFIX=$SUFFIX-usmr
	ARCH_ARGS+=(--use-stream-minus-residual)
fi
if [[ ! -z "${SOFT_MAX_SCALE+xxx}" ]]; then
	SUFFIX=$SUFFIX-ss
	ARCH_ARGS+=(--softmax-scale $SOFT_MAX_SCALE)
fi

if [[ ! -z "${LAYER_SCALE+xxx}" ]]; then
	SUFFIX=$SUFFIX-ls
	ARCH_ARGS+=(--layer-scale $LAYER_SCALE)
	if [[ ! -z "${LAYER_SCALE_SCALE+xxx}" ]]; then
		SUFFIX=${SUFFIX}S
		ARCH_ARGS+=(--layer-scale-scale $LAYER_SCALE_SCALE)
	fi
fi
if [[ ! -z "${QK_LAYER_SCALE+xxx}" ]]; then
	SUFFIX=$SUFFIX-qkls
	ARCH_ARGS+=(--qk-layer-scale $QK_LAYER_SCALE)
	if [[ ! -z "${QK_LAYER_SCALE_SCALE+xxx}" ]]; then
		SUFFIX=${SUFFIX}S
		ARCH_ARGS+=(--qk-layer-scale-scale $QK_LAYER_SCALE_SCALE)
	fi
fi
if [[ ! -z "${MLP_LAYER_SCALE+xxx}" ]]; then
	SUFFIX=$SUFFIX-mlpls
	ARCH_ARGS+=(--mlp-layer-scale $MLP_LAYER_SCALE)
	if [[ ! -z "${MLP_LAYER_SCALE_GATE_SCALE+xxx}" ]]; then
		SUFFIX=${SUFFIX}G
		ARCH_ARGS+=(--mlp-layer-scale-gate-scale $MLP_LAYER_SCALE_GATE_SCALE --no-bias-swiglu-fusion)
	fi
fi
if [[ ! -z "${LOGITS_LAYER_SCALE+xxx}" ]]; then
	SUFFIX=$SUFFIX-lgsls
	ARCH_ARGS+=(--logits-layer-scale $LOGITS_LAYER_SCALE)
	if [[ ! -z "${LOGITS_LAYER_SCALE_SCALE+xxx}" ]]; then
		SUFFIX=${SUFFIX}S
		ARCH_ARGS+=(--logits-layer-scale-scale $LOGITS_LAYER_SCALE_SCALE)
	fi
fi

# Training settings.
if [[ $NO_WARMUP = true ]]; then
	SUFFIX=$SUFFIX-nw
	WARMUP=0
else
	WARMUP=$((5*ITERS_PER_BT/2))  # 2.5BT.
fi
if [[ $TOKENS != $DEF_TOKENS ]]; then
	SUFFIX=$SUFFIX-${TOKENS}BT
fi
ITERS=$((ITERS_PER_BT*TOKENS))
DECAY_ITERS=$(($ITERS/5))

EXTRA_LOGS=()
SUFFIX=$SUFFIX$EXTRA_NAME
if [[ $EXTRA_LOG = true ]]; then
	EXTRA_LOGS+=(
		--log-validation-ppl-to-tensorboard
		--log-params-norm-per-param
		--log-num-zeros-in-grad
		--log-params-norm
		--log-progress
	)
fi

# Final preparations.
WANDB_PROJECT=opt_$SCRIPT_VERSION
EXP_NAME=$SIZE$SCALE$SUFFIX
ROOT_PATH=$TRAIN_ROOT/$EXP_NAME
DEBUG_ROOT=$ROOT_PATH/debug
SAVE_PATH=$ROOT_PATH/checkpoints
DIFFS_PATH=$ROOT_PATH/diffs
TRIGGER_DIR=$ROOT_PATH/triggerdir
TRIGGER_LOCK=$TRIGGER_DIR/training_lock
mkdir -p $SAVE_PATH
mkdir -p $DIFFS_PATH
mkdir -p $TRIGGER_DIR

if [[ -z ${WANDB_NAME+x} ]]; then
	WANDB_NAME=$EXP_NAME
fi

if [[ ${#WANDB_NAME} -gt 116 ]]; then
	# We compare against 116 because WANDB_NAME gets appended -n$NODE_COUNT-j$JOB_ID in the final sbatch.
	>&2 echo "WANDB_NAME is too long (it shouldn't exceed 116 characters): $WANDB_NAME"
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
	--swiglu
)
if [[ $UNTIE = true ]]; then
	LLAMA_ARGS+=(--untie-embeddings-and-output-weights)
fi

TRAINING_ARGS=(
	--micro-batch-size $MBS
	--global-batch-size $GBS
	--train-iters $ITERS
	--eval-interval $((ITERS + 1))
	--eval-iters 1
	--weight-decay $WEIGHT_DECAY
	--adam-beta1 $BETA1
	--muon-momentum $BETA1
	--adam-beta2 $BETA2
	--ademamix-beta3 $BETA3
	--ademamix-alpha $ALPHA
	--ademamix-beta3-warmup $ITERS
	--ademamix-alpha-warmup $ITERS
	--init-method-std $INIT_STD
	--clip-grad 1.0
	--lr $LR
	--min-lr $MIN_LR
	--trigger-path $TRIGGER_DIR
	--exit-signal-handler
)

DATA_ARGS=(
	--data-path $DATA_PATHS
	--data-cache-path $SCRATCH/data/cache_opt
	--split 100,0,0
)
	#--mock-data

LOGGING=(
	--log-interval 1
	--save-interval $SAVE_FREQ
	--save $SAVE_PATH
	--load $SAVE_PATH
	--tensorboard-dir $ROOT_PATH/tensorboard
	--wandb-project $WANDB_PROJECT
	--wandb-save-dir $ROOT_PATH/wandb
	--timing-log-level 1
	--tensorboard-log-interval 1
	--log-throughput
	--log-timers-to-tensorboard
)
LOGGING=(${LOGGING[@]} ${EXTRA_LOGS[@]})

SCHEDULER_ARGS=(
	--lr-decay-style WSD
	--lr-wsd-decay-style minus_sqrt
	--lr-wsd-decay-iters $DECAY_ITERS
	--lr-warmup-iters $WARMUP
)

EXTRA_ARGS+=(
	--async-save
)

ARGS="${LLAMA_ARGS[@]} ${TRAINING_ARGS[@]} ${SCHEDULER_ARGS[@]} ${DATA_ARGS[@]} ${LOGGING[@]} ${EXTRA_ARGS[@]} ${FP8_ARGS[@]} ${ARCH_ARGS[@]} ${OPT_ARGS[@]}"

#= RUNNING: Prepare and launch a slurm script =#
CMD="python3 pretrain_gpt.py $ARGS"

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
#SBATCH --account=a-infra01-1
#SBATCH --partition=${PARTITION:-normal}
#SBATCH --signal=SIGUSR2@180
#SBATCH --dependency=singleton

# Wake up.
echo [\$(date)] Starting job
echo [\$(date)] Using nodes: \$SLURM_JOB_NODELIST
srun -l bash -c 'echo \$(hostname) \$(nvidia-smi | grep -o "|\\s*[0-9]*MiB")'


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

srun -lu --cpus-per-task \$SLURM_CPUS_PER_TASK --mpi=pmix --environment=$CONTAINER bash -c "
	cd $CODE_PATH
	export PYTHONPATH=\$PWD:$EMERGING_OPTIMIZERS_PATH
	export RANK=\\\$SLURM_PROCID
	export LOCAL_RANK=\\\$SLURM_LOCALID
	mkdir -p \\\$TRITON_CACHE_DIR
	$CMD --wandb-exp-name $EXP_NAME-n$NODES-j\\\$SLURM_JOBID
"

# Goodbye lol.
echo [\$(date)] Goodbye
EOM
echo "Saved sbatch to $ROOT_PATH/submission.sbatch"

sbatch $ROOT_PATH/submission.sbatch
