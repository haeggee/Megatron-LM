# shellcheck shell=bash
#
# adamw — plain AdamW baseline. The anchor every other recipe is compared to.

OPTIMIZER=adam
EXP_TAG=adamw

LR=${LR:-3e-4}
MIN_LR=${MIN_LR:-1e-5}
KNOB_STR=lr${LR}

WEIGHT_DECAY=0.1
ADAM_BETA1=0.9
ADAM_BETA2=0.95
CLIP_GRAD=1.0

LR_DECAY_STYLE=WSD
LR_WARMUP_SAMPLES=0
LR_WSD_DECAY_STYLE=minus_sqrt
LR_WSD_DECAY_SAMPLES=732422

# Plain AdamW needs no optimizer-specific flags.
RECIPE_ARGS=()
