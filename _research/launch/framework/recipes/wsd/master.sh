# shellcheck shell=bash
#
# master-wsd — master on the WSD (warmup-stable-decay) schedule. master.sh
# itself now defaults to linear; this flips it to WSD. Cooldown shape/length and
# warmup fall through to common.sh's WSD defaults (minus_sqrt, ~20% of the run,
# warmup 0); override LR_WSD_DECAY_SAMPLES / LR_WARMUP_SAMPLES from the env.

source "$(dirname "${BASH_SOURCE[0]}")/../master.sh"
EXP_TAG=master-wsd
LR_DECAY_STYLE=WSD
