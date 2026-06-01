# shellcheck shell=bash
#
# normuon-wsd — normuon on the WSD (warmup-stable-decay) schedule instead of the
# framework's linear default. Cooldown shape/length and warmup fall through to
# common.sh's WSD defaults (minus_sqrt, ~20% of the run, warmup 0); override
# LR_WSD_DECAY_SAMPLES / LR_WARMUP_SAMPLES from the env to retune.

source "$(dirname "${BASH_SOURCE[0]}")/../normuon.sh"
EXP_TAG=normuon-wsd
LR_DECAY_STYLE=WSD
