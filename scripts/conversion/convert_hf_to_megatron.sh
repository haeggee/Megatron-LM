#!/bin/bash

# ============================================================================
# HuggingFace to Megatron Conversion Script
# ============================================================================
# This script converts HuggingFace safetensors checkpoints to Megatron format.
# Supports two-stage conversion: HF → torch → torch_dist (optional)
#
# Usage:
#   1. Edit the configuration section below
#   2. Run: bash scripts/conversion/convert_hf_to_megatron.sh
# ============================================================================

# ============================================================================
# CONFIGURATION SECTION - Edit these variables
# ============================================================================

# Input/Output Paths
HF_CHECKPOINT_PATH="/users/rkreft/scratch/apertus8b/hf_apertus_base"           # REQUIRED: Path to HuggingFace checkpoint directory
TORCH_OUTPUT_PATH="/capstor/store/cscs/swissai/infra01/MLLM/apertus-8b/base_models/Apertus8B-tokens15T-it2627139_tp2/torch"             # REQUIRED: Path to save Megatron torch checkpoint
TORCH_DIST_OUTPUT_PATH="/capstor/store/cscs/swissai/infra01/MLLM/apertus-8b/base_models/Apertus8B-tokens15T-it2627139_tp2/torch_dist"        # Optional: Path to save Megatron torch_dist checkpoint

# Model Configuration
MODEL_SIZE="llama3"              # Options: llama2-7B, llama2-13B, llama2-70B, llama3, mistral, yi-34B, qwen2.5
MODEL_TYPE="GPT"                 # Options: GPT, BERT

# Parallelism Settings (for target Megatron checkpoint)
TARGET_TENSOR_PARALLEL_SIZE=2    # default for 8b Apertus is 2
TARGET_PIPELINE_PARALLEL_SIZE=1  # Pipeline model parallel size
TARGET_EXPERT_PARALLEL_SIZE=1    # Expert model parallel size (for MoE models)

PRECISION="bf16"

# Path to tokenizer - if not set, will set eq to hf checkpoint path
TOKENIZER_PATH=""

# Optional Features
CONVERT_TO_TORCH_DIST=true      # Set to true to enable Stage 2: torch → torch_dist
TEST_LOGITS=false                # Validate conversion (only works with TP=1, PP=1)
MAKE_VOCAB_SIZE_DIVISIBLE_BY=128 # Correct value for llama and Apertus models
TRANSFORMER_IMPL="transformer_engine"  # Options: transformer_engine, local

# Advanced Settings
MEGATRON_LM_DIR=""               # Leave empty to auto-detect, or set custom path
MAX_QUEUE_SIZE=50                # Memory buffer size for conversion
LOADER_TRANSFORMER_IMPL="local"  # Loader transformer implementation

# ============================================================================
# SCRIPT LOGIC - Do not edit below unless you know what you're doing
# ============================================================================

set -euo pipefail  # Exit on error, undefined variables, and pipe failures

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Auto-detect MEGATRON_LM_DIR if not set
if [ -z "${MEGATRON_LM_DIR}" ]; then
    MEGATRON_LM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    echo -e "${BLUE}INFO: Auto-detected MEGATRON_LM_DIR=${MEGATRON_LM_DIR}${NC}"
fi

# Set TOKENIZER_PATH to HF_CHECKPOINT_PATH if not specified
if [ -z "${TOKENIZER_PATH}" ]; then
    TOKENIZER_PATH="${HF_CHECKPOINT_PATH}"
    echo -e "${BLUE}INFO: Using HF_CHECKPOINT_PATH as TOKENIZER_PATH${NC}"
fi

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo -e "\n${BLUE}===============================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===============================================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ WARNING: $1${NC}"
}

print_error() {
    echo -e "${RED}✗ ERROR: $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ INFO: $1${NC}"
}

# ============================================================================
# Validation Functions
# ============================================================================

validate_prerequisites() {
    print_header "Validating Prerequisites"

    local has_error=0

    # Check CUDA_DEVICE_MAX_CONNECTIONS
    if [ -z "${CUDA_DEVICE_MAX_CONNECTIONS:-}" ]; then
        print_warning "CUDA_DEVICE_MAX_CONNECTIONS not set, exporting CUDA_DEVICE_MAX_CONNECTIONS=1"
        export CUDA_DEVICE_MAX_CONNECTIONS=1
    else
        print_success "CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS}"
    fi

    # Set PYTHONPATH
    export PYTHONPATH=${MEGATRON_LM_DIR}:${PYTHONPATH:-}
    print_success "PYTHONPATH includes ${MEGATRON_LM_DIR}"

    if [ $has_error -eq 1 ]; then
        print_error "Prerequisite validation failed"
        exit 2
    fi

    print_success "All prerequisites validated"
}

validate_configuration() {
    print_header "Validating Configuration"

    local has_error=0

    # Check required paths
    if [ -z "${HF_CHECKPOINT_PATH}" ]; then
        print_error "HF_CHECKPOINT_PATH is not set"
        has_error=1
    elif [ ! -d "${HF_CHECKPOINT_PATH}" ]; then
        print_error "HF_CHECKPOINT_PATH does not exist: ${HF_CHECKPOINT_PATH}"
        has_error=1
    else
        print_success "HF_CHECKPOINT_PATH exists: ${HF_CHECKPOINT_PATH}"
    fi

    # Check HF checkpoint has config.json
    if [ -n "${HF_CHECKPOINT_PATH}" ] && [ -d "${HF_CHECKPOINT_PATH}" ]; then
        if [ ! -f "${HF_CHECKPOINT_PATH}/config.json" ]; then
            print_warning "config.json not found in HF_CHECKPOINT_PATH"
        else
            print_success "config.json found in HF checkpoint"
        fi
    fi

    if [ -z "${TORCH_OUTPUT_PATH}" ]; then
        print_error "TORCH_OUTPUT_PATH is not set"
        has_error=1
    else
        print_success "TORCH_OUTPUT_PATH set: ${TORCH_OUTPUT_PATH}"
    fi

    # Check torch_dist path if conversion is enabled
    if [ "$CONVERT_TO_TORCH_DIST" = true ]; then
        if [ -z "${TORCH_DIST_OUTPUT_PATH}" ]; then
            print_error "TORCH_DIST_OUTPUT_PATH is not set but CONVERT_TO_TORCH_DIST=true"
            has_error=1
        else
            print_success "TORCH_DIST_OUTPUT_PATH set: ${TORCH_DIST_OUTPUT_PATH}"
        fi
    fi

    # Validate TEST_LOGITS requirements
    if [ "$TEST_LOGITS" = true ]; then
        if [ ${TARGET_TENSOR_PARALLEL_SIZE} -ne 1 ] || [ ${TARGET_PIPELINE_PARALLEL_SIZE} -ne 1 ]; then
            print_error "TEST_LOGITS requires TP=1 and PP=1"
            has_error=1
        else
            print_success "TEST_LOGITS enabled (TP=1, PP=1)"
        fi
    fi

    if [ $has_error -eq 1 ]; then
        print_error "Configuration validation failed"
        exit 1
    fi

    print_success "All configuration validated"
}

print_configuration() {
    print_header "Configuration Summary"

    echo "Input/Output:"
    echo "  HF_CHECKPOINT_PATH:      ${HF_CHECKPOINT_PATH}"
    echo "  TORCH_OUTPUT_PATH:       ${TORCH_OUTPUT_PATH}"
    if [ "$CONVERT_TO_TORCH_DIST" = true ]; then
        echo "  TORCH_DIST_OUTPUT_PATH:  ${TORCH_DIST_OUTPUT_PATH}"
    fi
    echo ""
    echo "Model Configuration:"
    echo "  MODEL_SIZE:              ${MODEL_SIZE}"
    echo "  MODEL_TYPE:              ${MODEL_TYPE}"
    echo "  PRECISION:               ${PRECISION}"
    echo ""
    echo "Parallelism:"
    echo "  TARGET_TENSOR_PARALLEL_SIZE:    ${TARGET_TENSOR_PARALLEL_SIZE}"
    echo "  TARGET_PIPELINE_PARALLEL_SIZE:  ${TARGET_PIPELINE_PARALLEL_SIZE}"
    echo "  TARGET_EXPERT_PARALLEL_SIZE:    ${TARGET_EXPERT_PARALLEL_SIZE}"
    echo ""
    echo "Options:"
    echo "  CONVERT_TO_TORCH_DIST:   ${CONVERT_TO_TORCH_DIST}"
    echo "  TEST_LOGITS:             ${TEST_LOGITS}"
    echo "  TRANSFORMER_IMPL:        ${TRANSFORMER_IMPL}"
    echo "  VOCAB_DIVISIBLE_BY:      ${MAKE_VOCAB_SIZE_DIVISIBLE_BY}"
    echo ""
}

# ============================================================================
# Conversion Stage Functions
# ============================================================================

stage1_hf_to_torch() {
    print_header "Stage 1: HuggingFace → Megatron (torch format)"

    # Build command
    local cmd=(
        python "${MEGATRON_LM_DIR}/tools/checkpoint/convert.py"
        --model-type "${MODEL_TYPE}"
        --loader llama_mistral
        --saver core
        --checkpoint-type hf
        --model-size "${MODEL_SIZE}"
        --load-dir "${HF_CHECKPOINT_PATH}"
        --save-dir "${TORCH_OUTPUT_PATH}"
        --tokenizer-model "${TOKENIZER_PATH}"
        --target-tensor-parallel-size "${TARGET_TENSOR_PARALLEL_SIZE}"
        --target-pipeline-parallel-size "${TARGET_PIPELINE_PARALLEL_SIZE}"
        --target-expert-parallel-size "${TARGET_EXPERT_PARALLEL_SIZE}"
        --"${PRECISION}"
        --make-vocab-size-divisible-by "${MAKE_VOCAB_SIZE_DIVISIBLE_BY}"
        --saver-transformer-impl "${TRANSFORMER_IMPL}"
        --loader-transformer-impl "${LOADER_TRANSFORMER_IMPL}"
        --max-queue-size "${MAX_QUEUE_SIZE}"
    )

    # Add optional flags
    if [ "$TEST_LOGITS" = true ]; then
        cmd+=(--test-logits)
    fi

    # Print command
    print_info "Running command:"
    echo -e "${BLUE}${cmd[*]}${NC}\n"

    # Execute
    if "${cmd[@]}"; then
        print_success "Stage 1 completed successfully"
    else
        print_error "Stage 1 failed"
        exit 3
    fi
}

stage2_torch_to_torchdist() {
    print_header "Stage 2: Megatron (torch) → Megatron (torch_dist format)"

    # Calculate number of processes needed
    local nproc=$((TARGET_TENSOR_PARALLEL_SIZE * TARGET_PIPELINE_PARALLEL_SIZE))

    print_info "Using ${nproc} processes (TP=${TARGET_TENSOR_PARALLEL_SIZE} × PP=${TARGET_PIPELINE_PARALLEL_SIZE})"

    # Build command
    local cmd=(
        CUDA_DEVICE_MAX_CONNECTIONS=1
        torchrun
    )

    # Add nproc_per_node only if > 1
    if [ ${nproc} -gt 1 ]; then
        cmd+=(--nproc-per-node="${nproc}")
    fi

    cmd+=(
        "${MEGATRON_LM_DIR}/scripts/conversion/torch_2_torchdist.py"
        --"${PRECISION}"
        --load "${TORCH_OUTPUT_PATH}"
        --ckpt-convert-save "${TORCH_DIST_OUTPUT_PATH}"
    )

    # Print command
    print_info "Running command:"
    echo -e "${BLUE}${cmd[*]}${NC}\n"

    # Execute
    if "${cmd[@]}"; then
        print_success "Stage 2 completed successfully"
    else
        print_error "Stage 2 failed"
        exit 3
    fi
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    print_header "HuggingFace to Megatron Conversion"

    # Print configuration
    print_configuration

    # Validate
    validate_prerequisites
    validate_configuration

    # Stage 1: HF → torch (always runs)
    stage1_hf_to_torch

    # Stage 2: torch → torch_dist (optional)
    if [ "$CONVERT_TO_TORCH_DIST" = true ]; then
        stage2_torch_to_torchdist
    else
        print_info "Skipping Stage 2 (CONVERT_TO_TORCH_DIST=false)"
    fi

    # Success!
    print_header "Conversion Complete!"
    echo -e "${GREEN}✓ Torch checkpoint saved to: ${TORCH_OUTPUT_PATH}${NC}"
    if [ "$CONVERT_TO_TORCH_DIST" = true ]; then
        echo -e "${GREEN}✓ Torch_dist checkpoint saved to: ${TORCH_DIST_OUTPUT_PATH}${NC}"
    fi
    echo ""

    # Next steps
    echo -e "${BLUE}Next steps:${NC}"
    echo "  1. Verify checkpoint structure:"
    echo "     ls -la ${TORCH_OUTPUT_PATH}/"
    if [ "$CONVERT_TO_TORCH_DIST" = true ]; then
        echo "     ls -la ${TORCH_DIST_OUTPUT_PATH}/"
    fi
    echo ""
    echo "  2. Resume training with:"
    echo "     torchrun pretrain_gpt.py --load ${TORCH_OUTPUT_PATH} --use-checkpoint-args ..."
    echo ""
}

# Run main function
main
