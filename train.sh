#!/bin/bash
# run_training.sh - Run NL2SQL training with logging to terminal and file

# ============================================================================
# Configuration
# ============================================================================
SCRIPT_NAME="src/nl2sql/train/train_unsloth_response_only.py"
LOG_DIR="logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"

# ============================================================================
# Setup
# ============================================================================
# Create log directory if it doesn't exist
mkdir -p ${LOG_DIR}

echo "============================================================"
echo "NL2SQL Unsloth Training"
echo "============================================================"
echo "Start time: $(date)"
echo "Log file: ${LOG_FILE}"
echo "Script: ${SCRIPT_NAME}"
echo "============================================================"

# ============================================================================
# GPU Configuration (optional - uncomment as needed)
# ============================================================================
# Use specific GPU (if you have multiple)
# export CUDA_VISIBLE_DEVICES=0

# Show GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# ============================================================================
# W&B Configuration (optional)
# ============================================================================
# For offline mode (logs locally, sync later with `wandb sync`)
# export WANDB_MODE=offline

# To disable W&B completely
# export WANDB_DISABLED=true

# ============================================================================
# Run Training with Dual Logging
# ============================================================================
# Using 'tee' to write to both terminal and file
# '2>&1' redirects stderr to stdout so both are captured
# 'unbuffer' or 'stdbuf' ensures real-time output (not buffered)

# Option 1: Basic (may have buffering delays)
# python ${SCRIPT_NAME} 2>&1 | tee ${LOG_FILE}

# Option 2: Unbuffered output (recommended - requires 'expect' package)
# unbuffer python ${SCRIPT_NAME} 2>&1 | tee ${LOG_FILE}

# Option 3: Using stdbuf (usually pre-installed)
stdbuf -oL -eL python ${SCRIPT_NAME} 2>&1 | tee ${LOG_FILE}

# Capture exit code
EXIT_CODE=${PIPESTATUS[0]}

# ============================================================================
# Post-Training Summary
# ============================================================================
echo ""
echo "============================================================"
echo "Training Finished"
echo "============================================================"
echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "Log saved to: ${LOG_FILE}"
echo "============================================================"

# Also save a summary at end of log file
{
    echo ""
    echo "============================================================"
    echo "TRAINING COMPLETED"
    echo "Exit code: ${EXIT_CODE}"
    echo "End time: $(date)"
    echo "============================================================"
} >> ${LOG_FILE}

# Exit with the same code as python script
exit ${EXIT_CODE}