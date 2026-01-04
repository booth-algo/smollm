#!/bin/bash

# Script to find a free GPU and run QLoRA train or test script on it
# Usage: ./run_qlora.sh [train|test] [additional args...]

set -e  # Exit on error

# Parse mode argument
MODE=${1:-train}  # Default to 'train' if no argument provided

if [[ "$MODE" != "train" && "$MODE" != "test" ]]; then
    echo "Error: Invalid mode '$MODE'"
    echo "Usage: $0 [train|test] [additional args...]"
    echo ""
    echo "Examples:"
    echo "  $0 train          # Run QLoRA training script"
    echo "  $0 test           # Run QLoRA testing script"
    exit 1
fi

# Shift to remove the mode argument from $@
shift

echo "Mode: $MODE (QLoRA)"
echo "Checking available GPUs..."

# Get GPU information using nvidia-smi
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. Make sure NVIDIA drivers are installed."
    exit 1
fi

# Function to find the GPU with lowest memory usage
find_free_gpu() {
    # Get GPU memory usage for all GPUs
    # Format: GPU_ID MEMORY_USED MEMORY_TOTAL UTILIZATION
    gpu_info=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits)

    min_usage=100
    free_gpu=-1

    echo "GPU Status:" >&2
    echo "----------------------------------------" >&2

    while IFS=',' read -r gpu_id mem_used mem_total util; do
        # Trim whitespace
        gpu_id=$(echo "$gpu_id" | xargs)
        mem_used=$(echo "$mem_used" | xargs)
        mem_total=$(echo "$mem_total" | xargs)
        util=$(echo "$util" | xargs)

        # Calculate memory usage percentage
        mem_usage=$((mem_used * 100 / mem_total))

        echo "GPU $gpu_id: ${mem_used}MB/${mem_total}MB (${mem_usage}%) | Utilization: ${util}%" >&2

        # Find GPU with lowest utilization (prefer GPU utilization over memory)
        if [ "$util" -lt "$min_usage" ]; then
            min_usage=$util
            free_gpu=$gpu_id
        fi
    done <<< "$gpu_info"

    echo "----------------------------------------" >&2

    if [ "$free_gpu" -eq -1 ]; then
        echo "Error: No GPU found" >&2
        exit 1
    fi

    echo "Selected GPU $free_gpu (lowest utilization: ${min_usage}%)" >&2
    echo "$free_gpu"
}

# Force GPU 2
GPU_ID=2

# Set environment variable to use only the selected GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo ""
echo "Starting $MODE (QLoRA) on GPU $GPU_ID (FORCED)..."
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "----------------------------------------"
echo ""

# Run the appropriate script based on mode
if [ "$MODE" = "train" ]; then
    python train_vqa_qlora.py "$@"
elif [ "$MODE" = "test" ]; then
    python test_vqa_qlora.py "$@"
fi
