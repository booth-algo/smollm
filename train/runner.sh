#!/bin/bash

# General-purpose script to find a free GPU and run any Python script on it
# Usage: ./run_smol.sh <python_script.py> [additional args...]
# Example: ./run_smol.sh test_smol.py
# Example: ./run_smol.sh train_vqa_qlora.py --epochs 3

set -e  # Exit on error

# Check if Python script argument is provided
if [ $# -eq 0 ]; then
    echo "Error: No Python script specified"
    echo "Usage: $0 <python_script.py> [additional args...]"
    echo ""
    echo "Examples:"
    echo "  $0 test_smol.py"
    echo "  $0 train_vqa_qlora.py"
    echo "  $0 test_vqa.py ./smolvlm2-256m-vqa-final"
    exit 1
fi

PYTHON_SCRIPT=$1
shift  # Remove script name from arguments

# Validate Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script '$PYTHON_SCRIPT' not found"
    exit 1
fi

echo "Running: $PYTHON_SCRIPT"
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

        # Find GPU with lowest memory usage (prefer memory percentage over GPU utilization)
        if [ "$mem_usage" -lt "$min_usage" ]; then
            min_usage=$mem_usage
            free_gpu=$gpu_id
        fi
    done <<< "$gpu_info"

    echo "----------------------------------------" >&2

    if [ "$free_gpu" -eq -1 ]; then
        echo "Error: No GPU found" >&2
        exit 1
    fi

    echo "Selected GPU $free_gpu (lowest memory usage: ${min_usage}%)" >&2
    echo "$free_gpu"
}

# Find the free GPU
GPU_ID=$(find_free_gpu)

# Set environment variable to use only the selected GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo ""
echo "Starting '$PYTHON_SCRIPT' on GPU $GPU_ID..."
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "----------------------------------------"
echo ""

# Determine log file name based on script name
SCRIPT_NAME=$(basename "$PYTHON_SCRIPT" .py)
LOG_FILE="${SCRIPT_NAME}.log"

echo "📝 Logging output to: $LOG_FILE"
echo ""

# Run the Python script with remaining arguments and tee output to log file
python "$PYTHON_SCRIPT" "$@" 2>&1 | tee "$LOG_FILE"

# Capture exit code
EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Script completed successfully. Log saved to: $LOG_FILE"
else
    echo ""
    echo "❌ Script failed with exit code $EXIT_CODE. Log saved to: $LOG_FILE"
    exit $EXIT_CODE
fi
