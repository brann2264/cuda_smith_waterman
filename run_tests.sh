#!/bin/bash

# Create necessary directories
mkdir -p inputs
mkdir -p outputs

# Define your compiled executables (change these if your binaries are named differently)
CPU_EXEC="./sw_gold/sw_gold"
GPU_EXEC="./bin/sw"

# Define the dataset sizes you plan to test
SIZES=("1k" "10k" "100k" "1m" "5m" "10m")

echo "=========================================="
echo " Starting Smith-Waterman Benchmark Suite "
echo "=========================================="

# Loop through each size and run both CPU and GPU algorithms
for size in "${SIZES[@]}"; do
    INPUT_FILE="inputs/data_${size}.txt"
    
    # Check if the input file actually exists before running
    if [ ! -f "$INPUT_FILE" ]; then
        echo "[Warning] Input file $INPUT_FILE not found. Skipping $size."
        continue
    fi

    echo "------------------------------------------"
    echo " Running tests for size: $size"
    echo "------------------------------------------"
    
    # --- 1. Run CPU (Gold) Version ---
    # Using the flag-based argument structure from your SequentialImplementation
    CPU_OUT="outputs/sw_gold_${size}.txt"
    echo "-> Executing CPU version..."
    $CPU_EXEC -in "$INPUT_FILE" -out "$CPU_OUT" -match 2 -mismatch -1 -gap -1
    
    # --- 2. Run GPU (CUDA) Version ---
    # Using the positional argument structure from your CUDA wrapper
    GPU_OUT="outputs/sw_cuda_${size}.txt"
    echo "-> Executing GPU version..."
    $GPU_EXEC "$INPUT_FILE" "$GPU_OUT"
    
    echo "✓ Finished $size"
done

echo "=========================================="
echo " All tests complete!"
echo " Results are saved in the 'outputs/' directory."
echo "=========================================="