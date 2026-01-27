#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
benchmarks=("fcnet-naval" "fcnet-parkinsons" "fcnet-protein" "fcnet-slice")
model="gemini-2.0-flash"

# Set the number of benchmarks to run in parallel
# Since there are 4 benchmarks, we can set this to 4.
NUM_PARALLEL=4

# --- Script Body ---

echo "## Running the vanilla method (LLM) for all benchmarks in parallel..."
# We pipe the list of benchmarks to xargs.
# -P $NUM_PARALLEL runs up to NUM_PARALLEL processes at once.
# -I{} replaces {} with the benchmark name for each command.
printf "%s\n" "${benchmarks[@]}" | xargs -I{} -P $NUM_PARALLEL python experiments/benchmark_main.py \
    --max_num_evaluations 50 \
    --seed 10 \
    --run_all_seeds 1 \
    --method LLMKD \
    --method_name "LLM" \
    --benchmark "{}" \
    --model "$model" \
    --optimization_method "mohollm" \
    --candidates_per_request 20 \
    --n_workers 1

echo "## Running the LLMKD method for all benchmarks in parallel..."
printf "%s\n" "${benchmarks[@]}" | xargs -I{} -P $NUM_PARALLEL python experiments/benchmark_main.py \
    --max_num_evaluations 50 \
    --seed 10 \
    --run_all_seeds 1 \
    --method LLMKD \
    --method_name "LLMKD" \
    --benchmark "{}" \
    --model "$model" \
    --optimization_method "SpacePartitioning" \
    --m0 0.5 \
    --lam 0 \
    --candidates_per_request 5 \
    --partitions_per_trial 5 \
    --alpha_max 1.0 \
    --n_workers 1

echo "## Running baseline methods..."
methods=("RS" "BORE" "TPE" "CQR" "BOTorch" "REA")

# Loop through each method sequentially
for method in "${methods[@]}"; do
    echo "=> Running method '$method' for all benchmarks in parallel..."
    # For each method, we parallelize the runs across all benchmarks.
    # We use 'bash -c' to correctly pass the '$method' variable into the xargs command.
    # '$0' inside the command string refers to the first argument after the string, which is '$method'.
    printf "%s\n" "${benchmarks[@]}" | xargs -I{} -P $NUM_PARALLEL bash -c '
        python experiments/benchmark_main.py \
            --max_num_evaluations 50 \
            --seed 10 \
            --run_all_seeds 1 \
            --method "$0" \
            --benchmark "{}"
    ' "$method"
done

echo "All experiments completed successfully."