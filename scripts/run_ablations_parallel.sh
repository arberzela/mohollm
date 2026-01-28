#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
benchmarks_synth=("levy")
benchmarks_fcnet=("fcnet-naval" "fcnet-parkinsons" "fcnet-protein" "fcnet-slice")
model="gemini-2.0-flash"

# Set the number of ablation runs to execute in parallel
NUM_PARALLEL=8

# --- Ablation Studies ---

## 1. Leaf size (m0)
# --------------------------------------------------------------------------
echo "## Running ablation study 1/4: Leaf Size (m0)..."
leaf_sizes=(1 2.5 5 10)
# Since there's only one benchmark, we parallelize over the hyperparameter values.
printf "%s\n" "${leaf_sizes[@]}" | xargs -I{} -P $NUM_PARALLEL bash -c '
    python experiments/benchmark_main_synthetic.py \
        --max_num_evaluations 50 \
        --seed 5 \
        --run_all_seeds 1 \
        --method LLMKD \
        --method_name "LLMKD-m0-{}" \
        --benchmark "$0" \
        --model "$1" \
        --optimization_method "SpacePartitioning" \
        --m0 "{}" \
        --lam 0 \
        --candidates_per_request 5 \
        --partitions_per_trial 5 \
        --alpha_max 1.0 \
        --use_dimension_scaling "false" \
        --n_workers 1
' "${benchmarks_synth[0]}" "$model"

## 2. Candidates per region (k)
# --------------------------------------------------------------------------
echo "## Running ablation study 2/4: Candidates per Region (k)..."
candidates_per_region_settings=(1 3 5 7 10)
printf "%s\n" "${candidates_per_region_settings[@]}" | xargs -I{} -P $NUM_PARALLEL bash -c '
    python experiments/benchmark_main_synthetic.py \
        --max_num_evaluations 50 \
        --seed 5 \
        --run_all_seeds 1 \
        --method LLMKD \
        --method_name "LLMKD-k-{}" \
        --benchmark "$0" \
        --model "$1" \
        --optimization_method "SpacePartitioning" \
        --m0 0.5 \
        --lam 0 \
        --candidates_per_request "{}" \
        --partitions_per_trial 5 \
        --alpha_max 1.0 \
        --n_workers 1
' "${benchmarks_synth[0]}" "$model"

## 3. Partitions per trial
# --------------------------------------------------------------------------
echo "## Running ablation study 3/4: Partitions per Trial..."
partitions_per_trial_settings=(1 3 5 7)
printf "%s\n" "${partitions_per_trial_settings[@]}" | xargs -I{} -P $NUM_PARALLEL bash -c '
    python experiments/benchmark_main_synthetic.py \
        --max_num_evaluations 50 \
        --seed 5 \
        --run_all_seeds 1 \
        --method LLMKD \
        --method_name "LLMKD-{}-partitions-per-trial" \
        --benchmark "$0" \
        --model "$1" \
        --optimization_method "SpacePartitioning" \
        --m0 0.5 \
        --lam 0 \
        --candidates_per_request 5 \
        --partitions_per_trial "{}" \
        --alpha_max 1.0 \
        --n_workers 1
' "${benchmarks_synth[0]}" "$model"

## 4. Alpha max
# --------------------------------------------------------------------------
echo "## Running ablation study 4/4: Alpha Max..."
alpha_max_settings=(0.2 0.5 0.7 1.0)
# Here we have multiple benchmarks and settings, so we create all pairs.
# Then we pipe the pairs to xargs, which reads 2 arguments (-n 2) for each job.
( # Use a subshell to generate the pairs of (benchmark, alpha_max)
    for benchmark in "${benchmarks_fcnet[@]}"; do
        for alpha_max in "${alpha_max_settings[@]}"; do
            echo "$benchmark" "$alpha_max"
        done
    done
) | xargs -n 2 -P $NUM_PARALLEL bash -c '
    # Inside bash -c: $0 is the model, $1 is the benchmark, $2 is alpha_max
    python experiments/benchmark_main.py \
        --max_num_evaluations 50 \
        --seed 5 \
        --run_all_seeds 1 \
        --method LLMKD \
        --method_name "LLMKD-alpha-$2" \
        --benchmark "$1" \
        --model "$0" \
        --optimization_method "SpacePartitioning" \
        --m0 0.5 \
        --lam 0 \
        --candidates_per_request 5 \
        --partitions_per_trial 5 \
        --alpha_max "$2" \
        --n_workers 1
' "$model" # Pass the model name to be used as $0 inside the command

echo "All ablation studies completed successfully."