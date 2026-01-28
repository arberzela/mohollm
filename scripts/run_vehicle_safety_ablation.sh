
model="gemini-2.0-flash"
# Fail fast and make pipelines safer
set -euo pipefail

# If the script receives INT/TERM, kill all child processes (so xargs children stop)
trap 'echo "Interrupted, killing children..."; kill 0' INT TERM

PARALLEL=1

benchmarks=("vehicle_safety")
# seeds=(0 1 2 3 4 5 6 7 8 9)
seeds=(9)

# Run the LLMKD method for each benchmark using xargs
{
    for seed in "${seeds[@]}"; do
        for benchmark in "${benchmarks[@]}"; do
            printf "%s %s\n" "$seed" "$benchmark"
        done
    done
} | xargs -n2 -P "$PARALLEL" bash -c '
    seed="$0"
    benchmark="$1"
    python experiments/benchmark_main_synthetic.py \
        --max_num_evaluations 50 \
        --seed "$seed" \
        --run_all_seeds 0 \
        --method LLMKD \
        --method_name "LLMKD-uniform_region_sampling" \
        --benchmark "$benchmark" \
        --model '"$model"' \
        --optimization_method "SpacePartitioning" \
        --m0 0.5 \
        --lam 0 \
        --candidates_per_request 5 \
        --partitions_per_trial 5 \
        --alpha_max 1.0 \
        --n_workers 1
'
