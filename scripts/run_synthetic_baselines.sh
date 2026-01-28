
# Fail fast and make pipelines safer
set -euo pipefail

# If the script receives INT/TERM, kill all child processes (so xargs children stop)
trap 'echo "Interrupted, killing children..."; kill 0' INT TERM

# Number of parallel jobs for xargs (can be overridden by env var PARALLEL)
PARALLEL=4


benchmarks=("rastrigin")
methods=("REA")
seeds=(2)

# Parallelize baseline runs over (method, seed, benchmark) triplets and dispatch with xargs
{
    for method in "${methods[@]}"; do
        for seed in "${seeds[@]}"; do
            for benchmark in "${benchmarks[@]}"; do
                printf "%s %s %s\n" "$method" "$seed" "$benchmark"
            done
        done
    done
} | xargs -n3 -P "$PARALLEL" bash -c '
    method="$0"
    seed="$1"
    benchmark="$2"
    python experiments/benchmark_main_synthetic.py \
        --max_num_evaluations 50 \
        --seed "$seed" \
        --run_all_seeds 0 \
        --method "$method" \
        --benchmark "$benchmark"
'
