
benchmarks=("fcnet-naval" "fcnet-parkinsons" "fcnet-protein" "fcnet-slice")
model="gemini-2.0-flash"
seeds=(0 1 2 3 4 5 6 7 8 9)

# Fail fast and make pipelines safer
set -euo pipefail

# If the script receives INT/TERM, kill all child processes (so xargs children stop)
trap 'echo "Interrupted, killing children..."; kill 0' INT TERM

# Number of parallel jobs for xargs (can be overridden by env var PARALLEL)
PARALLEL=10

# # seeds to run

# # Run the vanilla method for each benchmark+seed using xargs to parallelize
# {
#     for seed in "${seeds[@]}"; do
#         for benchmark in "${benchmarks[@]}"; do
#             printf "%s %s\n" "$seed" "$benchmark"
#         done
#     done
# } | xargs -n2 -P "$PARALLEL" bash -c '
#     seed="$0"
#     benchmark="$1"
#     python experiments/benchmark_main_synthetic.py \
#         --max_num_evaluations 50 \
#         --seed "$seed" \
#         --run_all_seeds 0 \
#         --method LLMKD \
#         --method_name "LLM" \
#         --benchmark "$benchmark" \
#         --model '"$model"' \
#         --optimization_method "mohollm" \
#         --candidates_per_request 20 \
#         --n_workers 1
# '


# Run the LLMKD method for each benchmark+seed using xargs
{
    for seed in "${seeds[@]}"; do
        for benchmark in "${benchmarks[@]}"; do
            printf "%s %s\n" "$seed" "$benchmark"
        done
    done
} | xargs -n2 -P "$PARALLEL" bash -c '
    seed="$0"
    benchmark="$1"
    python experiments/benchmark_main.py \
        --max_num_evaluations 50 \
        --seed "$seed" \
        --run_all_seeds 0 \
        --method LLMKD \
        --method_name "RS + KD-Tree" \
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


# benchmarks=("vehicle_safety")
# model="gemini-2.0-flash"
# seeds=(0 1 2 3 4 5 6 7 8 9)

# {
#     for seed in "${seeds[@]}"; do
#         for benchmark in "${benchmarks[@]}"; do
#             printf "%s %s\n" "$seed" "$benchmark"
#         done
#     done
# } | xargs -n2 -P "$PARALLEL" bash -c '
#     seed="$0"
#     benchmark="$1"
#     python experiments/benchmark_main_synthetic.py \
#         --max_num_evaluations 50 \
#         --seed "$seed" \
#         --run_all_seeds 0 \
#         --method LLMKD \
#         --method_name "LLM" \
#         --benchmark "$benchmark" \
#         --model '"$model"' \
#         --optimization_method "mohollm" \
#         --candidates_per_request 20 \
#         --n_workers 1
# '
# # Run the LLMKD method for each benchmark+seed using xargs
# {
#     for seed in "${seeds[@]}"; do
#         for benchmark in "${benchmarks[@]}"; do
#             printf "%s %s\n" "$seed" "$benchmark"
#         done
#     done
# } | xargs -n2 -P "$PARALLEL" bash -c '
#     seed="$0"
#     benchmark="$1"
#     python experiments/benchmark_main_synthetic.py \
#         --max_num_evaluations 50 \
#         --seed "$seed" \
#         --run_all_seeds 0 \
#         --method LLMKD \
#         --method_name "LLMKD" \
#         --benchmark "$benchmark" \
#         --model '"$model"' \
#         --optimization_method "SpacePartitioning" \
#         --m0 0.5 \
#         --lam 0 \
#         --candidates_per_request 5 \
#         --partitions_per_trial 5 \
#         --alpha_max 1.0 \
#         --n_workers 1
# '