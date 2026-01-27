
model="gemini-2.0-flash"
# Fail fast and make pipelines safer
set -euo pipefail

# If the script receives INT/TERM, kill all child processes (so xargs children stop)
trap 'echo "Interrupted, killing children..."; kill 0' INT TERM

PARALLEL=4

benchmarks=("vehicle_safety_noise")
seeds=(0 1 2 3 4 5 6 7 8 9)
# seeds=(5)

# Run the LLMKD method for each benchmark using xargs
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
        --method_name "LLM" \
        --benchmark "$benchmark" \
        --model '"$model"' \
        --optimization_method "mohollm" \
        --candidates_per_request 20 \
        --n_workers 1
'

# methods=("RS" "BORE" "TPE" "CQR" "BOTorch" "REA")
# for method in "${methods[@]}"; do
#     # Run the baseline methods for each benchmark
#     for benchmark in "${benchmarks[@]}"; do
#         python experiments/benchmark_main.py \
#             --max_num_evaluations 50 \
#             --seed 10 \
#             --run_all_seeds 1 \
#             --method $method \
#             --benchmark "$benchmark"
#     done
# done
