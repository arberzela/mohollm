benchmarks_synth=("levy")
benchmarks_fcnet=("fcnet-naval" "fcnet-parkinsons" "fcnet-protein" "fcnet-slice")
model="gemini-2.0-flash"


# Ablations

# 1. leaf size. We use fixed leaf sizes here.
leaf_sizes=(1 2.5 5 10) # 1, 0.25 * d, 0.5 * d, d for levy

# Run the LLMKD method for each benchmark
for benchmark in "${benchmarks_synth[@]}"; do
    for leaf_size in "${leaf_sizes[@]}"; do
        python experiments/benchmark_main_synthetic.py \
            --max_num_evaluations 50 \
            --seed 10 \
            --run_all_seeds 1 \
            --method LLMKD \
            --method_name "LLMKD-m0-${leaf_size}" \
            --benchmark "$benchmark" \
            --model $model \
            --optimization_method "SpacePartitioning" \
            --m0 $leaf_size \
            --lam 0 \
            --candidates_per_request 5 \
            --partitions_per_trial 5 \
            --alpha_max 1.0 \
            --use_dimension_scaling "false" \
            --n_workers 1
    done
done



# 2. Candidates per region
candidates_per_region_settings=(1 3 5 7 10)
for benchmark in "${benchmarks_synth[@]}"; do
    for candidates_per_region in "${candidates_per_region_settings[@]}"; do
        python experiments/benchmark_main_synthetic.py \
            --max_num_evaluations 50 \
            --seed 10 \
            --run_all_seeds 1 \
            --method LLMKD \
            --method_name "LLMKD-k-${candidates_per_region}" \
            --benchmark "$benchmark" \partitioning
            --model $model \
            --optimization_method "SpacePartitioning" \
            --m0 0.5 \
            --lam 0 \
            --candidates_per_request $candidates_per_region \
            --partitions_per_trial 5 \
            --alpha_max 1.0 \
            --n_workers 1
    done
done



# 3. Partitions per trial
partitions_per_trial_settings=(1 3 5 7)
for benchmark in "${benchmarks_synth[@]}"; do
    for partitions_per_trial in "${partitions_per_trial_settings[@]}"; do
        python experiments/benchmark_main_synthetic.py \
            --max_num_evaluations 50 \
            --seed 10 \
            --run_all_seeds 1 \
            --method LLMKD \
            --method_name "LLMKD-${partitions_per_trial}-partitions-per-trial" \
            --benchmark "$benchmark" \
            --model $model \
            --optimization_method "SpacePartitioning" \
            --m0 0.5 \
            --lam 0 \
            --candidates_per_request 5 \
            --partitions_per_trial $partitions_per_trial \
            --alpha_max 1.0 \
            --n_workers 1
    done
done


# 4. Alpha max
alpha_max_settings=(0.2 0.5 0.7 1.0)

# Run the LLMKD method for each benchmark
for benchmark in "${benchmarks_fcnet[@]}"; do
    for alpha_max in "${alpha_max_settings[@]}"; do
        python experiments/benchmark_main.py \
            --max_num_evaluations 50 \
            --seed 10 \
            --run_all_seeds 1 \
            --method LLMKD \
            --method_name "LLMKD-alpha-${alpha_max}" \
            --benchmark "$benchmark" \
            --model $model \
            --optimization_method "SpacePartitioning" \
            --m0 0.5 \
            --lam 0 \
            --candidates_per_request 5 \
            --partitions_per_trial 5 \
            --alpha_max $alpha_max \
            --n_workers 1
    done
done