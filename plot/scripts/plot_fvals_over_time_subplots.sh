#!/bin/bash

set -euo pipefail

# Wrapper to create three fvals-over-time subplot figures:
# 1) Synthetic benchmarks
# 2) FCNet benchmarks
# 3) Real-world benchmarks

SCRIPT=./plot/plot_fvals_over_time_subplots.py

### 1) Synthetic benchmarks
whitelist="BORE,BOTorch,CQR,LLM,LLMKD,REA,RS,RS + KD-Tree"
python ${SCRIPT} \
  --benchmarks hartmann3 hartmann6 levy rosenbrock rastrigin \
  --titles "Hartmann3" "Hartmann6" "Levy" "Rosenbrock" "Rastrigin" \
  --trials 50 \
  --filename "synthetic_fvals" \
  --data_paths ./results/hartmann3/ ./results/hartmann6/ ./results/levy/ ./results/rosenbrock/ ./results/rastrigin/ \
  --whitelists "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist"


### 2) FCNet benchmarks
python ${SCRIPT} \
  --benchmarks fcnet-naval fcnet-parkinsons fcnet-protein fcnet-slice \
  --titles "Naval" "Parkinsons" "Protein" "Slice" \
  --trials 50 \
  --filename "fcnet_fvals" \
  --data_paths ./results/fcnet-naval/ ./results/fcnet-parkinsons/ ./results/fcnet-protein/ ./results/fcnet-slice/ \
  --whitelists "$whitelist" "$whitelist" "$whitelist" "$whitelist"


### 3) Real-world benchmarks
python ${SCRIPT} \
  --benchmarks Penicillin CarSideImpact VehicleSafety \
  --titles "Penicillin" "CarSideImpact" "VehicleSafety" \
  --trials 50 \
  --filename "real_world_fvals" \
  --data_paths ./results/penicillin_single_objective/ ./results/car_side_impact_single_objective/ ./results/vehicle_safety_single_objective/ \
  --whitelists "$whitelist" "$whitelist" "$whitelist"

# Singe plots
python ./plot/plot_fvals_paper.py \
    --benchmark "vehicle_safety" \
    --title "Vehicle Safety" \
    --columns "F1" \
    --trials 50 \
    --data_path ./results/vehicle_safety_single_objective/ \
    --filename "results/vehicle_safety_single_objective_llm_comparision" \
    --normalization_method "none" \
    --whitelist "LLM,LLMKD,LLM-llama3.1-8b-llm,LLMKD-llama3.1-8b,LLM-llama3.3-70b-llm,LLMKD-llama3.3-70b,LLMKD-qwen-30b,LLM-qwen-30b" \
    --use_log_scale "False"

echo "All plots generated."

