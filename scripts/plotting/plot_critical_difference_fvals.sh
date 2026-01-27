#!/bin/bash

python ./plot/plot_critical_difference_diagram_fvals.py \
    --benchmarks car_side_impact_single_objective,fcnet-naval,fcnet-parkinsons,fcnet-protein,fcnet-slice,hartmann3,hartmann6,levy,penicillin_single_objective,rastrigin,rosenbrock,vehicle_safety_single_objective \
    --title "Critical Difference" \
    --trials 50 \
    --whitelist "RS + KD-Tree,BORE,BOTorch,CQR,LLM,LLMKD,REA,RS,RS + KD-Tree" \
    --filename "critical_difference_diagram_fvals"

