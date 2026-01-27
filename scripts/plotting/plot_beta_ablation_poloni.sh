#!/bin/bash

# Script to generate log HV difference plot for beta ablations on Poloni benchmark with subplots
# Usage: ./plot_beta_ablation_poloni.sh

# Configuration
DATA_PATH="./results/Poloni/"  # Adjust this to your actual results path
TRIALS=75  # Adjust as needed

# # Run the plot script for individual comparisons
# python plot/plot_beta_ablation_poloni.py \
#     --benchmarks "Poloni" \
#     --titles "Poloni Beta Ablation LLM" \
#     --data_paths "$DATA_PATH" \
#     --trials $TRIALS \
#     --filename "poloni_beta_LLM" \
#     --whitelists "mohollm (Gemini 2.0 Flash) (Beta=0.0),mohollm (Gemini 2.0 Flash) (Beta=0.25),mohollm (Gemini 2.0 Flash) (Beta=0.5),mohollm (Gemini 2.0 Flash) (Beta=0.75),mohollm (Gemini 2.0 Flash) (Beta=1.0)"

# python plot/plot_beta_ablation_poloni.py \
#     --benchmarks "Poloni" \
#     --titles "Poloni Beta Ablation MOHOLLM" \
#     --data_paths "$DATA_PATH" \
#     --trials $TRIALS \
#     --filename "poloni_beta_MOHOLLM" \
#     --whitelists "MOHOLLM (Gemini 2.0 Flash) (Beta=0.0),MOHOLLM (Gemini 2.0 Flash) (Beta=0.25),MOHOLLM (Gemini 2.0 Flash) (Beta=0.5),MOHOLLM (Gemini 2.0 Flash) (Beta=0.75),MOHOLLM (Gemini 2.0 Flash) (Beta=1.0)"

# python plot/plot_beta_ablation_poloni.py \
#     --benchmarks "Poloni" \
#     --titles "Poloni Beta Ablation Combined" \
#     --data_paths "$DATA_PATH" \
#     --trials $TRIALS \
#     --filename "poloni_beta_combined" \
#     --whitelists "MOHOLLM (Gemini 2.0 Flash) (Beta=0.0),MOHOLLM (Gemini 2.0 Flash) (Beta=0.25),MOHOLLM (Gemini 2.0 Flash) (Beta=0.5),MOHOLLM (Gemini 2.0 Flash) (Beta=0.75),MOHOLLM (Gemini 2.0 Flash) (Beta=1.0),mohollm (Gemini 2.0 Flash) (Beta=0.0),mohollm (Gemini 2.0 Flash) (Beta=0.25),mohollm (Gemini 2.0 Flash) (Beta=0.5),mohollm (Gemini 2.0 Flash) (Beta=0.75),mohollm (Gemini 2.0 Flash) (Beta=1.0)"

# Run the plot script for subplot comparison (each beta value: MOHOLLM vs mohollm)
python plot/plot_beta_ablation_poloni.py \
    --benchmarks "Poloni" "Poloni" "Poloni" "Poloni" "Poloni" \
    --titles "\$\\beta=0.0\$" "\$\\beta=0.25\$" "\$\\beta=0.5\$" "\$\\beta=0.75\$" "\$\\beta=1.0\$" \
    --data_paths "$DATA_PATH" "$DATA_PATH" "$DATA_PATH" "$DATA_PATH" "$DATA_PATH" \
    --trials $TRIALS \
    --filename "poloni_beta_subplot_comparison" \
    --whitelists "MOHOLLM (Gemini 2.0 Flash) (Beta=0.0),mohollm (Gemini 2.0 Flash) (Beta=0.0)" "MOHOLLM (Gemini 2.0 Flash) (Beta=0.25),mohollm (Gemini 2.0 Flash) (Beta=0.25)" "MOHOLLM (Gemini 2.0 Flash) (Beta=0.5),mohollm (Gemini 2.0 Flash) (Beta=0.5)" "MOHOLLM (Gemini 2.0 Flash) (Beta=0.75),mohollm (Gemini 2.0 Flash) (Beta=0.75)" "MOHOLLM (Gemini 2.0 Flash) (Beta=1.0),mohollm (Gemini 2.0 Flash) (Beta=1.0)"

echo "Plot generation complete!"
