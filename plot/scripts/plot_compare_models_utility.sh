#!/bin/bash


# NOTE: Here we use real trial and not "trials as function evaluations"
python ./plot/compare_models_utility.py \
    --benchmarks VehicleSafety \
    --title "Utility Difference" \
    --trials 12 \
    --whitelist "MOHOLLM (gemma-2-9b-it-fast) (Context),MOHOLLM (gpt-4o-mini) (Context),MOHOLLM (gpt-oss-120b) (Context),MOHOLLM (Llama-3.3-70B-Instruct) (Context),MOHOLLM (Meta-Llama-3.1-8B-Instruct) (Context),MOHOLLM (Qwen3-32B) (Context),MOHOLLM (Gemini 2.0 Flash) (Context)"\
    --filename "utility_plots_models"
