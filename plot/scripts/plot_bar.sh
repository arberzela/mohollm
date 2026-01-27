#!/usr/bin/env bash


python plot/plot_bar_compare.py \
  --benchmark "vehicle_safety" \
  --data_path "./results/vehicle_safety_single_objective" \
  --whitelist "LLM,LLMKD,LLM-llama3.1-8b-llm,LLM-llama3.3-70b-llm,LLMKD-llama3.1-8b,LLMKD-llama3.3-70b" \
  --trials 50 \
  --filename "vehicle_safety" \
  --title "Bar plot" \
  --ylabel "Final function value"

