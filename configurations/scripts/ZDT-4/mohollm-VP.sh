#!/bin/bash
config_file="ZDT-4/mohollm-VP"
seeds=("31415927" "42" "6790")

model="gemini-1-5-flash"
method_name="mohollm test (Gemini 1.5 Flash)"

for seed in "${seeds[@]}"
do
    echo "Evaluating ZDT4 with seed $seed"
    python main.py \
        --config_file="$config_file" \
        --seed="$seed"\
        --model="$model" \
        --method_name="$method_name"
done