#!/bin/bash
config_file="ZDT3/mohollm-KD"
seeds=("31415927" "42" "6790")

model="gpt-4o-mini"
method_name="mohollm-KD (GPT-4o Mini)"

for seed in "${seeds[@]}"
do
    echo "Evaluating ZDT3 with seed $seed"
    python main.py \
        --config_file="$config_file" \
        --seed="$seed"\
        --model="$model" \
        --method_name="$method_name"
done