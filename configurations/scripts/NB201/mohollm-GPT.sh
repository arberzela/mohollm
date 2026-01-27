#!/bin/bash
config_file="NB201/mohollm"
seeds=("31415927" "42" "6790")

model="gpt-4o-mini"
method_name="mohollm (GPT-4o Mini)"

for seed in "${seeds[@]}"
do
    echo "Evaluating NB201 with seed $seed"
    python main.py \
        --config_file="$config_file" \
        --seed="$seed" \
        --model="$model" \
        --method_name="$method_name"
done