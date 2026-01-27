#!/bin/bash
config_files=(
    "Simple2D/mohollm-TestFunction4"
    "Simple2D/mohollm-ChankongHaimes"
    "Simple2D/mohollm-Poloni"
    "Simple2D/mohollm-SchafferN1"
    "Simple2D/mohollm-SchafferN2"
)
seeds=("6790")

model="gpt-4o-mini"
method_name="mohollm"

for seed in "${seeds[@]}"
do
    for config_file in "${config_files[@]}"
    do
        echo "Evaluating $config_file with seed $seed"
        python main.py \
            --config_file="$config_file" \
            --seed="$seed"\
            --model="$model" \
            --method_name="$method_name"
    done
done