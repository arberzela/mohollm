#!/bin/bash
config_files=(
    "Simple2D/MOHOLLM-TestFunction4-HVC"
    "Simple2D/MOHOLLM-ChankongHaimes-HVC"
    "Simple2D/MOHOLLM-Poloni-HVC"
    "Simple2D/MOHOLLM-SchafferN1-HVC"
    "Simple2D/MOHOLLM-SchafferN2-HVC"
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