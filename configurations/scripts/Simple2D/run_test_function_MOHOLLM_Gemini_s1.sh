#!/bin/bash
config_files=(
    "Simple2D/Gemini/MOHOLLM-TestFunction4-Gemini"
    "Simple2D/Gemini/MOHOLLM-ChankongHaimes-Gemini"
    "Simple2D/Gemini/MOHOLLM-Poloni-Gemini"
    "Simple2D/Gemini/MOHOLLM-SchafferN1-Gemini"
    "Simple2D/Gemini/MOHOLLM-SchafferN2-Gemini"
)
seeds=("31415927")

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