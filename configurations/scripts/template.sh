#!/bin/bash
config_file="<config_file>"
seeds=("31415927" "42" "6790")

model="<MODEL>"
method_name="<METHOD_NAME>"

for seed in "${seeds[@]}"
do
    echo "Evaluating <BENCHMARK> with seed $seed"
    python main.py \
        --config_file="$config_file" \
        --seed="$seed"\
        --model="$model" \
        --method_name="$method_name"
done