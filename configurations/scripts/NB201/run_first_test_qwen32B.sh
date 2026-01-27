#!/bin/bash
seeds=("31415927" "42" "6790")
model="Qwen2.5-32B-Instruct-AWQ"


config_file="NB201/mohollm"
method_name="mohollm (Qwen2.5-32B)"
for seed in "${seeds[@]}"
do
    echo "Evaluating NB201 with seed $seed"
    python main.py \
        --config_file="$config_file" \
        --seed="$seed" \
        --model="$model" \
        --method_name="$method_name"
done

config_file="NB201/mohollm-VP"
method_name="mohollm-VP (Qwen2.5-32B)"
for seed in "${seeds[@]}"
do
    echo "Evaluating NB201 with seed $seed"
    python main.py \
        --config_file="$config_file" \
        --seed="$seed" \
        --model="$model" \
        --method_name="$method_name"
done


config_file="NB201/mohollm-VP-VIS"
method_name="mohollm-VP-VIS (Qwen2.5-32B)"
for seed in "${seeds[@]}"
do
    echo "Evaluating NB201 with seed $seed"
    python main.py \
        --config_file="$config_file" \
        --seed="$seed" \
        --model="$model" \
        --method_name="$method_name"
done


config_file="NB201/mohollm-VP-PE"
method_name="mohollm-VP-PE (Qwen2.5-32B)"
for seed in "${seeds[@]}"
do
    echo "Evaluating NB201 with seed $seed"
    python main.py \
        --config_file="$config_file" \
        --seed="$seed" \
        --model="$model" \
        --method_name="$method_name"
done