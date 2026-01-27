#!/bin/bash
seeds=("31415927" "42" "6790")
model="gpt-4o-mini"


config_file="NB201/mohollm"
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

config_file="NB201/mohollm-VP"
method_name="mohollm-VP (GPT-4o Mini)"
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
method_name="mohollm-VP-VIS (GPT-4o Mini)"
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
method_name="mohollm-VP-PE (GPT-4o Mini)"
for seed in "${seeds[@]}"
do
    echo "Evaluating NB201 with seed $seed"
    python main.py \
        --config_file="$config_file" \
        --seed="$seed" \
        --model="$model" \
        --method_name="$method_name"
done