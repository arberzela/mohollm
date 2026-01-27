#!/bin/bash
# config_file="Welded_Beam/mohollm"
# seeds=("31415927" "42" "6790")

# model="gpt-4o-mini"
# method_name="mohollm (GPT-4o Mini)"

# for seed in "${seeds[@]}"
# do
#     echo "Evaluating welded beam with seed $seed"
#     python main.py \
#         --config_file="$config_file" \
#         --seed="$seed"\
#         --model="$model" \
#         --method_name="$method_name"
# done


# config_file="Welded_Beam/mohollm-VP"
# seeds=("31415927" "42" "6790")

# model="gpt-4o-mini"
# method_name="mohollm-VP (GPT-4o Mini)"

# for seed in "${seeds[@]}"
# do
#     echo "Evaluating welded beam with seed $seed"
#     python main.py \
#         --config_file="$config_file" \
#         --seed="$seed"\
#         --model="$model" \
#         --method_name="$method_name"
# done


config_file="Welded_Beam/mohollm-KD"
seeds=("31415927" "42" "6790")

model="gpt-4o-mini"
method_name="mohollm-KD (GPT-4o Mini)"

for seed in "${seeds[@]}"
do
    echo "Evaluating welded beam with seed $seed"
    python main.py \
        --config_file="$config_file" \
        --seed="$seed"\
        --model="$model" \
        --method_name="$method_name"
done