#!/bin/bash
# config_file="NB201/mohollm-VP-VIS"
# seeds=("31415927" "42" "6790")

# model="gpt-4o-mini"
# method_name="mohollm-VP-VIS (GPT-4o Mini)"

# for seed in "${seeds[@]}"
# do
#     echo "Evaluating NB201 with seed $seed"
#     python main.py \
#         --config_file="$config_file" \
#         --seed="$seed" \
#         --model="$model" \
#         --method_name="$method_name"
# done


# #!/bin/bash
# config_file="NB201/mohollm-VP-VIS-ALL"
# seeds=("31415927" "42" "6790")

# model="gpt-4o-mini"
# method_name="mohollm-VP-VIS-ALL-REGIONS (GPT-4o Mini)"

# for seed in "${seeds[@]}"
# do
#     echo "Evaluating NB201 with seed $seed"
#     python main.py \
#         --config_file="$config_file" \
#         --seed="$seed" \
#         --model="$model" \
#         --method_name="$method_name"
# done


config_file="NB201/mohollm-VP-VIS-HE-ALL"
seeds=("31415927" "42" "6790")

model="gpt-4o-mini"
method_name="mohollm-AAAAAAAA (GPT-4o Mini)"

for seed in "${seeds[@]}"
do
    echo "Evaluating NB201 with seed $seed"
    python main.py \
        --config_file="$config_file" \
        --seed="$seed" \
        --model="$model" \
        --method_name="$method_name"
done
