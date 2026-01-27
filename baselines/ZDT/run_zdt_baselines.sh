#!/bin/bash
trials=100
# methods=("RS" "RSBO" "LSBO" "NSGA2") # TODO: Run NSGA2 later
methods=("NSGA2")
seeds=("31415927" "42" "6790")

problem="zdt1"
for method in "${methods[@]}"
do  
    for seed in "${seeds[@]}"
    do
        echo "Evaluating $method with seed $seed for $problem"
        python run_zdt_baselines.py \
            --method "$method" \
            --seed "$seed" \
            --trials "$trials" \
            --problem "$problem"
    done
done

problem="zdt2"
for method in "${methods[@]}"
do  
    for seed in "${seeds[@]}"
    do
        echo "Evaluating $method with seed $seed for $problem"
        python run_zdt_baselines.py \
            --method "$method" \
            --seed "$seed" \
            --trials "$trials" \
            --problem "$problem"
    done
done


# problem="zdt3"
# for method in "${methods[@]}"
# do  
#     for seed in "${seeds[@]}"
#     do
#         echo "Evaluating $method with seed $seed for $problem"
#         python run_zdt_baselines.py \
#             --method "$method" \
#             --seed "$seed" \
#             --trials "$trials" \
#             --problem "$problem"
#     done
# done

problem="zdt4"
for method in "${methods[@]}"
do  
    for seed in "${seeds[@]}"
    do
        echo "Evaluating $method with seed $seed for $problem"
        python run_zdt_baselines.py \
            --method "$method" \
            --seed "$seed" \
            --trials "$trials" \
            --problem "$problem"
    done
done

problem="zdt6"
for method in "${methods[@]}"
do  
    for seed in "${seeds[@]}"
    do
        echo "Evaluating $method with seed $seed for $problem"
        python run_zdt_baselines.py \
            --method "$method" \
            --seed "$seed" \
            --trials "$trials" \
            --problem "$problem"
    done
done


