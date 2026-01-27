#!/bin/bash

trials=100
problem="welded_beam"
methods=("RS" "RSBO" "LSBO" "NSGA2")
methods=("LSBO")
seeds=("31415927" "42" "6790")


for method in "${methods[@]}"
do  
    for seed in "${seeds[@]}"
    do
        echo "Evaluating $method with seed $seed for $problem"
        python run_welded_baselines.py \
            --method "$method" \
            --seed "$seed" \
            --trials "$trials" \
            --problem "$problem"
    done
done

