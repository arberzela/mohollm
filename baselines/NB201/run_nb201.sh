#!/bin/bash
methods=("RS" "RSBO" "LSBO" "NSGA2")
seeds=("31415927" "42" "6790")
devices=("fpga" "1080ti_32" "titanx_256")

for method in "${methods[@]}"
do  
    for seed in "${seeds[@]}"
    do
        for device in "${devices[@]}"
        do
            echo "Evaluating $method with seed $seed for $device"
            python run_nb201.py \
                --method "$method" \
                --random_seed "$seed" \
                --metric "$device" \
                --n_workers "10"
        done
    done
done

