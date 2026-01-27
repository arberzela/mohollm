#!/bin/bash
trials=100
methods=("RS" "RSBO")

seeds=("31415927" "42" "6790")

# problems=("chankong_haimes" "test_function_4" "schaffer_n1" "schaffer_n2" "poloni")


problems=("nb201_cifar10_fpga_latency" "nb201_cifar10_pixel3_latency" "nb201_cifar10_raspi4_latency" "nb201_cifar10_eyeriss_latency" "nb201_cifar10_pixel2_latency" "nb201_cifar10_1080ti_1_latency" "nb201_cifar10_1080ti_32_latency" "nb201_cifar10_1080ti_256_latency" "nb201_cifar10_2080ti_1_latency" "nb201_cifar10_2080ti_32_latency" "nb201_cifar10_2080ti_256_latency" "nb201_cifar10_titanx_1_latency" "nb201_cifar10_titanx_32_latency" "nb201_cifar10_titanx_256_latency" "nb201_cifar10_titanxp_1_latency" "nb201_cifar10_titanxp_32_latency" "nb201_cifar10_titanxp_256_latency" "nb201_cifar10_titan_rtx_1_latency" "nb201_cifar10_titan_rtx_32_latency" "nb201_cifar10_titan_rtx_256_latency" "nb201_cifar10_essential_ph_1_latency" "nb201_cifar10_gold_6226_latency" "nb201_cifar10_gold_6240_latency" "nb201_cifar10_samsung_a50_latency" "nb201_cifar10_samsung_s7_latency" "nb201_cifar10_silver_4114_latency" "nb201_cifar10_silver_4210r_latency" "nb201_cifar100_fpga_latency" "nb201_cifar100_pixel3_latency" "nb201_cifar100_raspi4_latency" "nb201_cifar100_eyeriss_latency" "nb201_cifar100_pixel2_latency" "nb201_cifar100_1080ti_1_latency" "nb201_cifar100_1080ti_32_latency" "nb201_cifar100_1080ti_256_latency" "nb201_cifar100_2080ti_1_latency" "nb201_cifar100_2080ti_32_latency" "nb201_cifar100_2080ti_256_latency" "nb201_cifar100_titanx_1_latency" "nb201_cifar100_titanx_32_latency" "nb201_cifar100_titanx_256_latency" "nb201_cifar100_titanxp_1_latency" "nb201_cifar100_titanxp_32_latency" "nb201_cifar100_titanxp_256_latency" "nb201_cifar100_titan_rtx_1_latency" "nb201_cifar100_titan_rtx_32_latency" "nb201_cifar100_titan_rtx_256_latency" "nb201_cifar100_essential_ph_1_latency" "nb201_cifar100_gold_6226_latency" "nb201_cifar100_gold_6240_latency" "nb201_cifar100_samsung_a50_latency" "nb201_cifar100_samsung_s7_latency" "nb201_cifar100_silver_4114_latency" "nb201_cifar100_silver_4210r_latency" "nb201_imagenet16_fpga_latency" "nb201_imagenet16_pixel3_latency" "nb201_imagenet16_raspi4_latency" "nb201_imagenet16_eyeriss_latency" "nb201_imagenet16_pixel2_latency" "nb201_imagenet16_1080ti_1_latency" "nb201_imagenet16_1080ti_32_latency" "nb201_imagenet16_1080ti_256_latency" "nb201_imagenet16_2080ti_1_latency" "nb201_imagenet16_2080ti_32_latency" "nb201_imagenet16_2080ti_256_latency" "nb201_imagenet16_titanx_1_latency" "nb201_imagenet16_titanx_32_latency" "nb201_imagenet16_titanx_256_latency" "nb201_imagenet16_titanxp_1_latency" "nb201_imagenet16_titanxp_32_latency" "nb201_imagenet16_titanxp_256_latency" "nb201_imagenet16_titan_rtx_1_latency" "nb201_imagenet16_titan_rtx_32_latency" "nb201_imagenet16_titan_rtx_256_latency" "nb201_imagenet16_essential_ph_1_latency" "nb201_imagenet16_gold_6226_latency" "nb201_imagenet16_gold_6240_latency" "nb201_imagenet16_samsung_a50_latency" "nb201_imagenet16_samsung_s7_latency" "nb201_imagenet16_silver_4114_latency" "nb201_imagenet16_silver_4210r_latency")


seed_to_run=""

# Loop through arguments to find the --seed flag
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --seed)
            # The next argument ($2) is the value for the seed
            seed_to_run="$2"
            # "shift" consumes the arguments we've processed
            shift 2
            ;;
        *)
            # Handle unknown parameters
            echo "Unknown parameter passed: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$seed_to_run" ]]; then
    echo "Error: --seed flag is required."
    echo "Usage: $0 --seed <your_seed_number>"
    exit 1
fi


for problem in "${problems[@]}"
do
    for method in "${methods[@]}"
    do
        echo "Evaluating $method with seed $seed for $problem"
        python run_synetune_methods.py \
            --method "$method" \
            --seed "$seed_to_run" \
            --trials "$trials" \
            --problem "$problem"
    done
done
