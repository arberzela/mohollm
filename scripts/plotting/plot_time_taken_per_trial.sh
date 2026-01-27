#!/bin/bash


python ./plot/plot_time_per_trial.py \
    --benchmark "NB201" \
    --title "Usage cifar10 (1080ti_256_latency)" \
    --data_path "./results/NB201/cifar10/1080ti_256_latency" \
    --filename "/cifar10/1080ti_256_latency" \


python ./plot/plot_time_per_trial.py \
    --benchmark "NB201" \
    --title "Usage cifar10 (2080ti_256_latency)" \
    --data_path "./results/NB201/cifar10/2080ti_256_latency" \
    --filename "/cifar10/2080ti_256_latency" \


python ./plot/plot_time_per_trial.py \
    --benchmark "NB201" \
    --title "Usage cifar10 (titanxp_256_latency)" \
    --data_path "./results/NB201/cifar10/titanxp_256_latency" \
    --filename "/cifar10/titanxp_256_latency" \


python ./plot/plot_time_per_trial.py \
    --benchmark "NB201" \
    --title "Usage cifar10 (samsung_s7_latency)" \
    --data_path "./results/NB201/cifar10/samsung_s7_latency" \
    --filename "/cifar10/samsung_s7_latency" \


python ./plot/plot_time_per_trial.py \
    --benchmark "Penicillin" \
    --title "Usage Penicillin" \
    --data_path "./results/Penicillin/" \
    --filename "/penicillin" \

