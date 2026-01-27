#!/bin/bash


# PLOT CANDIDATE SAMPLER
python ./plot/plot_icl_adaptability.py \
    --benchmark "NB201" \
    --title "ICL Adaptability cifar10 (1080ti_256_latency) (Candidate Sampler)" \
    --data_path "./results/NB201/cifar10/1080ti_256_latency" \
    --filename "sampler/cifar10/1080ti_256_latency"

python ./plot/plot_icl_adaptability.py \
    --benchmark "NB201" \
    --title "ICL Adaptability cifar10 (2080ti_256_latency) (Candidate Sampler)" \
    --data_path "./results/NB201/cifar10/2080ti_256_latency" \
    --filename "sampler/cifar10/2080ti_256_latency"


python ./plot/plot_icl_adaptability.py \
    --benchmark "NB201" \
    --title "ICL Adaptability cifar10 (titanxp_256_latency) (Candidate Sampler)" \
    --data_path "./results/NB201/cifar10/titanxp_256_latency" \
    --filename "sampler/cifar10/titanxp_256_latency"


python ./plot/plot_icl_adaptability.py \
    --benchmark "NB201" \
    --title "ICL Adaptability cifar10 (samsung_s7_latency) (Candidate Sampler)" \
    --data_path "./results/NB201/cifar10/samsung_s7_latency" \
    --filename "sampler/cifar10/samsung_s7_latency"

python ./plot/plot_icl_adaptability.py \
    --benchmark "Penicillin" \
    --title "ICL Adaptability Penicillin (Candidate Sampler)" \
    --data_path "./results/Penicillin/" \
    --filename "sampler/penicillin"


# PLOT SURROGATE MODEL SUGGESTIONS
python ./plot/plot_icl_adaptability.py \
    --benchmark "NB201" \
    --title "ICL Adaptability cifar10 (1080ti_256_latency) (Surrogate Model)" \
    --data_path "./results/NB201/cifar10/1080ti_256_latency" \
    --filename "surrogate/cifar10/1080ti_256_latency" \
    --plot_surrogate_model true

python ./plot/plot_icl_adaptability.py \
    --benchmark "NB201" \
    --title "ICL Adaptability cifar10 (2080ti_256_latency) (Surrogate Model)" \
    --data_path "./results/NB201/cifar10/2080ti_256_latency" \
    --filename "surrogate/cifar10/2080ti_256_latency" \
    --plot_surrogate_model true


python ./plot/plot_icl_adaptability.py \
    --benchmark "NB201" \
    --title "ICL Adaptability cifar10 (titanxp_256_latency) (Surrogate Model)" \
    --data_path "./results/NB201/cifar10/titanxp_256_latency" \
    --filename "surrogate/cifar10/titanxp_256_latency" \
    --plot_surrogate_model true


python ./plot/plot_icl_adaptability.py \
    --benchmark "NB201" \
    --title "ICL Adaptability cifar10 (samsung_s7_latency) (Surrogate Model)" \
    --data_path "./results/NB201/cifar10/samsung_s7_latency" \
    --filename "surrogate/cifar10/samsung_s7_latency" \
    --plot_surrogate_model true

python ./plot/plot_icl_adaptability.py \
    --benchmark "Penicillin" \
    --title "ICL Adaptability Penicillin (Surrogate Model)" \
    --data_path "./results/Penicillin/" \
    --filename "surrogate/penicillin" \
    --plot_surrogate_model true