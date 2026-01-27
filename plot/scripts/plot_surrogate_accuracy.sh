#!/bin/bash


# python ./plot/plot_surrogate_accuracy.py \
#     --benchmark "Penicillin" \
#     --title "Surrogate Accuracy" \
#     --data_path "./results/Penicillin/" \
#     --filename "penicillin" \
#     --n 700 \
#     --whitelist "MOHOLLM + GP (Gemini 2.0 Flash) (Context),MOHOLLM + TabPFN (Gemini 2.0 Flash) (Context),MOHOLLM (Gemini 2.0 Flash) (Context),mohollm (Gemini 2.0 Flash)"


# python ./plot/plot_surrogate_accuracy.py \
#     --benchmark "VehicleSafety" \
#     --title "Surrogate Accuracy" \
#     --data_path "./results/VehicleSafety/" \
#     --filename "vehicle_safety" \
#     --n 700 \
#     --whitelist "MOHOLLM + GP (Gemini 2.0 Flash) (Context),MOHOLLM + TabPFN (Gemini 2.0 Flash) (Context),MOHOLLM (Gemini 2.0 Flash) (Context),mohollm (Gemini 2.0 Flash)"

# python ./plot/plot_surrogate_accuracy.py \
#     --benchmark "CarSideImpact" \
#     --title "Surrogate Accuracy" \
#     --data_path "./results/CarSideImpact/" \
#     --filename "car_side_impact" \
#     --n 700 \
#     --whitelist "MOHOLLM + GP (Gemini 2.0 Flash) (Context),MOHOLLM + TabPFN (Gemini 2.0 Flash) (Context),MOHOLLM (Gemini 2.0 Flash) (Context),mohollm (Gemini 2.0 Flash)"


# python ./plot/plot_surrogate_accuracy.py \
#     --benchmark "Penicillin" \
#     --title "Surrogate Accuracy" \
#     --data_path "./results/Penicillin/" \
#     --filename "penicillin_with_botorch_gp" \
#     --n 700 \
#     --whitelist "MOHOLLM + GP (BoTorch) (Gemini 2.0 Flash) (Context),MOHOLLM + TabPFN (Gemini 2.0 Flash) (Context),MOHOLLM (Gemini 2.0 Flash) (Context),mohollm (Gemini 2.0 Flash)"


# python ./plot/plot_surrogate_accuracy.py \
#     --benchmark "VehicleSafety" \
#     --title "Surrogate Accuracy" \
#     --data_path "./results/VehicleSafety/" \
#     --filename "vehicle_safety_with_botorch_gp" \
#     --n 700 \
#     --whitelist "MOHOLLM + GP (BoTorch) (Gemini 2.0 Flash) (Context),MOHOLLM + TabPFN (Gemini 2.0 Flash) (Context),MOHOLLM (Gemini 2.0 Flash) (Context),mohollm (Gemini 2.0 Flash)"

# PYTHONPATH=. python ./plot/plot_surrogate_accuracy.py \
#     --benchmark "CarSideImpact" \
#     --title "CarSideImpact" \
#     --data_path "./results/CarSideImpact/" \
#     --filename "surr_acc_car_side_impact_paper" \
#     --n 700 \
#     --whitelist "MOHOLLM + GP (BoTorch) (Gemini 2.0 Flash) (Context),MOHOLLM + TabPFN (Gemini 2.0 Flash) (Context),MOHOLLM (Gemini 2.0 Flash) (Context)"

# whitelist="MOHOLLM + GP (BoTorch) (Gemini 2.0 Flash) (Context),MOHOLLM + TabPFN (Gemini 2.0 Flash) (Context),MOHOLLM (Gemini 2.0 Flash) (Context),mohollm (Gemini 2.0 Flash),MOHOLLM (gemma-2-9b-it-fast) (Context),MOHOLLM (gpt-oss-120b) (Context),MOHOLLM (Llama-3.3-70B-Instruct) (Context),MOHOLLM (Meta-Llama-3.1-8B-Instruct) (Context),MOHOLLM (Qwen3-32B) (Context),MOHOLLM (gpt-4o-mini) (Context)"

# python ./plot/plot_surrogate_accuracy.py \
#     --benchmark "VehicleSafety" \
#     --title "VehicleSafety" \
#     --data_path "./results/VehicleSafety/" \
#     --filename "vehicle_safety_with_botorch_gp" \
#     --n 700 \
#     --whitelist "MOHOLLM + GP (BoTorch) (Gemini 2.0 Flash) (Context),MOHOLLM + TabPFN (Gemini 2.0 Flash) (Context),MOHOLLM (Gemini 2.0 Flash) (Context),mohollm (Gemini 2.0 Flash)"

# python ./plot/plot_surrogate_accuracy.py \
#     --benchmark "Penicillin" \
#     --title "Penicillin" \
#     --data_path "./results/Penicillin/" \
#     --filename "penicillin_with_botorch_gp" \
#     --n 700 \
#     --whitelist "MOHOLLM + GP (BoTorch) (Gemini 2.0 Flash) (Context),MOHOLLM + TabPFN (Gemini 2.0 Flash) (Context),MOHOLLM (Gemini 2.0 Flash) (Context),mohollm (Gemini 2.0 Flash)"


# python ./plot/plot_surrogate_accuracy.py \
#     --benchmark "CarSideImpact" \
#     --title "Surrogate Accuracy" \
#     --data_path "./results/CarSideImpact/" \
#     --filename "car_side_impact_with_botorch_gp" \
#     --n 700 \
#     --whitelist "$whitelist"


# Paper

PYTHONPATH=. python ./plot/plot_surrogate_accuracy.py \
    --benchmark "CarSideImpact" \
    --title "CarSideImpact" \
    --data_path "./results/CarSideImpact/" \
    --filename "surr_acc_car_side_impact_paper" \
    --n 700 \
    --whitelist "MOHOLLM + GP (BoTorch) (Gemini 2.0 Flash) (Context),MOHOLLM + TabPFN (Gemini 2.0 Flash) (Context),MOHOLLM (Gemini 2.0 Flash) (Context)"

PYTHONPATH=. python ./plot/plot_surrogate_accuracy.py \
    --benchmark "Penicillin" \
    --title "Penicillin" \
    --data_path "./results/Penicillin/" \
    --filename "surr_acc_penicillin_paper" \
    --n 700 \
    --whitelist "MOHOLLM + GP (BoTorch) (Gemini 2.0 Flash) (Context),MOHOLLM + TabPFN (Gemini 2.0 Flash) (Context),MOHOLLM (Gemini 2.0 Flash) (Context)"

PYTHONPATH=. python ./plot/plot_surrogate_accuracy.py \
    --benchmark "VehicleSafety" \
    --title "VehicleSafety" \
    --data_path "./results/VehicleSafety/" \
    --filename "surr_acc_vehicle_safety_paper" \
    --n 700 \
    --whitelist "MOHOLLM + GP (BoTorch) (Gemini 2.0 Flash) (Context),MOHOLLM + TabPFN (Gemini 2.0 Flash) (Context),MOHOLLM (Gemini 2.0 Flash) (Context)"
