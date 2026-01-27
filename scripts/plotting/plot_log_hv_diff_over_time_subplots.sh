#!/bin/bash

# Subplots for log hypervolume difference (same style as HV/EAF subplots)
# TODO: Unknown max hv for these benchmarks
# whitelist="CTAEA,EHVI,GDE3,IBEA,MOEAD,,NSGA2,NSGA3,PESA2,qLogEHVI,RNSGA2,SMSEMOA,SPEA2,UNSGA3,mohollm (Gemini 2.0 Flash),MOHOLLM (Gemini 2.0 Flash),MOHOLLM (Gemini 2.0 Flash) (Context)"

# python3 ./plot/plot_log_hv_diff_over_time_subplots.py \
#     --benchmarks DTLZ1 DTLZ2 DTLZ3 BraninCurrin ChankongHaimes GMM Poloni SchafferN1 SchafferN2 TestFunction4 ToyRobust Kursawe \
#     --titles "DTLZ1" "DTLZ2" "DTLZ3" "BraninCurrin" "ChankongHaimes" "GMM" "Poloni" "SchafferN1" "SchafferN2" "TestFunction4" "ToyRobust" "Kursawe" \
#     --trials 50 \
#     --filename "synthetic_log_hv_diff" \
#     --data_paths ./results/DTLZ1/ ./results/DTLZ2/ ./results/DTLZ3/ ./results/BraninCurrin/ ./results/ChankongHaimes/ ./results/GMM/ ./results/Poloni/ ./results/SchafferN1/ ./results/SchafferN2/ ./results/TestFunction4/ ./results/ToyRobust/ ./results/Kursawe/ \
#     --whitelists "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist"

whitelist="CTAEA,EHVI,GDE3,IBEA,MOEAD,MOHOLLM (Gemini 2.0 Flash) (Context),mohollm (Gemini 2.0 Flash),NSGA2,NSGA3,PESA2,qLogEHVI,RNSGA2,SMSEMOA,SPEA2,UNSGA3"
python3 ./plot/plot_log_hv_diff_over_time_subplots.py \
    --benchmarks Penicillin CarSideImpact VehicleSafety \
    --titles "Penicillin" "CarSideImpact" "VehicleSafety" \
    --trials 50 \
    --filename "real_world_log_hv_diff" \
    --data_paths ./results/Penicillin/ ./results/CarSideImpact/ ./results/VehicleSafety/ \
    --whitelists "$whitelist" "$whitelist" "$whitelist"

# TODO: Unknown max hv for these benchmarks
# ONLY LLM AND MOHOLLM
# whitelist="mohollm (Gemini 2.0 Flash),MOHOLLM (Gemini 2.0 Flash),MOHOLLM (Gemini 2.0 Flash) (Context)"
# python3 ./plot/plot_log_hv_diff_over_time_subplots.py \
#     --benchmarks DTLZ1 DTLZ2 DTLZ3 BraninCurrin ChankongHaimes GMM Poloni SchafferN1 SchafferN2 TestFunction4 ToyRobust Kursawe \
#     --titles "DTLZ1" "DTLZ2" "DTLZ3" "BraninCurrin" "ChankongHaimes" "GMM" "Poloni" "SchafferN1" "SchafferN2" "TestFunction4" "ToyRobust" "Kursawe" \
#     --trials 50 \
#     --filename "synthetic_log_hv_diff_only_llm" \
#     --data_paths ./results/DTLZ1/ ./results/DTLZ2/ ./results/DTLZ3/ ./results/BraninCurrin/ ./results/ChankongHaimes/ ./results/GMM/ ./results/Poloni/ ./results/SchafferN1/ ./results/SchafferN2/ ./results/TestFunction4/ ./results/ToyRobust/ ./results/Kursawe/ \
#     --whitelists "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist"

whitelist="MOHOLLM (Gemini 2.0 Flash) (Context),mohollm (Gemini 2.0 Flash)"
python3 ./plot/plot_log_hv_diff_over_time_subplots.py \
    --benchmarks Penicillin CarSideImpact VehicleSafety \
    --titles "Penicillin" "CarSideImpact" "VehicleSafety" \
    --trials 50 \
    --filename "real_world_log_hv_diff_only_llm" \
    --data_paths ./results/Penicillin/ ./results/CarSideImpact/ ./results/VehicleSafety/ \
    --whitelists "$whitelist" "$whitelist" "$whitelist"
