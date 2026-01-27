#!/bin/bash

# # Subplots for empirical attainment (same style as HV subplots)
# whitelist="CTAEA,EHVI,GDE3,IBEA,MOEAD,,NSGA2,NSGA3,PESA2,qLogEHVI,RNSGA2,SMSEMOA,SPEA2,UNSGA3,mohollm (Gemini 2.0 Flash),MOHOLLM (Gemini 2.0 Flash),MOHOLLM (Gemini 2.0 Flash) (Context)"
whitelist="GDE3,NSGA2,qLogEHVI,mohollm (Gemini 2.0 Flash),MOHOLLM (Gemini 2.0 Flash),MOHOLLM (Gemini 2.0 Flash) (Context)"

python3 ./plot/plot_pareto_set_empirical_attainment_subplots.py \
    --benchmarks DTLZ1 DTLZ2 DTLZ3 BraninCurrin ChankongHaimes GMM Poloni SchafferN1 SchafferN2 TestFunction4 ToyRobust Kursawe \
    --titles "DTLZ1" "DTLZ2" "DTLZ3" "BraninCurrin" "ChankongHaimes" "GMM" "Poloni" "SchafferN1" "SchafferN2" "TestFunction4" "ToyRobust" "Kursawe" \
    --trials 50 \
    --filename "synthetic_eaf" \
    --data_paths ./results/DTLZ1/ ./results/DTLZ2/ ./results/DTLZ3/ ./results/BraninCurrin/ ./results/ChankongHaimes/ ./results/GMM/ ./results/Poloni/ ./results/SchafferN1/ ./results/SchafferN2/ ./results/TestFunction4/ ./results/ToyRobust/ ./results/Kursawe/ \
    --columns "F1,F2" "F1,F2" "F1,F2" "F1,F2" "F1,F2" "F1,F2" "F1,F2" "F1,F2" "F1,F2" "F1,F2" "F1,F2" "F1,F2" \
    --whitelists "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist"


# Real-world benchmarks
whitelist="GDE3,NSGA2,qLogEHVI,mohollm (Gemini 2.0 Flash),MOHOLLM (Gemini 2.0 Flash) (Context)"

python3 ./plot/plot_pareto_set_empirical_attainment_subplots.py \
    --benchmarks Penicillin CarSideImpact VehicleSafety \
    --titles "Penicillin" "CarSideImpact" "VehicleSafety" \
    --trials 50 \
    --filename "real_world_eaf" \
    --data_paths ./results/Penicillin/ ./results/CarSideImpact/ ./results/VehicleSafety/ \
    --columns "F1,F2" "F1,F2" "F1,F2" \
    --whitelists "$whitelist" "$whitelist" "$whitelist"


# ONLY LLM AND MOHOLLM
# whitelist="mohollm (Gemini 2.0 Flash),MOHOLLM (Gemini 2.0 Flash),MOHOLLM (Gemini 2.0 Flash) (Context)"
# python3 ./plot/plot_pareto_set_empirical_attainment_subplots.py \
#     --benchmarks DTLZ1 DTLZ2 DTLZ3 BraninCurrin ChankongHaimes GMM Poloni SchafferN1 SchafferN2 TestFunction4 ToyRobust Kursawe \
#     --titles "DTLZ1" "DTLZ2" "DTLZ3" "BraninCurrin" "ChankongHaimes" "GMM" "Poloni" "SchafferN1" "SchafferN2" "TestFunction4" "ToyRobust" "Kursawe" \
#     --trials 50 \
#     --filename "synthetic_eaf_only_llm" \
#     --data_paths ./results/DTLZ1/ ./results/DTLZ2/ ./results/DTLZ3/ ./results/BraninCurrin/ ./results/ChankongHaimes/ ./results/GMM/ ./results/Poloni/ ./results/SchafferN1/ ./results/SchafferN2/ ./results/TestFunction4/ ./results/ToyRobust/ ./results/Kursawe/ \
#     --columns "F1,F2" "F1,F2" "F1,F2" "F1,F2" "F1,F2" "F1,F2" "F1,F2" "F1,F2" "F1,F2" "F1,F2" "F1,F2" "F1,F2" \
#     --whitelists "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist"


# BETA ABLATION ON POLONI
# DATA_PATH="./results/Poloni/"
# TRIALS=50
# python3 ./plot/plot_pareto_set_empirical_attainment_subplots.py \
#     --benchmarks "Poloni" "Poloni" "Poloni" "Poloni" "Poloni" \
#     --titles "\$\\beta=0.0\$" "\$\\beta=0.25\$" "\$\\beta=0.5\$" "\$\\beta=0.75\$" "\$\\beta=1.0\$" \
#     --data_paths "$DATA_PATH" "$DATA_PATH" "$DATA_PATH" "$DATA_PATH" "$DATA_PATH" \
#     --trials $TRIALS \
#     --filename "poloni_beta_eaf_comparison" \
#     --columns "F1,F2" "F1,F2" "F1,F2" "F1,F2" "F1,F2" \
#     --whitelists "MOHOLLM (Gemini 2.0 Flash) (Beta=0.0),mohollm (Gemini 2.0 Flash) (Beta=0.0)" "MOHOLLM (Gemini 2.0 Flash) (Beta=0.25),mohollm (Gemini 2.0 Flash) (Beta=0.25)" "MOHOLLM (Gemini 2.0 Flash) (Beta=0.5),mohollm (Gemini 2.0 Flash) (Beta=0.5)" "MOHOLLM (Gemini 2.0 Flash) (Beta=0.75),mohollm (Gemini 2.0 Flash) (Beta=0.75)" "MOHOLLM (Gemini 2.0 Flash) (Beta=1.0),mohollm (Gemini 2.0 Flash) (Beta=1.0)"
