#!/bin/bash


# benchmarks=(DTLZ1 DTLZ2 DTLZ3 BraninCurrin ChankongHaimes GMM Poloni SchafferN1 SchafferN2 TestFunction4 ToyRobust Kursawe)
# # synthetic
# for b in "${benchmarks[@]}"; do
#     lower=$(echo "$b" | tr '[:upper:]' '[:lower:]')
#     python ./plot/plot_focus.py \
#         --benchmark "$b" \
#         --title "$b" \
#         --whitelist "MOHOLLM (Gemini 2.0 Flash),mohollm (Gemini 2.0 Flash)" \
#         --data_path "./results/$b/" \
#         --filename "/${lower}" \
#         --trials 12
# done


# benchmarks=(Penicillin CarSideImpact VehicleSafety)
# # Real world
# for b in "${benchmarks[@]}"; do
#     lower=$(echo "$b" | tr '[:upper:]' '[:lower:]')
#     python ./plot/plot_focus.py \
#         --benchmark "$b" \
#         --title "$b" \
#         --whitelist "MOHOLLM (Gemini 2.0 Flash) (Context),mohollm (Gemini 2.0 Flash)" \
#         --data_path "./results/$b/" \
#         --filename "/${lower}" \
#         --trials 12
# done


# # Prompt ablations
# python ./plot/plot_focus.py \
#     --benchmark "Penicillin" \
#     --title "Penicillin" \
#     --whitelist "mohollm (Gemini 2.0 Flash),mohollm (Gemini 2.0 Flash) - alpha_prompt=0.0,mohollm (Gemini 2.0 Flash) - alpha_prompt=0.5,mohollm (Gemini 2.0 Flash) - alpha_prompt=0.25,mohollm (Gemini 2.0 Flash) - alpha_prompt=0.75,mohollm (Gemini 2.0 Flash) - alpha_prompt=1.0" \
#     --data_path "./results/Penicillin/" \
#     --filename "/penicillin_prompt_ablations_LLM" \
#     --trials 12


# python ./plot/plot_focus.py \
#     --benchmark "Penicillin" \
#     --title "Penicillin" \
#     --whitelist "MOHOLLM (Gemini 2.0 Flash) (Context),MOHOLLM (Gemini 2.0 Flash) - alpha_prompt=0.0,MOHOLLM (Gemini 2.0 Flash) - alpha_prompt=0.5,MOHOLLM (Gemini 2.0 Flash) - alpha_prompt=0.25,MOHOLLM (Gemini 2.0 Flash) - alpha_prompt=0.75,MOHOLLM (Gemini 2.0 Flash) - alpha_prompt=1.0" \
#     --data_path "./results/Penicillin/" \
#     --filename "/penicillin_prompt_ablations_MOHOLLM" \
#     --trials 12

# After per-benchmark focus plots, also generate an aggregated focus plot across benchmarks
benchmarks="DTLZ1,DTLZ2,DTLZ3,BraninCurrin,ChankongHaimes,GMM,Poloni,SchafferN1,SchafferN2,TestFunction4,ToyRobust,Kursawe, Penicillin,CarSideImpact,VehicleSafety"
python ./plot/aggregate_focus.py \
    --data_path ./results \
    --benchmarks "$benchmarks" \
    --trials 12 \
    --output_path ./plots/feature_focus/aggregate \
    --filename aggregate_focus \
    --title "Aggregated Focus Score"