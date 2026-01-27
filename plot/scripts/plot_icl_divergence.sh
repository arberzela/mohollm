#!/bin/bash


# python ./plot/plot_icl_divergence.py \
#     --benchmark "Penicillin" \
#     --title "ICL Divergence Penicillin (Different Prompt Instructions)" \
#     --data_path "./results/Penicillin/" \
#     --filename "sampler/penicillin_prompt" \
#     --trials 12 \
#     --whitelist "mohollm (Gemini 2.0 Flash),mohollm (Gemini 2.0 Flash) - alpha_prompt=0.0,mohollm (Gemini 2.0 Flash) - alpha_prompt=0.5,mohollm (Gemini 2.0 Flash) - alpha_prompt=0.25,mohollm (Gemini 2.0 Flash) - alpha_prompt=0.75,mohollm (Gemini 2.0 Flash) - alpha_prompt=1.0"


# python ./plot/plot_icl_divergence.py \
#     --benchmark "Penicillin" \
#     --title "ICL Divergence Penicillin (Different Prompt Instructions)" \
#     --data_path "./results/Penicillin/" \
#     --filename "sampler/penicillin_prompt_partitioning" \
#     --trials 12 \
#     --whitelist "MOHOLLM (Gemini 2.0 Flash) - alpha_prompt=0.0,MOHOLLM (Gemini 2.0 Flash) - alpha_prompt=0.5,MOHOLLM (Gemini 2.0 Flash) - alpha_prompt=0.25,MOHOLLM (Gemini 2.0 Flash) - alpha_prompt=0.75,MOHOLLM (Gemini 2.0 Flash) - alpha_prompt=1.0,MOHOLLM (Gemini 2.0 Flash) (Context)"



# benchmarks=(DTLZ1 DTLZ2 DTLZ3 BraninCurrin ChankongHaimes GMM Poloni SchafferN1 SchafferN2 TestFunction4 ToyRobust Kursawe)
# # synthetic
# for b in "${benchmarks[@]}"; do
#     lower=$(echo "$b" | tr '[:upper:]' '[:lower:]')
#     python ./plot/plot_icl_divergence.py \
#         --benchmark "$b" \
#         --title "$b" \
#         --whitelist "MOHOLLM (Gemini 2.0 Flash),mohollm (Gemini 2.0 Flash)" \
#         --data_path "./results/$b/" \
#         --filename "/${lower}" \
#         --trials 12
# done

# # synthetic
# python ./plot/plot_icl_divergence.py \
#     --benchmark "Poloni" \
#     --title "Poloni" \
#     --whitelist "MOHOLLM (Gemini 2.0 Flash) (Beta=0.0),MOHOLLM (Gemini 2.0 Flash) (Beta=0.25),MOHOLLM (Gemini 2.0 Flash) (Beta=0.5),MOHOLLM (Gemini 2.0 Flash) (Beta=0.75),MOHOLLM (Gemini 2.0 Flash) (Beta=1.0),mohollm (Gemini 2.0 Flash) (Beta=0.0),mohollm (Gemini 2.0 Flash) (Beta=0.25),mohollm (Gemini 2.0 Flash) (Beta=0.5),mohollm (Gemini 2.0 Flash) (Beta=0.75),mohollm (Gemini 2.0 Flash) (Beta=1.0)" \
#     --data_path "./results/Poloni/" \
#     --filename "beta_icl_divergence" \
#     --trials 12



# benchmarks=(Penicillin CarSideImpact VehicleSafety)
# # Real world
# for b in "${benchmarks[@]}"; do
#     lower=$(echo "$b" | tr '[:upper:]' '[:lower:]')
#     python ./plot/plot_icl_divergence.py \
#         --benchmark "$b" \
#         --title "$b" \
#         --whitelist "MOHOLLM (Gemini 2.0 Flash) (Context),mohollm (Gemini 2.0 Flash)" \
#         --data_path "./results/$b/" \
#         --filename "/${lower}" \
#         --trials 12
# done




# # ICL Divergence of different LLMs
# python ./plot/plot_icl_divergence.py \
#         --benchmark "VehicleSafety" \
#         --title "VehicleSafety" \
#         --whitelist "MOHOLLM (Gemini 2.0 Flash) (Context),MOHOLLM (gemma-2-9b-it-fast) (Context),MOHOLLM (gpt-oss-120b) (Context),MOHOLLM (Llama-3.3-70B-Instruct) (Context),MOHOLLM (Meta-Llama-3.1-8B-Instruct) (Context),MOHOLLM (Qwen3-32B) (Context),MOHOLLM (gpt-4o-mini) (Context)" \
#         --data_path "./results/VehicleSafety/" \
#         --filename "/VehicleSafety_model_comparision" \
#         --trials 12



# After creating per-benchmark plots, generate an aggregated ICL Divergence plot
benchmarks="DTLZ1,DTLZ2,DTLZ3,BraninCurrin,ChankongHaimes,GMM,Poloni,SchafferN1,SchafferN2,TestFunction4,ToyRobust,Kursawe,Penicillin,CarSideImpact,VehicleSafety"
python ./plot/aggregate_icl_divergence.py \
    --data_path ./results \
    --benchmarks "$benchmarks" \
    --trials 12 \
    --output_path ./plots/icl_divergence/aggregate \
    --filename aggregate_icl_divergence \
    --title "Aggregated ICL Divergence"