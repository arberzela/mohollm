#!/bin/bash

python ./plot/plot_pareto_set_empirical_attainment.py \
    --benchmark ChankongHaimes \
    --title "Pareto set empirical attainment (ChankongHaimes)" \
    --columns "F1,F2" \
    --trials 70 \
    --data_path ./results/ChankongHaimes/ \
    --filename "MOHOLLM_ChankongHaimes"

python ./plot/plot_pareto_set_empirical_attainment.py \
    --benchmark TestFunction4 \
    --title "Pareto set empirical attainment (TestFunction4)" \
    --columns "F1,F2" \
    --trials 70 \
    --data_path ./results/TestFunction4/ \
    --filename "MOHOLLM_TestFunction4"

python ./plot/plot_pareto_set_empirical_attainment.py \
    --benchmark SchafferN1 \
    --title "Pareto set empirical attainment (SchafferN1)" \
    --columns "F1,F2" \
    --trials 70 \
    --data_path ./results/SchafferN1/ \
    --filename "MOHOLLM_SchafferN1"


python ./plot/plot_pareto_set_empirical_attainment.py \
    --benchmark SchafferN2 \
    --title "Pareto set empirical attainment (SchafferN2)" \
    --columns "F1,F2" \
    --trials 70 \
    --data_path ./results/SchafferN2/ \
    --filename "MOHOLLM_SchafferN2"


python ./plot/plot_pareto_set_empirical_attainment.py \
    --benchmark Poloni \
    --title "Pareto set empirical attainment (Poloni)" \
    --columns "F1,F2" \
    --trials 70 \
    --data_path ./results/Poloni/ \
    --filename "MOHOLLM_Poloni"

# python ./plot/plot_pareto_set_empirical_attainment.py \
#     --benchmark NB201 \
#     --title "Pareto set empirical attainment(NB201 Titanx)" \
#     --filter "titanx" \
#     --columns "error,latency" \
#     --trials 50 \
#     --data_path ./results/NB201/ \
#     --filename "Titanx_256"\
#     --blacklist "QwQ-32B,Legacy-Prompt,Qwen2.5-32B,mohollm-VP-VIS,mohollm-VP-PE"


# python ./plot/plot_pareto_set_empirical_attainment.py \
#     --benchmark WELDED_BEAM \
#     --title "Pareto set empirical attainment(Welded Beam)" \
#     --filter "welded_beam" \
#     --columns "F1,F2" \
#     --trials 45 \
#     --data_path ./results/WELDED_BEAM/ \
#     --filename "WELDED_BEAM"\
#     --x_lim "-0.005" "0.1"\
#     --y_lim "-5000" "200000.0"

# python ./plot/plot_pareto_set_empirical_attainment.py \
#     --benchmark ZDT \
#     --title "Pareto set empirical attainment(ZDT1)" \
#     --filter "zdt1" \
#     --columns "F1,F2" \
#     --trials 50 \
#     --data_path ./results/ZDT1/ \
#     --filename "ZDT1"

# python ./plot/plot_pareto_set_empirical_attainment.py \
#     --benchmark ZDT \
#     --title "Pareto set empirical attainment(ZDT2)" \
#     --filter "zdt2" \
#     --columns "F1,F2" \
#     --trials 50 \
#     --data_path ./results/ZDT2/ \
#     --filename "ZDT2"

# python ./plot/plot_pareto_set_empirical_attainment.py \
#     --benchmark ZDT \
#     --title "Pareto set empirical attainment(ZDT3)" \
#     --filter "zdt3" \
#     --columns "F1,F2" \
#     --trials 60 \
#     --data_path ./results/ZDT3/ \
#     --filename "ZDT3"\
#     --blacklist "QwQ-32B,Legacy-Prompt,mohollm-Int-ToT (Gemini 1.5 Flash),RS,NSGA2,RSBO,LSBO,GPT-4o Mini"


# python ./plot/plot_pareto_set_empirical_attainment.py \
#     --benchmark ZDT \
#     --title "Pareto set empirical attainment(ZDT4)" \
#     --filter "zdt4" \
#     --columns "F1,F2" \
#     --trials 50 \
#     --data_path ./results/ZDT4/ \
#     --filename "ZDT4"

# python ./plot/plot_pareto_set_empirical_attainment.py \
#     --benchmark ZDT \
#     --title "Pareto set empirical attainment(ZDT6)" \
#     --filter "zdt6" \
#     --columns "F1,F2" \
#     --trials 50 \
#     --data_path ./results/ZDT6/ \
#     --filename "ZDT6"