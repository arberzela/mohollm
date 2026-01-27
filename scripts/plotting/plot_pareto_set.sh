#!/bin/bash

python ./plot/plot_pareto_set.py \
    --benchmark ChankongHaimes \
    --title "Pareto set (ChankongHaimes)" \
    --columns "F1,F2" \
    --trials 70 \
    --data_path ./results/ChankongHaimes/ \
    --filename "MOHOLLM_ChankongHaimes"

python ./plot/plot_pareto_set.py \
    --benchmark TestFunction4 \
    --title "Pareto set (TestFunction4)" \
    --columns "F1,F2" \
    --trials 70 \
    --data_path ./results/TestFunction4/ \
    --filename "MOHOLLM_TestFunction4"

python ./plot/plot_pareto_set.py \
    --benchmark SchafferN1 \
    --title "Pareto set (SchafferN1)" \
    --columns "F1,F2" \
    --trials 70 \
    --data_path ./results/SchafferN1/ \
    --filename "MOHOLLM_SchafferN1"

python ./plot/plot_pareto_set.py \
    --benchmark SchafferN2 \
    --title "Pareto set (SchafferN2)" \
    --columns "F1,F2" \
    --trials 70 \
    --data_path ./results/SchafferN2/ \
    --filename "MOHOLLM_SchafferN2"

python ./plot/plot_pareto_set.py \
    --benchmark Poloni \
    --title "Pareto set (Poloni)" \
    --columns "F1,F2" \
    --trials 70 \
    --data_path ./results/Poloni/ \
    --filename "MOHOLLM_Poloni"

# python ./plot/plot_pareto_set.py \
#     --benchmark NB201 \
#     --title "Pareto set (NB201 Titanx)" \
#     --filter "titanx" \
#     --columns "error,latency" \
#     --trials 50 \
#     --data_path ./results/NB201/ \
#     --filename "Titanx_256"\
#     --blacklist "QwQ-32B,Legacy-Prompt"

# python ./plot/plot_pareto_set.py \
#     --benchmark NB201 \
#     --title "Pareto set (NB201 Titanx)" \
#     --filter "fpga" \
#     --columns "error,latency" \
#     --trials 100 \
#     --data_path ./results/NB201/ \
#     --filename "fpga"


# python ./plot/plot_pareto_set.py \
#     --benchmark ZDT \
#     --title "Pareto set (ZDT3)" \
#     --filter "zdt3" \
#     --columns "f1,f2" \
#     --trials 70 \
#     --data_path ./results/ZDT3/ \
#     --filename "ZDT3"\
#     --blacklist "QwQ-32B,Legacy-Prompt,LSBO,RS,NSGA2,RSBO"

# python ./plot/plot_pareto_set.py \
#     --benchmark WELDED_BEAM \
#     --title "Pareto set (Welded Beam)" \
#     --filter "welded_beam" \
#     --columns "F1,F2" \
#     --trials 45 \
#     --data_path ./results/WELDED_BEAM/ \
#     --filename "WELDED_BEAM"\
#     --x_lim "-0.005" "0.1"\
#     --y_lim "-5000" "50000.0"


# python ./plot/plot_pareto_set.py \
#     --benchmark WELDED_BEAM \
#     --title "Pareto set (Welded Beam)" \
#     --filter "welded_beam" \
#     --columns "F1,F2" \
#     --trials 45 \
#     --data_path ./results/WELDED_BEAM/ \
#     --filename "WELDED_BEAM_y_10000"\
#     --blacklist "LSBO,NSGA2,RSBO,RS"\
#     --x_lim "-0.005" "0.1"\
#     --y_lim "-100" "15000.0"

# python ./plot/plot_pareto_set.py \
#     --benchmark ZDT \
#     --title "Pareto set (ZDT1)" \
#     --filter "zdt1" \
#     --columns "F1,F2" \
#     --trials 50 \
#     --data_path ./results/ZDT1/ \
#     --filename "ZDT1"

# python ./plot/plot_pareto_set.py \
#     --benchmark ZDT \
#     --title "Pareto set (ZDT2)" \
#     --filter "zdt2" \
#     --columns "F1,F2" \
#     --trials 50 \
#     --data_path ./results/ZDT2/ \
#     --filename "ZDT2"

# python ./plot/plot_pareto_set.py \
#     --benchmark ZDT \
#     --title "Pareto set (ZDT3)" \
#     --filter "zdt3" \
#     --columns "F1,F2" \
#     --trials 60 \
#     --data_path ./results/ZDT3/ \
#     --filename "ZDT3"\
#     --blacklist "QwQ-32B,Legacy-Prompt,mohollm-Int-ToT (Gemini 1.5 Flash),RS,NSGA2,RSBO,LSBO,GPT-4o Mini"



# python ./plot/plot_pareto_set.py \
#     --benchmark ZDT \
#     --title "Pareto set (ZDT4)" \
#     --filter "zdt4" \
#     --columns "F1,F2" \
#     --trials 50 \
#     --data_path ./results/ZDT4/ \
#     --filename "ZDT4"

# python ./plot/plot_pareto_set.py \
#     --benchmark ZDT \
#     --title "Pareto set (ZDT6)" \
#     --filter "zdt6" \
#     --columns "F1,F2" \
#     --trials 50 \
#     --data_path ./results/ZDT6/ \
#     --filename "ZDT6"
