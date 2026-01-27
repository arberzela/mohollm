#!/bin/bash

#######################################
# Plots for masterthesis for synthetic plots

# whitelist="CTAEA,EHVI,GDE3,IBEA,MOEAD,MOHOLLM (Gemini 2.0 Flash),mohollm (Gemini 2.0 Flash),NSGA2,NSGA3,PESA2,qLogEHVI,RNSGA2,SMSEMOA,SPEA2,UNSGA3"

# python ./plot/plot_hv_over_time.py \
#     --benchmark DTLZ1 \
#     --title "Hypervolume (DTLZ1)" \
#     --columns "F1,F2" \
#     --trials 50 \
#     --data_path ./results/DTLZ1/ \
#     --filename "MOHOLLM_DTLZ1" \
#     --whitelist "$whitelist" \

# python ./plot/plot_hv_over_time.py \
#     --benchmark DTLZ2 \
#     --title "Hypervolume (DTLZ2)" \
#     --columns "F1,F2" \
#     --trials 50 \
#     --data_path ./results/DTLZ2/ \
#     --filename "MOHOLLM_DTLZ2" \
#     --whitelist "$whitelist" \


# python ./plot/plot_hv_over_time.py \
#     --benchmark BraninCurrin \
#     --title "Hypervolume (BraninCurrin)" \
#     --columns "F1,F2" \
#     --trials 50 \
#     --data_path ./results/BraninCurrin/ \
#     --filename "MOHOLLM_BraninCurrin" \
#     --whitelist "$whitelist" \


# python ./plot/plot_hv_over_time.py \
#     --benchmark ChankongHaimes \
#     --title "Hypervolume (ChankongHaimes)" \
#     --columns "F1,F2" \
#     --trials 50 \
#     --data_path ./results/ChankongHaimes/ \
#     --filename "MOHOLLM_ChankongHaimes" \
#     --whitelist  "$whitelist" \
#     --y_lim "0.3" "0.75"


# python ./plot/plot_hv_over_time.py \
#     --benchmark GMM \
#     --title "Hypervolume (GMM)" \
#     --columns "F1,F2" \
#     --trials 50 \
#     --data_path ./results/GMM/ \
#     --filename "MOHOLLM_GMM" \
#     --whitelist "$whitelist" \

# python ./plot/plot_hv_over_time.py \
#     --benchmark Poloni \
#     --title "Hypervolume (Poloni)" \
#     --columns "F1,F2" \
#     --trials 50 \
#     --data_path ./results/Poloni/ \
#     --filename "MOHOLLM_Poloni" \
#     --whitelist "$whitelist" \
#     --y_lim "0.8" "1.0005" \

# python ./plot/plot_hv_over_time.py \
#     --benchmark SchafferN1 \
#     --title "Hypervolume (SchafferN1)" \
#     --columns "F1,F2" \
#     --trials 50 \
#     --data_path ./results/SchafferN1/ \
#     --filename "MOHOLLM_SchafferN1" \
#     --whitelist "$whitelist" \
#     --y_lim "0.985" "1.0005"

# python ./plot/plot_hv_over_time.py \
#     --benchmark SchafferN2 \
#     --title "Hypervolume (SchafferN2)" \
#     --columns "F1,F2" \
#     --trials 50 \
#     --data_path ./results/SchafferN2/ \
#     --filename "MOHOLLM_SchafferN2" \
#     --whitelist "$whitelist" \
#     --y_lim "0.8" "1.0005"


# python ./plot/plot_hv_over_time.py \
#     --benchmark TestFunction4 \
#     --title "Hypervolume (TestFunction4)" \
#     --columns "F1,F2" \
#     --trials 50 \
#     --data_path ./results/TestFunction4/ \
#     --filename "MOHOLLM_TestFunction4" \
#     --whitelist "$whitelist" \
#     --y_lim "0.65" "1.0005"

# python ./plot/plot_hv_over_time.py \
#     --benchmark ToyRobust \
#     --title "Hypervolume (ToyRobust)" \
#     --columns "F1,F2" \
#     --trials 50 \
#     --data_path ./results/ToyRobust/ \
#     --filename "MOHOLLM_ToyRobust" \
#     --whitelist "$whitelist" \
#     --y_lim "0.85" "1" \

# #######################################

# # Real world benchmarks
# python ./plot/plot_hv_over_time.py \
#     --benchmark Penicillin \
#     --title "Hypervolume (Penicillin)" \
#     --columns "F1,F2,F3" \
#     --trials 50 \
#     --data_path ./results/Penicillin/ \
#     --filename "MOHOLLM_Penicillin" \
#     --whitelist  "$whitelist"

# python ./plot/plot_hv_over_time.py \
#     --benchmark CarSideImpact \
#     --title "Hypervolume (CarSideImpact)" \
#     --columns "F1,F2,F3,F4" \
#     --trials 50 \
#     --data_path ./results/CarSideImpact/ \
#     --filename "MOHOLLM_CarSideImpact" \
#     --whitelist "$whitelist" \

# python ./plot/plot_hv_over_time.py \
#     --benchmark VehicleSafety \
#     --title "Hypervolume (VehicleSafety)" \
#     --columns "F1,F2,F3" \
#     --trials 50 \
#     --data_path ./results/VehicleSafety/ \
#     --filename "MOHOLLM_VehicleSafety" \
#     --whitelist "$whitelist" \


benchmarks=("cifar10" "cifar100" "imagenet16")

devices=(
    "fpga_latency" "pixel3_latency" "raspi4_latency" "eyeriss_latency" "pixel2_latency"
    "1080ti_1_latency" "1080ti_32_latency" "1080ti_256_latency"
    "2080ti_1_latency" "2080ti_32_latency" "2080ti_256_latency"
    "titanx_1_latency" "titanx_32_latency" "titanx_256_latency"
    "titanxp_1_latency" "titanxp_32_latency" "titanxp_256_latency"
    "titan_rtx_1_latency" "titan_rtx_32_latency" "titan_rtx_256_latency"
    "essential_ph_1_latency" "gold_6226_latency" "gold_6240_latency"
    "samsung_a50_latency" "samsung_s7_latency" "silver_4114_latency" "silver_4210r_latency"
)

# whitelist="MOHOLLM (Gemini 2.0 Flash),mohollm (Gemini 2.0 Flash),MOHOLLM (Gemini 2.0 Flash) (Context),MOHOLLM (Qwen3-4B) (Context)"
whitelist="CTAEA,EHVI,GDE3,IBEA,MOEAD,MOHOLLM (Gemini 2.0 Flash),mohollm (Gemini 2.0 Flash),NSGA2,NSGA3,PESA2,qLogEHVI,RNSGA2,SMSEMOA,SPEA2,UNSGA3"

# Loop over all device folders in results/NB201 and plot for each
for benchmark in "${benchmarks[@]}"; do
    for device in "${devices[@]}"; do
        echo "./results/NB201/$benchmark/$device"
        python ./plot/plot_hv_over_time.py \
            --benchmark NB201 \
            --title "Hypervolume $benchmark ($device)" \
            --columns "F1,F2" \
            --trials 50 \
            --data_path "./results/NB201/$benchmark/$device" \
            --filename "$benchmark/$device" \
            --normalization_method "nb201" \
            --nb201_device_metric "$device" \
            --y_lim "0.5" "1.0005" \
            --whitelist "$whitelist"
    done
done







# python ./plot/plot_hv_over_time.py \
#     --benchmark Penicillin \
#     --title "Hypervolume (Penicillin)" \
#     --columns "F1,F2" \
#     --trials 50 \
#     --data_path ./results/Penicillin/ \
#     --filename "MOHOLLM_Penicillin" \
#     --whitelist "$whitelist" \


# python ./plot/plot_hv_over_time.py \
#     --benchmark Penicillin \
#     --title "Hypervolume (Penicillin)" \
#     --columns "F1,F2" \
#     --trials 50 \
#     --data_path ./results/Penicillin/ \
#     --filename "MOHOLLM_Penicillin" \
#     --whitelist "CTAEA,EpsMOAE,EpsNSGA2,GD3,IBEA,MOEAD,MOHOLLM (Gemini 2.0 Flash),MOHOLLM (Gemini 2.0 Flash) (Context),mohollm (Gemini 2.0 Flash),NSGA2,NSGA3,PESA2,RNSGA2,RVEA,SMSEMOA,SPEA2,UNSGA3"

# # Ablations: Different components
# python ./plot/plot_hv_over_time.py \
#     --benchmark Penicillin \
#     --title "Hypervolume (Penicillin)" \
#     --columns "F1,F2" \
#     --trials 50 \
#     --data_path ./results/Penicillin/ \
#     --filename "MOHOLLM_Penicillin_ablations_components" \
#     --whitelist "MOHOLLM (Gemini 2.0 Flash),MOHOLLM (Gemini 2.0 Flash) (Context),mohollm (Gemini 2.0 Flash),MOHOLLM + GP (Gemini 2.0 Flash) (Context),MOHOLLM + TabPFN (Gemini 2.0 Flash) (Context),RS (KD-Partitioning) + LLM Surrogate (Gemini 2.0 Flash) (Context),RS + LLM Surrogate (Gemini 2.0 Flash) (Context)"






# # Ablations: Penicillin
# # ALPHA_MAX
# python ./plot/plot_hv_over_time.py \
#     --benchmark Penicillin \
#     --title "Hypervolume (Penicillin) alpha_max" \
#     --columns "F1,F2,F3" \
#     --trials 35 \
#     --data_path ./results/Penicillin/ \
#     --filename "MOHOLLM_Penicillin_alpha_max" \
#     --whitelist "MOHOLLM (Gemini 2.0 Flash) (Context) (Default),MOHOLLM (Gemini 2.0 Flash) (Context) - alpha_max=0,MOHOLLM (Gemini 2.0 Flash) (Context) - alpha_max=0.4,MOHOLLM (Gemini 2.0 Flash) (Context) - alpha_max=0.8,MOHOLLM (Gemini 2.0 Flash) (Context) - alpha_max=2.0"

# # CANDIDATES_PER_REQUEST
# python ./plot/plot_hv_over_time.py \
#     --benchmark Penicillin \
#     --title "Hypervolume (Penicillin) candidates_per_request" \
#     --columns "F1,F2,F3" \
#     --trials 35 \
#     --data_path ./results/Penicillin/ \
#     --filename "MOHOLLM_Penicillin_candidates_per_request" \
#     --whitelist "MOHOLLM (Gemini 2.0 Flash) (Context) (Default),MOHOLLM (Gemini 2.0 Flash) (Context) - candidates_per_request=1,MOHOLLM (Gemini 2.0 Flash) (Context) - candidates_per_request=3,MOHOLLM (Gemini 2.0 Flash) (Context) - candidates_per_request=7"

# # Leaf size (m0)
# python ./plot/plot_hv_over_time.py \
#     --benchmark Penicillin \
#     --title "Hypervolume (Penicillin) (leaf size)" \
#     --columns "F1,F2,F3" \
#     --trials 35 \
#     --data_path ./results/Penicillin/ \
#     --filename "MOHOLLM_Penicillin_leaf_size" \
#     --whitelist "MOHOLLM (Gemini 2.0 Flash) (Context) (Default),MOHOLLM (Gemini 2.0 Flash) (Context) - m0=1,MOHOLLM (Gemini 2.0 Flash) (Context) - m0=3,MOHOLLM (Gemini 2.0 Flash) (Context) - m0=10"

# # ALL ABLATIONS IN ONE
# python ./plot/plot_hv_over_time.py \
#     --benchmark Penicillin \
#     --title "Hypervolume (Penicillin)" \
#     --columns "F1,F2,F3" \
#     --trials 35 \
#     --data_path ./results/Penicillin/ \
#     --filename "MOHOLLM_Penicillin_all_test" \
#     --whitelist "MOHOLLM (Gemini 2.0 Flash) (Context) (Default),MOHOLLM (Gemini 2.0 Flash) (Context) - m0=1,MOHOLLM (Gemini 2.0 Flash) (Context) - m0=3,MOHOLLM (Gemini 2.0 Flash) (Context) - m0=10,MOHOLLM (Gemini 2.0 Flash) (Context) (Default),MOHOLLM (Gemini 2.0 Flash) (Context) - candidates_per_request=1,MOHOLLM (Gemini 2.0 Flash) (Context) - candidates_per_request=3,MOHOLLM (Gemini 2.0 Flash) (Context) - candidates_per_request=7,MOHOLLM (Gemini 2.0 Flash) (Context) (Default),MOHOLLM (Gemini 2.0 Flash) (Context) - alpha_max=0,MOHOLLM (Gemini 2.0 Flash) (Context) - alpha_max=0.4,MOHOLLM (Gemini 2.0 Flash) (Context) - alpha_max=0.8,MOHOLLM (Gemini 2.0 Flash) (Context) - alpha_max=2.0,mohollm (Gemini 2.0 Flash),MOHOLLM (Gemini 2.0 Flash),MOHOLLM (Gemini 2.0 Flash) (Context),mohollm (Gemini 2.0 Flash) - alpha_prompt=0.0,mohollm (Gemini 2.0 Flash) - alpha_prompt=0.5,mohollm (Gemini 2.0 Flash) - alpha_prompt=0.25,mohollm (Gemini 2.0 Flash) - alpha_prompt=0.75,mohollm (Gemini 2.0 Flash) - alpha_prompt=1.0,MOHOLLM (Gemini 2.0 Flash) (Context)"




# # Ablations prompt instructions
# python ./plot/plot_hv_over_time.py \
#     --benchmark "Penicillin" \
#     --title "Hypervolume Penicillin (Different Prompt Instructions)" \
#     --data_path "./results/Penicillin/" \
#     --columns "F1,F2,F3" \
#     --filename "sampler/penicillin_prompt_test2" \
#     --trials 50 \
#     --whitelist "mohollm (Gemini 2.0 Flash) - alpha_prompt=0.0,mohollm (Gemini 2.0 Flash) - alpha_prompt=0.5,mohollm (Gemini 2.0 Flash) - alpha_prompt=0.25,mohollm (Gemini 2.0 Flash) - alpha_prompt=0.75,mohollm (Gemini 2.0 Flash) - alpha_prompt=1.0"



# # Ablations: Vehicle Safety
# # ALPHA_MAX
# python ./plot/plot_hv_over_time.py \
#     --benchmark VehicleSafety \
#     --title "Hypervolume (VehicleSafety) alpha_max" \
#     --columns "F1,F2,F3" \
#     --trials 35 \
#     --data_path ./results/VehicleSafety/ \
#     --filename "MOHOLLM_VehicleSafety_alpha_max" \
#     --whitelist "MOHOLLM (Gemini 2.0 Flash) (Context) (Default),MOHOLLM (Gemini 2.0 Flash) (Context) - alpha_max=0,MOHOLLM (Gemini 2.0 Flash) (Context) - alpha_max=0.4,MOHOLLM (Gemini 2.0 Flash) (Context) - alpha_max=0.8,MOHOLLM (Gemini 2.0 Flash) (Context) - alpha_max=2.0"

# # CANDIDATES_PER_REQUEST
# python ./plot/plot_hv_over_time.py \
#     --benchmark VehicleSafety \
#     --title "Hypervolume (VehicleSafety) candidates_per_request" \
#     --columns "F1,F2,F3" \
#     --trials 35 \
#     --data_path ./results/VehicleSafety/ \
#     --filename "MOHOLLM_Vehicle_Safety_candidates_per_request" \
#     --whitelist "MOHOLLM (Gemini 2.0 Flash) (Context) (Default),MOHOLLM (Gemini 2.0 Flash) (Context) - candidates_per_request=1,MOHOLLM (Gemini 2.0 Flash) (Context) - candidates_per_request=3,MOHOLLM (Gemini 2.0 Flash) (Context) - candidates_per_request=7"




# # Leaf size (m0)
# python ./plot/plot_hv_over_time.py \
#     --benchmark VehicleSafety \
#     --title "Hypervolume (VehicleSafety) (leaf size)" \
#     --columns "F1,F2,F3" \
#     --trials 35 \
#     --data_path ./results/VehicleSafety/ \
#     --filename "MOHOLLM_Vehicle_Safety_leaf_size" \
#     --whitelist "MOHOLLM (Gemini 2.0 Flash) (Context) (Default),MOHOLLM (Gemini 2.0 Flash) (Context) - m0=1,MOHOLLM (Gemini 2.0 Flash) (Context) - m0=3,MOHOLLM (Gemini 2.0 Flash) (Context) - m0=10"

# # ALL ABLATIONS IN ONE
# python ./plot/plot_hv_over_time.py \
#     --benchmark VehicleSafety \
#     --title "Hypervolume (VehicleSafety)" \
#     --columns "F1,F2,F3" \
#     --trials 35 \
#     --data_path ./results/VehicleSafety/ \
#     --filename "MOHOLLM_VehicleSafety_all" \
#     --whitelist "MOHOLLM (Gemini 2.0 Flash) (Context) (Default),MOHOLLM (Gemini 2.0 Flash) (Context) - m0=1,MOHOLLM (Gemini 2.0 Flash) (Context) - m0=3,MOHOLLM (Gemini 2.0 Flash) (Context) - m0=10,MOHOLLM (Gemini 2.0 Flash) (Context) (Default),MOHOLLM (Gemini 2.0 Flash) (Context) - candidates_per_request=1,MOHOLLM (Gemini 2.0 Flash) (Context) - candidates_per_request=3,MOHOLLM (Gemini 2.0 Flash) (Context) - candidates_per_request=7,MOHOLLM (Gemini 2.0 Flash) (Context) (Default),MOHOLLM (Gemini 2.0 Flash) (Context) - alpha_max=0,MOHOLLM (Gemini 2.0 Flash) (Context) - alpha_max=0.4,MOHOLLM (Gemini 2.0 Flash) (Context) - alpha_max=0.8,MOHOLLM (Gemini 2.0 Flash) (Context) - alpha_max=2.0,mohollm (Gemini 2.0 Flash),MOHOLLM (Gemini 2.0 Flash),MOHOLLM (Gemini 2.0 Flash) (Context)"


# # Ablations: CarSideImpact
# # ALPHA_MAX
# python ./plot/plot_hv_over_time.py \
#     --benchmark CarSideImpact \
#     --title "Hypervolume (CarSideImpact) alpha_max" \
#     --columns "F1,F2,F3,F4" \
#     --trials 35 \
#     --data_path ./results/CarSideImpact/ \
#     --filename "MOHOLLM_CarSideImpact_alpha_max" \
#     --whitelist "MOHOLLM (Gemini 2.0 Flash) (Context),MOHOLLM (Gemini 2.0 Flash) (Context) - alpha_max=0,MOHOLLM (Gemini 2.0 Flash) (Context) - alpha_max=0.4,MOHOLLM (Gemini 2.0 Flash) (Context) - alpha_max=0.8,MOHOLLM (Gemini 2.0 Flash) (Context) - alpha_max=2.0"

# # CANDIDATES_PER_REQUEST
# python ./plot/plot_hv_over_time.py \
#     --benchmark CarSideImpact \
#     --title "Hypervolume (CarSideImpact) candidates_per_request" \
#     --columns "F1,F2,F3,F4" \
#     --trials 35 \
#     --data_path ./results/CarSideImpact/ \
#     --filename "MOHOLLM_CarSideImpact_candidates_per_request" \
#     --whitelist "MOHOLLM (Gemini 2.0 Flash) (Context),MOHOLLM (Gemini 2.0 Flash) (Context) - candidates_per_request=1,MOHOLLM (Gemini 2.0 Flash) (Context) - candidates_per_request=3,MOHOLLM (Gemini 2.0 Flash) (Context) - candidates_per_request=7"


# # Partitions per trial
# python ./plot/plot_hv_over_time.py \
#     --benchmark CarSideImpact \
#     --title "Hypervolume (CarSideImpact) partitions_per_trial" \
#     --columns "F1,F2,F3,F4" \
#     --trials 35 \
#     --data_path ./results/CarSideImpact/ \
#     --filename "MOHOLLM_CarSideImpact_partitions_per_trial" \
#     --whitelist "MOHOLLM (Gemini 2.0 Flash) (Context),MOHOLLM (Gemini 2.0 Flash) (Context) - partitions_per_trial=1,MOHOLLM (Gemini 2.0 Flash) (Context) - partitions_per_trial=3,MOHOLLM (Gemini 2.0 Flash) (Context) - partitions_per_trial=7"

# # Leaf size (m0)
# python ./plot/plot_hv_over_time.py \
#     --benchmark CarSideImpact \
#     --title "Hypervolume (CarSideImpact) (leaf size)" \
#     --columns "F1,F2,F3,F4" \
#     --trials 35 \
#     --data_path ./results/CarSideImpact/ \
#     --filename "MOHOLLM_CarSideImpact_leaf_size" \
#     --whitelist "MOHOLLM (Gemini 2.0 Flash) (Context),MOHOLLM (Gemini 2.0 Flash) (Context) - m0=1,MOHOLLM (Gemini 2.0 Flash) (Context) - m0=3,MOHOLLM (Gemini 2.0 Flash) (Context) - m0=10"

# # ALL ABLATIONS IN ONE
# python ./plot/plot_hv_over_time.py \
#     --benchmark CarSideImpact \
#     --title "Hypervolume (CarSideImpact)" \
#     --columns "F1,F2,F3,F4" \
#     --trials 35 \
#     --data_path ./results/CarSideImpact/ \
#     --filename "MOHOLLM_CarSideImpact_all_test" \
#     --whitelist "MOHOLLM (Gemini 2.0 Flash) (Context),MOHOLLM (Gemini 2.0 Flash) (Context) - m0=1,MOHOLLM (Gemini 2.0 Flash) (Context) - m0=3,MOHOLLM (Gemini 2.0 Flash) (Context) - m0=10,MOHOLLM (Gemini 2.0 Flash) (Context) (Default),MOHOLLM (Gemini 2.0 Flash) (Context) - candidates_per_request=1,MOHOLLM (Gemini 2.0 Flash) (Context) - candidates_per_request=3,MOHOLLM (Gemini 2.0 Flash) (Context) - candidates_per_request=7,MOHOLLM (Gemini 2.0 Flash) (Context) (Default),MOHOLLM (Gemini 2.0 Flash) (Context) - alpha_max=0,MOHOLLM (Gemini 2.0 Flash) (Context) - alpha_max=0.4,MOHOLLM (Gemini 2.0 Flash) (Context) - alpha_max=0.8,MOHOLLM (Gemini 2.0 Flash) (Context) - alpha_max=2.0,mohollm (Gemini 2.0 Flash),MOHOLLM (Gemini 2.0 Flash),MOHOLLM (Gemini 2.0 Flash) (Context),mohollm (Gemini 2.0 Flash) - alpha_prompt=0.0,mohollm (Gemini 2.0 Flash) - alpha_prompt=0.5,mohollm (Gemini 2.0 Flash) - alpha_prompt=0.25,mohollm (Gemini 2.0 Flash) - alpha_prompt=0.75,mohollm (Gemini 2.0 Flash) - alpha_prompt=1.0,MOHOLLM (Gemini 2.0 Flash) (Context)"
