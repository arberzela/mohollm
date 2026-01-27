
#!/bin/bash

# Plots for masterthesis for synthetic plots (subplots)

# whitelist="CTAEA,EHVI,GDE3,IBEA,MOEAD,NSGA2,NSGA3,PESA2,qLogEHVI,RNSGA2,SMSEMOA,SPEA2,UNSGA3,mohollm (Gemini 2.0 Flash),MOHOLLM (Gemini 2.0 Flash),MOHOLLM (Gemini 2.0 Flash) (Context)"

# # # synthetic
# python3 ./plot/plot_hv_over_time_subplots.py \
# 	--benchmarks DTLZ1 DTLZ2 DTLZ3 BraninCurrin ChankongHaimes GMM Poloni SchafferN1 SchafferN2 TestFunction4 ToyRobust Kursawe \
# 	--titles "DTLZ1" "DTLZ2" "DTLZ3" "BraninCurrin" "ChankongHaimes" "GMM" "Poloni" "SchafferN1" "SchafferN2" "TestFunction4" "ToyRobust" "Kursawe" \
# 	--trials 50 \
#     --filename "synthetic_hv" \
# 	--data_paths ./results/DTLZ1/ ./results/DTLZ2/ ./results/DTLZ3/ ./results/BraninCurrin/ ./results/ChankongHaimes/ ./results/GMM/ ./results/Poloni/ ./results/SchafferN1/ ./results/SchafferN2/ ./results/TestFunction4/ ./results/ToyRobust/ ./results/Kursawe/ \
# 	--whitelists "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" \
#     --normalization_method "reference_point"


# whitelist="EHVI,GDE3,MOEAD,NSGA2,NSGA3,qLogEHVI,mohollm (Gemini 2.0 Flash),MOHOLLM (Gemini 2.0 Flash),MOHOLLM (Gemini 2.0 Flash) (Context)"
# # # synthetic
# python3 ./plot/plot_hv_over_time_subplots.py \
# 	--benchmarks BraninCurrin ChankongHaimes Kursawe \
# 	--titles "BraninCurrin" "ChankongHaimes" "Kursawe" \
# 	--trials 50 \
#     --filename "synthetic_hv_presentation" \
# 	--data_paths ./results/BraninCurrin/ ./results/ChankongHaimes/ ./results/Kursawe/ \
# 	--whitelists "$whitelist" "$whitelist" "$whitelist" \
#     --normalization_method "reference_point"


# whitelist="EHVI,GDE3,MOEAD,NSGA2,NSGA3,qLogEHVI,mohollm (Gemini 2.0 Flash),MOHOLLM (Gemini 2.0 Flash),MOHOLLM (Gemini 2.0 Flash) (Context)"
# # synthetic
# python3 ./plot/plot_hv_over_time_subplots.py \
# 	--benchmarks DTLZ1 DTLZ2 DTLZ3 \
# 	--titles "DTLZ1" "DTLZ2" "DTLZ3" \
# 	--trials 50 \
#     --filename "synthetic_hv_presentation_1" \
# 	--data_paths ./results/DTLZ1/ ./results/DTLZ2/ ./results/DTLZ3/  \
# 	--whitelists "$whitelist" "$whitelist" "$whitelist" "$whitelist" \
#     --normalization_method "reference_point"


# whitelist="EHVI,GDE3,MOEAD,NSGA2,NSGA3,qLogEHVI,mohollm (Gemini 2.0 Flash),MOHOLLM (Gemini 2.0 Flash),MOHOLLM (Gemini 2.0 Flash) (Context)"
# # synthetic
# python3 ./plot/plot_hv_over_time_subplots.py \
# 	--benchmarks GMM Poloni SchafferN1 \
# 	--titles "GMM" "Poloni" "SchafferN1" \
# 	--trials 50 \
#     --filename "synthetic_hv_presentation_2" \
# 	--data_paths ./results/GMM/ ./results/Poloni/ ./results/SchafferN1/  \
# 	--whitelists "$whitelist" "$whitelist" "$whitelist" \
#     --normalization_method "reference_point"

# python3 ./plot/plot_hv_over_time_subplots.py \
# 	--benchmarks SchafferN2 TestFunction4 ToyRobust \
# 	--titles "SchafferN2" "TestFunction4" "ToyRobust" \
# 	--trials 50 \
#     --filename "synthetic_hv_presentation_3" \
# 	--data_paths ./results/SchafferN2/ ./results/TestFunction4/ ./results/ToyRobust/ \
# 	--whitelists "$whitelist" "$whitelist" "$whitelist" \
#     --normalization_method "reference_point"

# whitelist="EHVI,GDE3,MOEAD,NSGA2,NSGA3,qLogEHVI,mohollm (Gemini 2.0 Flash),MOHOLLM (Gemini 2.0 Flash) (Context)"
# python3 ./plot/plot_hv_over_time_subplots.py \
# 	--benchmarks Penicillin CarSideImpact VehicleSafety \
# 	--titles "Penicillin" "CarSideImpact" "VehicleSafety" \
# 	--trials 50 \
#     --filename "real_world_hv_presentation" \
# 	--data_paths ./results/Penicillin/ ./results/CarSideImpact/ ./results/VehicleSafety/ \
# 	--whitelists "$whitelist" "$whitelist" "$whitelist" \
#     --normalization_method "reference_point"

# whitelist="CTAEA,EHVI,GDE3,IBEA,MOEAD,MOHOLLM (Gemini 2.0 Flash) (Context),mohollm (Gemini 2.0 Flash),NSGA2,NSGA3,PESA2,qLogEHVI,RNSGA2,SMSEMOA,SPEA2,UNSGA3"
# real world
# python3 ./plot/plot_hv_over_time_subplots.py \
# 	--benchmarks Penicillin CarSideImpact VehicleSafety \
# 	--titles "Penicillin" "CarSideImpact" "VehicleSafety" \
# 	--trials 50 \
#     --filename "real_world_hv_paper" \
# 	--data_paths ./results/Penicillin/ ./results/CarSideImpact/ ./results/VehicleSafety/ \
# 	--whitelists "$whitelist" "$whitelist" "$whitelist" \
#     --normalization_method "reference_point"

# # ## ONLY LLM AND MOHOLLM

# whitelist="mohollm (Gemini 2.0 Flash),MOHOLLM (Gemini 2.0 Flash),MOHOLLM (Gemini 2.0 Flash) (Context)"

# # synthetic
# python3 ./plot/plot_hv_over_time_subplots.py \
# 	--benchmarks DTLZ1 DTLZ2 DTLZ3 BraninCurrin ChankongHaimes GMM Poloni SchafferN1 SchafferN2 TestFunction4 ToyRobust Kursawe \
# 	--titles "DTLZ1" "DTLZ2" "DTLZ3" "BraninCurrin" "ChankongHaimes" "GMM" "Poloni" "SchafferN1" "SchafferN2" "TestFunction4" "ToyRobust" "Kursawe" \
# 	--trials 50 \
#     --filename "synthetic_hv_only_llm" \
# 	--data_paths ./results/DTLZ1/ ./results/DTLZ2/ ./results/DTLZ3/ ./results/BraninCurrin/ ./results/ChankongHaimes/ ./results/GMM/ ./results/Poloni/ ./results/SchafferN1/ ./results/SchafferN2/ ./results/TestFunction4/ ./results/ToyRobust/ ./results/Kursawe/ \
# 	--whitelists "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" \
#     --normalization_method "reference_point"



# whitelist="MOHOLLM (Gemini 2.0 Flash) (Context),mohollm (Gemini 2.0 Flash)"
# # real world
# python3 ./plot/plot_hv_over_time_subplots.py \
# 	--benchmarks Penicillin CarSideImpact VehicleSafety \
# 	--titles "Penicillin" "CarSideImpact" "VehicleSafety" \
# 	--trials 50 \
#     --filename "real_world_hv_only_llm" \
# 	--data_paths ./results/Penicillin/ ./results/CarSideImpact/ ./results/VehicleSafety/ \
# 	--whitelists "$whitelist" "$whitelist" "$whitelist" \
#     --normalization_method "reference_point"




# ############################## ABLATIONS:

# 1. Leaf size
# whitelist="MOHOLLM (Gemini 2.0 Flash) (Context),MOHOLLM (Gemini 2.0 Flash) (Context) - m0=1,MOHOLLM (Gemini 2.0 Flash) (Context) - m0=3,MOHOLLM (Gemini 2.0 Flash) (Context) - m0=10"
# python3 ./plot/plot_hv_over_time_subplots.py \
# 	--benchmarks Penicillin CarSideImpact VehicleSafety \
# 	--titles "Penicillin" "CarSideImpact" "VehicleSafety" \
# 	--trials 35 \
#     --filename "real_world_ablations_leaf_size" \
# 	--data_paths ./results/Penicillin/ ./results/CarSideImpact/ ./results/VehicleSafety/ \
# 	--whitelists "$whitelist" "$whitelist" "$whitelist" \
#     --normalization_method "reference_point" \
#     --num_seeds 5

# # 2. alpha_max
# whitelist="MOHOLLM (Gemini 2.0 Flash) (Context),MOHOLLM (Gemini 2.0 Flash) (Context) - alpha_max=0,MOHOLLM (Gemini 2.0 Flash) (Context) - alpha_max=0.4,MOHOLLM (Gemini 2.0 Flash) (Context) - alpha_max=0.8,MOHOLLM (Gemini 2.0 Flash) (Context) - alpha_max=2.0"
# python3 ./plot/plot_hv_over_time_subplots.py \
# 	--benchmarks Penicillin CarSideImpact VehicleSafety \
# 	--titles "Penicillin" "CarSideImpact" "VehicleSafety" \
# 	--trials 35 \
#     --filename "real_world_ablations_alpha_max" \
# 	--data_paths ./results/Penicillin/ ./results/CarSideImpact/ ./results/VehicleSafety/ \
# 	--whitelists "$whitelist" "$whitelist" "$whitelist" \
#     --normalization_method "reference_point" \
#     --num_seeds 5


# # 3. Candidates per request
# whitelist="MOHOLLM (Gemini 2.0 Flash) (Context),MOHOLLM (Gemini 2.0 Flash) (Context) - candidates_per_request=1,MOHOLLM (Gemini 2.0 Flash) (Context) - candidates_per_request=3,MOHOLLM (Gemini 2.0 Flash) (Context) - candidates_per_request=7"
# python3 ./plot/plot_hv_over_time_subplots.py \
# 	--benchmarks Penicillin CarSideImpact VehicleSafety \
# 	--titles "Penicillin" "CarSideImpact" "VehicleSafety" \
# 	--trials 35 \
#     --filename "real_world_ablations_candidates_per_request" \
# 	--data_paths ./results/Penicillin/ ./results/CarSideImpact/ ./results/VehicleSafety/ \
# 	--whitelists "$whitelist" "$whitelist" "$whitelist" \
#     --normalization_method "reference_point" \
#     --num_seeds 5

# # 4. Partitions per trial
# whitelist="MOHOLLM (Gemini 2.0 Flash) (Context),MOHOLLM (Gemini 2.0 Flash) (Context) - partitions_per_trial=1,MOHOLLM (Gemini 2.0 Flash) (Context) - partitions_per_trial=3,MOHOLLM (Gemini 2.0 Flash) (Context) - partitions_per_trial=7"
# python3 ./plot/plot_hv_over_time_subplots.py \
# 	--benchmarks Penicillin CarSideImpact VehicleSafety \
# 	--titles "Penicillin" "CarSideImpact" "VehicleSafety" \
# 	--trials 35 \
#     --filename "real_world_ablations_partitions_per_trial" \
# 	--data_paths ./results/Penicillin/ ./results/CarSideImpact/ ./results/VehicleSafety/ \
# 	--whitelists "$whitelist" "$whitelist" "$whitelist" \
#     --normalization_method "reference_point" \
#     --num_seeds 5


# # 5. Ablations on different components
# whitelist="MOHOLLM (Gemini 2.0 Flash) (Context),MOHOLLM + TabPFN (Gemini 2.0 Flash) (Context),MOHOLLM + GP (BoTorch) (Gemini 2.0 Flash) (Context)"
# python3 ./plot/plot_hv_over_time_subplots.py \
# 	--benchmarks Penicillin CarSideImpact VehicleSafety \
# 	--titles "Penicillin" "CarSideImpact" "VehicleSafety" \
# 	--trials 50 \
#     --filename "real_world_ablations_components_surrogate_model" \
# 	--data_paths ./results/Penicillin/ ./results/CarSideImpact/ ./results/VehicleSafety/ \
# 	--whitelists "$whitelist" "$whitelist" "$whitelist" \
#     --normalization_method "reference_point"

# whitelist="MOHOLLM (Gemini 2.0 Flash) (Context),RS (KD-Partitioning) + LLM Surrogate (Gemini 2.0 Flash) (Context),RS + LLM Surrogate (Gemini 2.0 Flash) (Context),qLogEHVI (KD-Partitioning) + LLM Surrogate (Gemini 2.0 Flash) (Context)"
# python3 ./plot/plot_hv_over_time_subplots.py \
# 	--benchmarks Penicillin CarSideImpact VehicleSafety \
# 	--titles "Penicillin" "CarSideImpact" "VehicleSafety" \
# 	--trials 50 \
#     --filename "real_world_ablations_components_candidate_sampler" \
# 	--data_paths ./results/Penicillin/ ./results/CarSideImpact/ ./results/VehicleSafety/ \
# 	--whitelists "$whitelist" "$whitelist" "$whitelist" \
#     --normalization_method "reference_point"

# whitelist="MOHOLLM (Gemini 2.0 Flash) (Context),mohollm (Gemini 2.0 Flash),RS (KD-Partitioning) + LLM Surrogate (Gemini 2.0 Flash) (Context),RS + LLM Surrogate (Gemini 2.0 Flash) (Context),"
# python3 ./plot/plot_hv_over_time_subplots.py \
# 	--benchmarks Penicillin CarSideImpact VehicleSafety \
# 	--titles "Penicillin" "CarSideImpact" "VehicleSafety" \
# 	--trials 50 \
#     --filename "real_world_ablations_components_candidate_sampler_presentation" \
# 	--data_paths ./results/Penicillin/ ./results/CarSideImpact/ ./results/VehicleSafety/ \
# 	--whitelists "$whitelist" "$whitelist" "$whitelist" \
#     --normalization_method "reference_point"


# whitelist="MOHOLLM (Gemini 2.0 Flash) (Context),RS (KD-Partitioning) + LLM Surrogate (Gemini 2.0 Flash) (Context),RS + LLM Surrogate (Gemini 2.0 Flash) (Context),qLogEHVI (KD-Partitioning) + LLM Surrogate (Gemini 2.0 Flash) (Context),qLogEHVI"
# python3 ./plot/plot_hv_over_time_subplots.py \
# 	--benchmarks Penicillin VehicleSafety \
# 	--titles "Penicillin"  "VehicleSafety" \
# 	--trials 50 \
#     --filename "real_world_ablations_components_candidate_sampler" \
# 	--data_paths ./results/Penicillin/ ./results/VehicleSafety/ \
# 	--whitelists "$whitelist" "$whitelist" \
#     --normalization_method "reference_point"

# whitelist="MOHOLLM (Gemini 2.0 Flash) (Context),qLogEHVI (KD-Partitioning) + LLM Surrogate (Gemini 2.0 Flash) (Context),qLogEHVI"
# python3 ./plot/plot_hv_over_time_subplots.py \
# 	--benchmarks Penicillin CarSideImpact VehicleSafety \
# 	--titles "Penicillin" "CarSideImpact" "VehicleSafety" \
# 	--trials 50 \
#     --filename "real_world_ablations_components_candidate_sampler_appendix" \
# 	--data_paths ./results/Penicillin/ ./results/CarSideImpact/ ./results/VehicleSafety/ \
# 	--whitelists "$whitelist" "$whitelist" "$whitelist" \
#     --normalization_method "reference_point"




# # just a test
# whitelist="CTAEA,EHVI,GDE3,IBEA,MOEAD,MOHOLLM (Gemini 2.0 Flash) (Context),NSGA2,NSGA3,PESA2,qLogEHVI,RNSGA2,SMSEMOA,SPEA2,UNSGA3,qLogEHVI (KD-Partitioning) + LLM Surrogate (Gemini 2.0 Flash) (Context)"
# # real world
# python3 ./plot/plot_hv_over_time_subplots.py \
# 	--benchmarks Penicillin CarSideImpact VehicleSafety \
# 	--titles "Penicillin" "CarSideImpact" "VehicleSafety" \
# 	--trials 50 \
#     --filename "real_world_hv_test" \
# 	--data_paths ./results/Penicillin/ ./results/CarSideImpact/ ./results/VehicleSafety/ \
# 	--whitelists "$whitelist" "$whitelist" "$whitelist" \
#     --normalization_method "reference_point"


# whitelist="MOHOLLM (Gemini 2.0 Flash) (Context),MOHOLLM (gemma-2-9b-it-fast) (Context),MOHOLLM (gpt-oss-120b) (Context),MOHOLLM (Llama-3.3-70B-Instruct) (Context),MOHOLLM (Meta-Llama-3.1-8B-Instruct) (Context),MOHOLLM (Qwen3-32B) (Context),MOHOLLM (gpt-4o-mini) (Context)"
# # real world
# python3 ./plot/plot_hv_over_time_subplots.py \
# 	--benchmarks VehicleSafety  \
# 	--titles "VehicleSafety"  \
# 	--trials 50 \
#     --filename "real_world_hv_other_models_mohollm" \
# 	--data_paths ./results/VehicleSafety/  \
# 	--whitelists "$whitelist" \
#     --normalization_method "reference_point"

# whitelist="mohollm (Gemini 2.0 Flash),mohollm (gemma-2-9b-it) (Context),mohollm (gpt-oss-120b) (Context),mohollm (llama3-3-70B) (Context),mohollm (llama3-1-8B) (Context),mohollm (Qwen3-32B) (Context),mohollm (gpt-4o-mini) (Context)"
# # real world
# python3 ./plot/plot_hv_over_time_subplots.py \
# 	--benchmarks VehicleSafety  \
# 	--titles "VehicleSafety"  \
# 	--trials 50 \
#     --filename "real_world_hv_other_models_llm" \
# 	--data_paths ./results/VehicleSafety/  \
# 	--whitelists "$whitelist" \
#     --normalization_method "reference_point"

# whitelist="mohollm (Gemini 2.0 Flash),mohollm (gemma-2-9b-it) (Context),mohollm (gpt-oss-120b) (Context),mohollm (llama3-3-70B) (Context),mohollm (llama3-1-8B) (Context),mohollm (Qwen3-32B) (Context),mohollm (gpt-4o-mini) (Context),MOHOLLM (Gemini 2.0 Flash) (Context),MOHOLLM (gemma-2-9b-it-fast) (Context),MOHOLLM (gpt-oss-120b) (Context),MOHOLLM (Llama-3.3-70B-Instruct) (Context),MOHOLLM (Meta-Llama-3.1-8B-Instruct) (Context),MOHOLLM (Qwen3-32B) (Context),MOHOLLM (gpt-4o-mini) (Context)"
# # real world
# python3 ./plot/plot_hv_over_time_subplots.py \
# 	--benchmarks VehicleSafety  \
# 	--titles "VehicleSafety"  \
# 	--trials 50 \
#     --filename "real_world_hv_other_models_combined" \
# 	--data_paths ./results/VehicleSafety/  \
# 	--whitelists "$whitelist" \
#     --normalization_method "reference_point"


# Model comparison: MOHOLLM vs mohollm for each model (one subplot per model)
# python3 ./plot/plot_hv_over_time_subplots.py \
#     --benchmarks "VehicleSafety" "VehicleSafety" "VehicleSafety" "VehicleSafety" "VehicleSafety" "VehicleSafety" \
#     --titles "Gemma 2 9B" "GPT Oss 120B" "Llama 3.3 70B" "Llama 3.1 8B" "Qwen3 32B" "GPT 4o Mini" \
#     --trials 50 \
#     --filename "real_world_model_comparison_subplots_independent" \
#     --data_paths ./results/VehicleSafety/ ./results/VehicleSafety/ ./results/VehicleSafety/ ./results/VehicleSafety/ ./results/VehicleSafety/ ./results/VehicleSafety/ \
#     --whitelists "MOHOLLM (gemma-2-9b-it-fast) (Context),mohollm (gemma-2-9b-it) (Context)" \
#                  "MOHOLLM (gpt-oss-120b) (Context),mohollm (gpt-oss-120b) (Context)" \
#                  "MOHOLLM (Llama-3.3-70B-Instruct) (Context),mohollm (llama3-3-70B) (Context)" \
#                  "MOHOLLM (Meta-Llama-3.1-8B-Instruct) (Context),mohollm (llama3-1-8B) (Context)" \
#                  "MOHOLLM (Qwen3-32B) (Context),mohollm (Qwen3-32B) (Context)" \
#                  "MOHOLLM (gpt-4o-mini) (Context),mohollm (gpt-4o-mini) (Context)" \
#     --normalization_method "reference_point" \
#     --simplify_legend



# # Prompt ablations
# whitelist="MOHOLLM (Gemini 2.0 Flash),MOHOLLM (Gemini 2.0 Flash) (Context),MOHOLLM (Gemini 2.0 Flash) (minimal)"
# # real world
# python3 ./plot/plot_hv_over_time_subplots.py \
# 	--benchmarks Penicillin VehicleSafety CarSideImpact  \
# 	--titles "Penicillin" "VehicleSafety" "CarSideImpact"  \
# 	--trials 50 \
#     --filename "real_world_prompt_ablations" \
# 	--data_paths ./results/Penicillin/ ./results/VehicleSafety/ ./results/CarSideImpact/  \
# 	--whitelists "$whitelist" "$whitelist" "$whitelist" \
#     --normalization_method "reference_point"


# python ./plot/plot_hv_over_time_subplots.py \
#     --benchmarks "Penicillin" \
#     --titles "Penicillin" \
#     --data_path "./results/Penicillin/" \
#     --filename "penicillin_prompt_hv" \
#     --trials 50 \
#     --whitelist "mohollm (Gemini 2.0 Flash),mohollm (Gemini 2.0 Flash) - alpha_prompt=0.0,mohollm (Gemini 2.0 Flash) - alpha_prompt=0.5,mohollm (Gemini 2.0 Flash) - alpha_prompt=0.25,mohollm (Gemini 2.0 Flash) - alpha_prompt=0.75,mohollm (Gemini 2.0 Flash) - alpha_prompt=1.0" \
#     --normalization_method "reference_point"


# python ./plot/plot_hv_over_time_subplots.py \
#     --benchmarks "Penicillin" \
#     --titles "Penicillin" \
#     --data_path "./results/Penicillin/" \
#     --filename "penicillin_prompt_partitioning_hv" \
#     --trials 50 \
#     --whitelist "MOHOLLM (Gemini 2.0 Flash) - alpha_prompt=0.0,MOHOLLM (Gemini 2.0 Flash) - alpha_prompt=0.5,MOHOLLM (Gemini 2.0 Flash) - alpha_prompt=0.25,MOHOLLM (Gemini 2.0 Flash) - alpha_prompt=0.75,MOHOLLM (Gemini 2.0 Flash) - alpha_prompt=1.0,MOHOLLM (Gemini 2.0 Flash) (Context)" \
#     --normalization_method "reference_point"


# NB201
# benchmarks=("cifar10" "cifar100" "imagenet16")

# devices=(
#     "fpga_latency" "pixel3_latency" "raspi4_latency" "eyeriss_latency" "pixel2_latency"
#     "1080ti_1_latency" "1080ti_32_latency" "1080ti_256_latency"
#     "2080ti_1_latency" "2080ti_32_latency" "2080ti_256_latency"
#     "titanx_1_latency" "titanx_32_latency" "titanx_256_latency"
#     "titanxp_1_latency" "titanxp_32_latency" "titanxp_256_latency"
#     "titan_rtx_1_latency" "titan_rtx_32_latency" "titan_rtx_256_latency"
#     "essential_ph_1_latency" "gold_6226_latency" "gold_6240_latency"
#     "samsung_a50_latency" "samsung_s7_latency" "silver_4114_latency" "silver_4210r_latency"
# )

# # whitelist for NB201 plots
# whitelist="CTAEA,EHVI,GDE3,IBEA,MOEAD,MOHOLLM (Gemini 2.0 Flash),mohollm (Gemini 2.0 Flash),NSGA2,NSGA3,PESA2,qLogEHVI,RNSGA2,SMSEMOA,SPEA2,UNSGA3"

# # For NB201 we create one plot per benchmark, where each subplot is a device
# for benchmark in "${benchmarks[@]}"; do
#     echo "./results/NB201/$benchmark/*"
#     # build data paths for all devices for this benchmark
#     data_paths=()
#     for device in "${devices[@]}"; do
#         data_paths+=("./results/NB201/${benchmark}/${device}")
#     done

#     # nb201 device metrics (one per subplot)
#     nb201_metrics=("${devices[@]}")

#     # whitelists: same whitelist for each subplot
#     whitelists_args=()
#     for device in "${devices[@]}"; do
#         whitelists_args+=("$whitelist")
#     done

#     # call the plotting script with one subplot per device
#     python ./plot/plot_hv_over_time_subplots.py \
#         --benchmarks "${devices[@]}" \
#         --titles "${devices[@]}" \
#         --trials 50 \
#         --filename "NB201/${benchmark}" \
#         --data_paths "${data_paths[@]}" \
#         --normalization_method "nb201" \
#         --nb201_device_metrics "${nb201_metrics[@]}" \
#         --whitelists "${whitelists_args[@]}"
# done


# # Poloni beta ablations
# python ./plot/plot_hv_over_time_subplots.py \
#     --benchmarks "Poloni" \
#     --titles "Poloni" \
#     --data_path "./results/Poloni/" \
#     --filename "poloni_beta_ablations_llm" \
#     --trials 75 \
#     --whitelist "mohollm (Gemini 2.0 Flash) (Beta=0.0),mohollm (Gemini 2.0 Flash) (Beta=0.25),mohollm (Gemini 2.0 Flash) (Beta=0.5),mohollm (Gemini 2.0 Flash) (Beta=0.75),mohollm (Gemini 2.0 Flash) (Beta=1.0)" \
#     --normalization_method "reference_point"


# python ./plot/plot_hv_over_time_subplots.py \
#     --benchmarks "Poloni" \
#     --titles "Poloni" \
#     --data_path "./results/Poloni/" \
#     --filename "poloni_beta_ablations_mohollm" \
#     --trials 75 \
#     --whitelist "MOHOLLM (Gemini 2.0 Flash) (Beta=0.0),MOHOLLM (Gemini 2.0 Flash) (Beta=0.25),MOHOLLM (Gemini 2.0 Flash) (Beta=0.5),MOHOLLM (Gemini 2.0 Flash) (Beta=0.75),MOHOLLM (Gemini 2.0 Flash) (Beta=1.0)" \
#     --normalization_method "reference_point"

# python ./plot/plot_hv_over_time_subplots.py \
#     --benchmarks "Poloni" \
#     --titles "Poloni" \
#     --data_path "./results/Poloni/" \
#     --filename "poloni_beta_ablations_combined" \
#     --trials 75 \
#     --whitelist "MOHOLLM (Gemini 2.0 Flash) (Beta=0.0),MOHOLLM (Gemini 2.0 Flash) (Beta=0.25),MOHOLLM (Gemini 2.0 Flash) (Beta=0.5),MOHOLLM (Gemini 2.0 Flash) (Beta=0.75),MOHOLLM (Gemini 2.0 Flash) (Beta=1.0),mohollm (Gemini 2.0 Flash) (Beta=0.0),mohollm (Gemini 2.0 Flash) (Beta=0.25),mohollm (Gemini 2.0 Flash) (Beta=0.5),mohollm (Gemini 2.0 Flash) (Beta=0.75),mohollm (Gemini 2.0 Flash) (Beta=1.0)" \
#     --normalization_method "reference_point"


############### PAPER ###################

# whitelist="EHVI,GDE3,MOEAD,NSGA2,NSGA3,qLogEHVI,mohollm (Gemini 2.0 Flash),MOHOLLM (Gemini 2.0 Flash) (Context)"

# python3 ./plot/plot_hv_over_time_subplots_real.py \
# 	--benchmarks Penicillin CarSideImpact VehicleSafety \
# 	--titles "Penicillin" "CarSideImpact" "VehicleSafety" \
# 	--trials 50 \
#     --filename "real_world_hv_paper" \
# 	--data_paths ./results/Penicillin/ ./results/CarSideImpact/ ./results/VehicleSafety/ \
# 	--whitelists "$whitelist" "$whitelist" "$whitelist" \
#     --normalization_method "reference_point"

# python3 ./plot/plot_hv_over_time_subplots.py \
#         --benchmarks DTLZ1 BraninCurrin ChankongHaimes SchafferN1 Kursawe \
#         --titles "DTLZ1" "BraninCurrin" "ChankongHaimes" "SchafferN1" "Kursawe" \
#         --trials 50 \
#         --filename "synthetic_hv_paper_1" \
#         --data_paths ./results/DTLZ1/ ./results/BraninCurrin/ ./results/ChankongHaimes/ ./results/SchafferN1/ ./results/Kursawe/ \
#         --whitelists "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist"\
#         --normalization_method "reference_point"


whitelist="CTAEA,EHVI,GDE3,IBEA,MOEAD,NSGA2,NSGA3,PESA2,qLogEHVI,RNSGA2,SMSEMOA,SPEA2,UNSGA3,mohollm (Gemini 2.0 Flash),MOHOLLM (Gemini 2.0 Flash) (Context)"

# synthetic
# python3 ./plot/plot_hv_over_time_subplots.py \
# 	--benchmarks DTLZ1 DTLZ2 DTLZ3 BraninCurrin ChankongHaimes GMM Poloni SchafferN1 SchafferN2 TestFunction4 ToyRobust Kursawe \
# 	--titles "DTLZ1" "DTLZ2" "DTLZ3" "BraninCurrin" "ChankongHaimes" "GMM" "Poloni" "SchafferN1" "SchafferN2" "TestFunction4" "ToyRobust" "Kursawe" \
# 	--trials 50 \
#     --filename "synthetic_hv_paper_all" \
# 	--data_paths ./results/DTLZ1/ ./results/DTLZ2/ ./results/DTLZ3/ ./results/BraninCurrin/ ./results/ChankongHaimes/ ./results/GMM/ ./results/Poloni/ ./results/SchafferN1/ ./results/SchafferN2/ ./results/TestFunction4/ ./results/ToyRobust/ ./results/Kursawe/ \
# 	--whitelists "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" "$whitelist" \
#     --normalization_method "reference_point"

# python3 ./plot/plot_hv_over_time_subplots_real.py \
# 	--benchmarks Penicillin CarSideImpact VehicleSafety \
# 	--titles "Penicillin" "CarSideImpact" "VehicleSafety" \
# 	--trials 50 \
#     --filename "real_world_hv_paper_all" \
# 	--data_paths ./results/Penicillin/ ./results/CarSideImpact/ ./results/VehicleSafety/ \
# 	--whitelists "$whitelist" "$whitelist" "$whitelist" \
#     --normalization_method "reference_point"


# whitelist="MOHOLLM (Gemini 2.0 Flash) (Context),MOHOLLM (gemma-2-9b-it-fast) (Context),MOHOLLM (gpt-oss-120b) (Context),MOHOLLM (Llama-3.3-70B-Instruct) (Context),MOHOLLM (Meta-Llama-3.1-8B-Instruct) (Context),MOHOLLM (Qwen3-32B) (Context),MOHOLLM (gpt-4o-mini) (Context)"
# # real world
# python3 ./plot/plot_hv_over_time_subplots_real.py \
# 	--benchmarks VehicleSafety  \
# 	--titles "VehicleSafety"  \
# 	--trials 50 \
#     --filename "real_world_hv_other_models_mohollm_paper" \
# 	--data_paths ./results/VehicleSafety/  \
# 	--whitelists "$whitelist" \
#     --normalization_method "reference_point"


# whitelist="mohollm (Gemini 2.0 Flash),mohollm (gemma-2-9b-it) (Context),mohollm (gpt-oss-120b) (Context),mohollm (llama3-3-70B) (Context),mohollm (llama3-1-8B) (Context),mohollm (Qwen3-32B) (Context),mohollm (gpt-4o-mini) (Context),MOHOLLM (Gemini 2.0 Flash) (Context),MOHOLLM (gemma-2-9b-it-fast) (Context),MOHOLLM (gpt-oss-120b) (Context),MOHOLLM (Llama-3.3-70B-Instruct) (Context),MOHOLLM (Meta-Llama-3.1-8B-Instruct) (Context),MOHOLLM (Qwen3-32B) (Context),MOHOLLM (gpt-4o-mini) (Context)"
# # real world
# python3 ./plot/plot_hv_over_time_subplots_real.py \
# 	--benchmarks VehicleSafety  \
# 	--titles "VehicleSafety"  \
# 	--trials 50 \
#     --filename "real_world_hv_other_models_combined_paper" \
# 	--data_paths ./results/VehicleSafety/  \
# 	--whitelists "$whitelist" \
#     --normalization_method "reference_point"


# # Prompt ablations
# whitelist="MOHOLLM (Gemini 2.0 Flash),MOHOLLM (Gemini 2.0 Flash) (Context),MOHOLLM (Gemini 2.0 Flash) (minimal)"
# # real world
# python3 ./plot/plot_hv_over_time_subplots_real.py \
# 	--benchmarks Penicillin CarSideImpact VehicleSafety \
# 	--titles "Penicillin" "CarSideImpact" "VehicleSafety"  \
# 	--trials 50 \
#     --filename "real_world_prompt_ablations_paper" \
# 	--data_paths ./results/Penicillin/ ./results/CarSideImpact/ ./results/VehicleSafety/  \
# 	--whitelists "$whitelist" "$whitelist" "$whitelist" \
#     --normalization_method "reference_point"

# # Prompt ablations
# python3 ./plot/plot_hv_over_time_subplots_real.py \
# 	--benchmarks CarSideImpact  \
# 	--titles "CarSideImpact"  \
# 	--trials 50 \
#     --filename "real_world_prompt_ablations_single_paper" \
# 	--data_paths ./results/CarSideImpact/  \
# 	--whitelists "$whitelist" \
#     --normalization_method "reference_point"

# Candidate sampler
#whitelist="MOHOLLM (Gemini 2.0 Flash) (Context),mohollm (Gemini 2.0 Flash),RS (KD-Partitioning) + LLM Surrogate (Gemini 2.0 Flash) (Context),RS + LLM Surrogate (Gemini 2.0 Flash) (Context),"
# python3 ./plot/plot_hv_over_time_subplots_real.py \
# 	--benchmarks Penicillin CarSideImpact VehicleSafety \
# 	--titles "Penicillin" "CarSideImpact" "VehicleSafety" \
# 	--trials 50 \
#     --filename "real_world_ablations_components_candidate_sampler_paper" \
# 	--data_paths ./results/Penicillin/ ./results/CarSideImpact/ ./results/VehicleSafety/ \
# 	--whitelists "$whitelist" "$whitelist" "$whitelist" \
#     --normalization_method "reference_point"

# python3 ./plot/plot_hv_over_time_subplots_real.py \
# 	--benchmarks CarSideImpact \
# 	--titles "CarSideImpact" \
# 	--trials 50 \
#     --filename "real_world_ablations_components_candidate_sampler_single_paper" \
# 	--data_paths ./results/CarSideImpact/ \
# 	--whitelists "$whitelist" \
#     --normalization_method "reference_point"

# Surrogate model
#whitelist="MOHOLLM (Gemini 2.0 Flash) (Context),MOHOLLM + TabPFN (Gemini 2.0 Flash) (Context),MOHOLLM + GP (BoTorch) (Gemini 2.0 Flash) (Context)"
# python3 ./plot/plot_hv_over_time_subplots_real.py \
# 	--benchmarks Penicillin CarSideImpact VehicleSafety \
# 	--titles "Penicillin" "CarSideImpact" "VehicleSafety" \
# 	--trials 50 \
#     --filename "real_world_ablations_components_surrogate_model_paper" \
# 	--data_paths ./results/Penicillin/ ./results/CarSideImpact/ ./results/VehicleSafety/ \
# 	--whitelists "$whitelist" "$whitelist" "$whitelist" \
#     --normalization_method "reference_point"

# python3 ./plot/plot_hv_over_time_subplots_real.py \
# 	--benchmarks CarSideImpact \
# 	--titles "CarSideImpact" \
# 	--trials 50 \
#     --filename "real_world_ablations_components_surrogate_model_single_paper" \
# 	--data_paths ./results/CarSideImpact/ \
# 	--whitelists "$whitelist" \
#     --normalization_method "reference_point"



# 1. Leaf size
whitelist="MOHOLLM (Gemini 2.0 Flash) (Context),MOHOLLM (Gemini 2.0 Flash) (Context) - m0=1,MOHOLLM (Gemini 2.0 Flash) (Context) - m0=3,MOHOLLM (Gemini 2.0 Flash) (Context) - m0=10"
python3 ./plot/plot_hv_over_time_subplots_real.py \
	--benchmarks Penicillin CarSideImpact VehicleSafety \
	--titles "Penicillin" "CarSideImpact" "VehicleSafety" \
	--trials 35 \
    --filename "real_world_ablations_leaf_size_paper" \
	--data_paths ./results/Penicillin/ ./results/CarSideImpact/ ./results/VehicleSafety/ \
	--whitelists "$whitelist" "$whitelist" "$whitelist" \
    --normalization_method "reference_point" \
    --num_seeds 5 \
    --ablations

# 2. alpha_max
whitelist="MOHOLLM (Gemini 2.0 Flash) (Context),MOHOLLM (Gemini 2.0 Flash) (Context) - alpha_max=0,MOHOLLM (Gemini 2.0 Flash) (Context) - alpha_max=0.4,MOHOLLM (Gemini 2.0 Flash) (Context) - alpha_max=0.8,MOHOLLM (Gemini 2.0 Flash) (Context) - alpha_max=2.0"
python3 ./plot/plot_hv_over_time_subplots_real.py \
	--benchmarks Penicillin CarSideImpact VehicleSafety \
	--titles "Penicillin" "CarSideImpact" "VehicleSafety" \
	--trials 35 \
    --filename "real_world_ablations_alpha_max_paper" \
	--data_paths ./results/Penicillin/ ./results/CarSideImpact/ ./results/VehicleSafety/ \
	--whitelists "$whitelist" "$whitelist" "$whitelist" \
    --normalization_method "reference_point" \
    --num_seeds 5 \
    --ablations


# 3. Candidates per request
whitelist="MOHOLLM (Gemini 2.0 Flash) (Context),MOHOLLM (Gemini 2.0 Flash) (Context) - candidates_per_request=1,MOHOLLM (Gemini 2.0 Flash) (Context) - candidates_per_request=3,MOHOLLM (Gemini 2.0 Flash) (Context) - candidates_per_request=7"
python3 ./plot/plot_hv_over_time_subplots_real.py \
	--benchmarks Penicillin CarSideImpact VehicleSafety \
	--titles "Penicillin" "CarSideImpact" "VehicleSafety" \
	--trials 35 \
    --filename "real_world_ablations_candidates_per_request_paper" \
	--data_paths ./results/Penicillin/ ./results/CarSideImpact/ ./results/VehicleSafety/ \
	--whitelists "$whitelist" "$whitelist" "$whitelist" \
    --normalization_method "reference_point" \
    --num_seeds 5 \
    --ablations

# 4. Partitions per trial
whitelist="MOHOLLM (Gemini 2.0 Flash) (Context),MOHOLLM (Gemini 2.0 Flash) (Context) - partitions_per_trial=1,MOHOLLM (Gemini 2.0 Flash) (Context) - partitions_per_trial=3,MOHOLLM (Gemini 2.0 Flash) (Context) - partitions_per_trial=7"
python3 ./plot/plot_hv_over_time_subplots_real.py \
	--benchmarks Penicillin CarSideImpact VehicleSafety \
	--titles "Penicillin" "CarSideImpact" "VehicleSafety" \
	--trials 35 \
    --filename "real_world_ablations_partitions_per_trial_paper" \
	--data_paths ./results/Penicillin/ ./results/CarSideImpact/ ./results/VehicleSafety/ \
	--whitelists "$whitelist" "$whitelist" "$whitelist" \
    --normalization_method "reference_point" \
    --num_seeds 5 \
    --ablations