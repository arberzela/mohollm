script="plot_fvals_paper.py"




# blacklist="ASHA,ASHABORE,ASHABCQR,BOHB,growth-rate,LLMKD-alpha-0.2,LLMKD-alpha-0.5,LLMKD-alpha-0.7,LLMKD-alpha-1.0,TPE,LLM-llama3.1-8b-llm,LLMKD-llama3.1-8b,LLM-llama3.3-70b-llm,LLMKD-llama3.3-70b"

# python ./plot/$script \
#        --benchmark "CarSideImpact" \
#        --title "CarSideImpact" \
#        --columns "F1" \
#        --trials 50 \
#        --data_path ./results/car_side_impact_single_objective_1/ \
#        --filename "car_side_impact_single_objective_1" \
#        --normalization_method "none" \
#        --whitelist "LLM,LLMKD,LLM-llama3.1-8b-llm,LLMKD-llama3.1-8b,LLM-llama3.3-70b-llm,LLMKD-llama3.3-70b,LLMKD-qwen-30b,LLM-qwen-30b" \
#        --use_log_scale "False"
    #    --y_lim "12" "14.5"

# python ./plot/$script \
#        --benchmark "fcnet-slice" \
#        --title "Slice" \
#        --columns "F1" \
#        --trials 50 \
#        --data_path ./results/fcnet-slice/ \
#        --filename "fcnet-slice-ablations" \
#        --normalization_method "none" \
#        --whitelist "LLM,LLMKD,LLMKD-exploitation,LLMKD-exploration,LLMKD-ucb1,LLMKD-uniform_region_sampling" \
#        --use_log_scale "False" \
#        --y_lim "-0.0010" "-0.0001"

# python ./plot/$script \
#        --benchmark "vehicle_safety" \
#        --title "Vehicle Safety" \
#        --columns "F1" \
#        --trials 50 \
#        --data_path ./results/vehicle_safety_single_objective/ \
#        --filename "vehicle_safety_single_objective_ablations" \
#        --normalization_method "none" \
#        --whitelist "LLM,LLMKD,LLMKD-exploitation,LLMKD-exploration,LLMKD-ucb1,LLMKD-uniform_region_sampling" \
#        --use_log_scale "False" \
#        --y_lim "-1675" "-1660"

# python ./plot/$script \
#     --benchmark "vehicle_safety" \
#     --title "Vehicle Safety" \
#     --columns "F1" \
#     --trials 50 \
#     --data_path ./results/vehicle_safety_single_objective/ \
#     --filename "results/vehicle_safety_single_objective" \
#     --normalization_method "none" \
#     --blacklist "$blacklist" \
#     --use_log_scale "False"
#     # --y_lim "12" "14.5"


python ./plot/plot_fvals_paper.py \
       --benchmark "vehicle_safety_noise" \
       --title "Vehicle Safety Noise" \
       --columns "F1" \
       --trials 50 \
       --data_path ./results/vehicle_safety_noise_single_objective/ \
       --filename "vehicle_safety_noise_single_objective" \
       --normalization_method "none" \
       --whitelist "LLM,LLMKD,BORE,BOTorch,CQR,REA,RS,TPE" \
       --use_log_scale "False" \
       --y_lim "-1675" "-1660"


# python ./plot/$script \
#        --benchmark "fcnet-slice" \
#        --title "Slice" \
#        --columns "F1" \
#        --trials 50 \
#        --data_path ./results/fcnet-slice/ \
#        --filename "fcnet-slice-ablations-context" \
#        --normalization_method "none" \
#        --whitelist "LLM,LLMKD,LLM (no-context),LLMKD (no-context)," \
#        --use_log_scale "False" \
#        --y_lim "-0.0010" "-0.0001"



# # benchmarks=("nas201-cifar10" "nas201-cifar100" "nas201-ImageNet16-120")
# # benchmarks=("fcnet-naval" "fcnet-parkinsons" "fcnet-protein" "fcnet-slice" "tabrepo-CatBoost-2dplanes" "tabrepo-CatBoost-Airlines-DepDelay-10M" "tabrepo-CatBoost-Allstate-Claims-Severity" "tabrepo-CatBoost-Amazon-employee-access" "tabrepo-CatBoost-APSFailure" "tabrepo-CatBoost-Australian" "tabrepo-CatBoost-Bioresponse" "tabrepo-CatBoost-Brazilian-houses" "tabrepo-CatBoost-Buzzinsocialmedia-Twitter" "tabrepo-CatBoost-CIFAR-10" "tabrepo-RandomForest-2dplanes" "tabrepo-RandomForest-Airlines-DepDelay-10M" "tabrepo-RandomForest-Allstate-Claims-Severity" "tabrepo-RandomForest-Amazon-employee-access" "tabrepo-RandomForest-APSFailure" "tabrepo-RandomForest-Australian" "tabrepo-RandomForest-Bioresponse" "tabrepo-RandomForest-Brazilian-houses" "tabrepo-RandomForest-Buzzinsocialmedia-Twitter" "tabrepo-RandomForest-CIFAR-10")
# # for benchmark in "${benchmarks[@]}"g
# # do
# #     python ./plot/$script \
# #         --benchmark $benchmark \
# #         --title "Fvals ($benchmark)" \
# #         --columns "F1" \
# #         --trials 100 \
# #         --data_path ./results/SyneTune/$benchmark/ \
# #         --filename "fvals_$benchmark" \
# #         --normalization_method "none" \
# #         --blacklist "$blacklist" \
# #         --use_log_scale "False"
# # done

# # --- THE SIX FUNCTIONS ---
# #python ./plot/$script \
# #        --benchmark "ackley"  \
# #        --title "Ackley" \
# #        --columns "F1" \
# #        --trials 100 \
# #        --data_path ./results/SyneTune/ackley/ \
# #        --filename "fvals_ackley" \
# #        --normalization_method "none" \
# #        --blacklist "$blacklist" \
# #        --use_log_scale "False"
# #
# python ./plot/$script \
#        --benchmark "hartmann3"  \
#        --title "Hartmann3" \
#        --columns "F1" \
#        --trials 50 \
#        --data_path ./results/hartmann3/ \
#        --filename "fvals_hartmann3" \
#        --normalization_method "none" \
#        --blacklist "$blacklist" \
#        --use_log_scale "False"

# python ./plot/$script \
#        --benchmark "hartmann6"  \
#        --title "Hartmann6" \
#        --columns "F1" \
#        --trials 50 \
#        --data_path ./results/hartmann6/ \
#        --filename "fvals_hartmann6" \
#        --normalization_method "none" \
#        --blacklist "$blacklist" \
#        --use_log_scale "False"

# python ./plot/$script \
#     --benchmark "levy"  \
#     --title "Levy" \
#     --columns "F1" \
#     --trials 50 \
#     --data_path ./results/levy/ \
#     --filename "fvals_levy" \
#     --normalization_method "none" \
#     --blacklist "$blacklist" \
#     --use_log_scale "False"

# python ./plot/$script \
#        --benchmark "rosenbrock"  \
#        --title "Rosenbrock" \
#        --columns "F1" \
#        --trials 50 \
#        --data_path ./results/rosenbrock/ \
#        --filename "fvals_rosenbrock" \
#        --normalization_method "none" \
#        --blacklist "$blacklist" \
#        --use_log_scale "False" \
#        --y_lim "-2000" "20"


# python ./plot/$script \
#        --benchmark "rastrigin"  \
#        --title "Rastrigin" \
#        --columns "F1" \
#        --trials 50 \
#        --data_path ./results/rastrigin/ \
#        --filename "fvals_rastrigin" \
#        --normalization_method "none" \
#        --blacklist "$blacklist" \
#        --use_log_scale "False"



# ## # --- THE SIX FUNCTIONS END ---
# #
# ## # --- FCNET ---
# python ./plot/$script \
#        --benchmark "fcnet-naval" \
#        --title "Naval" \
#        --columns "F1" \
#        --trials 50 \
#        --data_path ./results/fcnet-naval/ \
#        --filename "fvals_fcnet-naval" \
#        --normalization_method "none" \
#        --blacklist "$blacklist" \
#        --use_log_scale "False" \
#        --y_lim "-0.005" "0.0005"


# python ./plot/$script \
#        --benchmark "fcnet-parkinsons" \
#        --title "Parkinsons" \
#        --columns "F1" \
#        --trials 50 \
#        --data_path ./results/fcnet-parkinsons/ \
#        --filename "fvals_fcnet-parkinsons" \
#        --normalization_method "none" \
#        --blacklist "$blacklist" \
#        --use_log_scale "False" \
#        --y_lim "-0.05" "0.0005"


# python ./plot/$script \
#        --benchmark "fcnet-protein" \
#        --title "Protein" \
#        --columns "F1" \
#        --trials 50 \
#        --data_path ./results/fcnet-protein/ \
#        --filename "fvals_fcnet-protein" \
#        --normalization_method "none" \
#        --blacklist "$blacklist" \
#        --use_log_scale "False" \
#        --y_lim "-0.30" "-0.22"

# python ./plot/$script \
#        --benchmark "fcnet-slice" \
#        --title "Slice" \
#        --columns "F1" \
#        --trials 50 \
#        --data_path ./results/fcnet-slice/ \
#        --filename "fvals_fcnet-slice" \
#        --normalization_method "none" \
#        --blacklist "$blacklist" \
#        --use_log_scale "False" \
#        --y_lim "-0.0010" "-0.0001"


# python ./plot/plot_fvals_paper.py \
#        --benchmark "fcnet-slice" \
#        --title "Slice" \
#        --columns "F1" \
#        --trials 50 \
#        --data_path ./results/fcnet-slice/ \
#        --filename "fvals_fcnet-slice_ablation" \
#        --normalization_method "none" \
#        --whitelist "LLMKD,LLMKD-exploitation,LLMKD-exploration,LLMKD-ucb1,LLMKD-uniform_region_sampling" \
#        --use_log_scale "False" \
#        --y_lim "-0.0010" "-0.0001"







# python ./plot/$script \
#        --benchmark "penicillin" \
#        --title "Penicillin" \
#        --columns "F1" \
#        --trials 50 \
#        --data_path ./results/penicillin_single_objective/ \
#        --filename "penicillin_single_objective" \
#        --normalization_method "none" \
#        --blacklist "$blacklist" \
#        --use_log_scale "False" \
#        --y_lim "12" "14.5"


# python ./plot/$script \
#     --benchmark "car_side_impact" \
#     --title "Car Side Impact" \
#     --columns "F1" \
#     --trials 50 \
#     --data_path ./results/car_side_impact_single_objective/ \
#     --filename "car_side_impact_single_objective" \
#     --normalization_method "none" \
#     --blacklist "$blacklist" \
#     --use_log_scale "False"
#     # --y_lim "12" "14.5"

# python ./plot/$script \
#     --benchmark "vehicle_safety" \
#     --title "Vehicle Safety" \
#     --columns "F1" \
#     --trials 50 \
#     --data_path ./results/vehicle_safety_single_objective/ \
#     --filename "results/vehicle_safety_single_objective" \
#     --normalization_method "none" \
#     --blacklist "$blacklist" \
#     --use_log_scale "False"
#     # --y_lim "12" "14.5"

# python ./plot/$script \
#     --benchmark "vehicle_safety" \
#     --title "Vehicle Safety" \
#     --columns "F1" \
#     --trials 50 \
#     --data_path ./results/vehicle_safety_single_objective/ \
#     --filename "results/vehicle_safety_single_objective_llm_comparision" \
#     --normalization_method "none" \
#     --whitelist "LLM,LLMKD,LLM-llama3.1-8b-llm,LLMKD-llama3.1-8b,LLM-llama3.3-70b-llm,LLMKD-llama3.3-70b,LLMKD-qwen-30b,LLM-qwen-30b" \
#     --use_log_scale "False"
#     # --y_lim "12" "14.5"

python ./plot/$script \
    --benchmark "Parkinsons" \
    --title "Parkinsons" \
    --columns "F1" \
    --trials 50 \
    --data_path ./results/fcnet_parkinsons_1/ \
    --filename "results/fcnet_parkinsons_1_model_ablations" \
    --normalization_method "none" \
    --whitelist "LLM,LLMKD,LLM-llama3.1-8b-llm,LLMKD-llama3.1-8b,LLM-llama3.3-70b-llm,LLMKD-llama3.3-70b,LLMKD-qwen-30b,LLM-qwen-30b" \
    --use_log_scale "False" \
    --y_lim "-0.06" "0.002"


# ## --- FCNET END---
# #
# ## --- FCNET Alpha comparision ---

# #blacklist="ASHA,ASHABORE,ASHABCQR,BOHB,growth-rate,BORE,BOTorch,CQR,RS,TPE,RE"
# blacklist="ASHA,ASHABORE,ASHABCQR,BOHB,growth-rate,BORE,BOTorch,CQR,RS,TPE,RE,LLM-25,LLMKD-k-1,LLMKD-k-3,LLMKD-k-5,LLMKD-k-7,LLMKD-k-10,LLMKD-m0-0.5*d,LLMKD-m0-1*d,LLMKD-m0-0.25*d,LLMKD-m0-1,LLMKD-1-partition-per-trial,LLMKD-3-partition-per-trial,LLMKD-5-partition-per-trial,LLMKD-7-partition-per-trial,REA,LLM,LLMKD,RS + KD-Tree"
# python ./plot/$script \
#         --benchmark "fcnet-naval" \
#         --title "Naval" \
#         --columns "F1" \
#         --trials 50 \
#         --data_path ./results/fcnet-naval/ \
#         --filename "alpha-comparision/fvals_fcnet-naval" \
#         --normalization_method "none" \
#         --blacklist "$blacklist" \
#         --use_log_scale "False" \
#         --y_lim "-0.00010" "-0.00002"

# blacklist="ASHA,ASHABORE,ASHABCQR,BOHB,growth-rate,BORE,BOTorch,CQR,RS,TPE,RE,LLM-25,LLMKD-k-1,LLMKD-k-3,LLMKD-k-5,LLMKD-k-7,LLMKD-k-10,LLMKD-m0-0.5*d,LLMKD-m0-1*d,LLMKD-m0-0.25*d,LLMKD-m0-1,LLMKD-1-partition-per-trial,LLMKD-3-partition-per-trial,LLMKD-5-partition-per-trial,LLMKD-7-partition-per-trial,REA,LLM,LLMKD,RS + KD-Tree"
# python ./plot/$script \
#         --benchmark "fcnet-parkinsons" \
#         --title "Parkinsons" \
#         --columns "F1" \
#         --trials 50 \
#         --data_path ./results/fcnet-parkinsons/ \
#         --filename "alpha-comparision/fvals_fcnet-parkinsons" \
#         --normalization_method "none" \
#         --blacklist "$blacklist" \
#         --use_log_scale "False" \
#         --y_lim "-0.025" "-0.005"

# blacklist="ASHA,ASHABORE,ASHABCQR,BOHB,growth-rate,BORE,BOTorch,CQR,RS,TPE,RE,LLM-25,LLMKD-k-1,LLMKD-k-3,LLMKD-k-5,LLMKD-k-7,LLMKD-k-10,LLMKD-m0-0.5*d,LLMKD-m0-1*d,LLMKD-m0-0.25*d,LLMKD-m0-1,LLMKD-1-partition-per-trial,LLMKD-3-partition-per-trial,LLMKD-5-partition-per-trial,LLMKD-7-partition-per-trial,REA,LLM,LLMKD,RS + KD-Tree"
# python ./plot/$script \
#         --benchmark "fcnet-protein" \
#         --title "Protein" \
#         --columns "F1" \
#         --trials 50 \
#         --data_path ./results/fcnet-protein/ \
#         --filename "alpha-comparision/fvals_fcnet-protein" \
#         --normalization_method "none" \
#         --blacklist "$blacklist" \
#         --use_log_scale "False" \
#         --y_lim "-0.25" "-0.22"
# blacklist="ASHA,ASHABORE,ASHABCQR,BOHB,growth-rate,BORE,BOTorch,CQR,RS,TPE,RE,LLM-25,LLMKD-k-1,LLMKD-k-3,LLMKD-k-5,LLMKD-k-7,LLMKD-k-10,LLMKD-m0-0.5*d,LLMKD-m0-1*d,LLMKD-m0-0.25*d,LLMKD-m0-1,LLMKD-1-partition-per-trial,LLMKD-3-partition-per-trial,LLMKD-5-partition-per-trial,LLMKD-7-partition-per-trial,REA,LLM,LLMKD,RS + KD-Tree"
# python ./plot/$script \
#         --benchmark "fcnet-slice" \
#         --title "Slice" \
#         --columns "F1" \
#         --trials 50 \
#         --data_path ./results/fcnet-slice/ \
#         --filename "alpha-comparision/fvals_fcnet-slice" \
#         --normalization_method "none" \
#         --blacklist "$blacklist" \
#         --use_log_scale "False" \
#         --y_lim "-0.0004" "-0.0001"
# # #
# ## --- FCNET Alpha comparision END ---